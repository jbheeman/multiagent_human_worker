from google.cloud import bigquery
import pandas as pd
import numpy as np
from datetime import datetime, timezone

import json

from pyparsing import line
import os

# If Colab doesn't auto-detect your default project, set it explicitly:
PROJECT_ID = "personageneration-485120"
client = bigquery.Client(project=PROJECT_ID)
table_id = "gdelt-bq.gdeltv2.gkg_partitioned"
table = client.get_table(table_id)
[(f.name, f.field_type) for f in table.schema]



def unix_to_datatime(timestamp_ms):
    return datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)


def get_schwartz_vector(date_str, client):
    """
    Robust GDELT Query that handles sparse data and provides fallbacks.
    Targeting 2015-2023 window.
    """
    
    # We broaden the query to capture LIWC (c5) as a backup for Moral Foundations (c25)
    sql = """
    SELECT
      -- --- EGO VALUES (General Inquirer - Usually Reliable) ---
      AVG(SAFE_CAST(REGEXP_EXTRACT(GCAM, r'c2.156:([0-9.]+)') AS FLOAT64)) as val_power,
      AVG(SAFE_CAST(REGEXP_EXTRACT(GCAM, r'c5.7:([0-9.]+)') AS FLOAT64))   as val_achievement,
      AVG(SAFE_CAST(REGEXP_EXTRACT(GCAM, r'c2.152:([0-9.]+)') AS FLOAT64)) as val_hedonism,
      AVG(SAFE_CAST(REGEXP_EXTRACT(GCAM, r'c2.17:([0-9.]+)') AS FLOAT64))  as val_stimulation,

      -- --- SOCIAL VALUES (Primary: Moral Foundations c25) ---
      AVG(SAFE_CAST(REGEXP_EXTRACT(GCAM, r'c25.4:([0-9.]+)') AS FLOAT64))  as val_universalism, 
      AVG(SAFE_CAST(REGEXP_EXTRACT(GCAM, r'c25.6:([0-9.]+)') AS FLOAT64))  as val_benevolence,  
      AVG(SAFE_CAST(REGEXP_EXTRACT(GCAM, r'c25.10:([0-9.]+)') AS FLOAT64)) as val_tradition,    
      AVG(SAFE_CAST(REGEXP_EXTRACT(GCAM, r'c25.8:([0-9.]+)') AS FLOAT64))  as val_conformity,   
      AVG(SAFE_CAST(REGEXP_EXTRACT(GCAM, r'c25.2:([0-9.]+)') AS FLOAT64))  as val_security,
      
      -- --- BACKUP SIGNALS (LIWC c1/c5 - Always Present) ---
      -- Used if c25 returns NULL
      AVG(SAFE_CAST(REGEXP_EXTRACT(GCAM, r'c5.31:([0-9.]+)') AS FLOAT64))  as backup_social, -- Social processes
      AVG(SAFE_CAST(REGEXP_EXTRACT(GCAM, r'c5.46:([0-9.]+)') AS FLOAT64))  as backup_risk,   -- Risk/Danger (Security)
      AVG(SAFE_CAST(REGEXP_EXTRACT(GCAM, r'c5.10:([0-9.]+)') AS FLOAT64))  as backup_anger   -- Anger (Conformity/Tradition)

    FROM `gdelt-bq.gdeltv2.gkg_partitioned`
    -- 7-Day Window centered on target
    WHERE _PARTITIONDATE BETWEEN DATE_SUB(DATE(@target_date), INTERVAL 3 DAY) 
                             AND DATE_ADD(DATE(@target_date), INTERVAL 3 DAY)
      AND V2Locations LIKE '%#US#%'
      AND GCAM IS NOT NULL
    """
    
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("target_date", "DATE", date_str)
        ]
    )
    
    try:
        results = list(client.query(sql, job_config=job_config).result())
        if not results:
            # Fatal error: No GDELT data at all for this week (Rare)
            return get_flat_prior()
        row = results[0]
    except Exception as e:
        print(f"Query Failed: {e}")
        return get_flat_prior()

    # --- DEFENSIVE DATA MAPPING ---
    
    # Helper: "Zero if None"
    def z(val): return val if val is not None else 0.0

    # 1. Ego Values (Direct Map)
    scores = {
        "POWER":       z(row.val_power),
        "ACHIEVEMENT": z(row.val_achievement),
        "HEDONISM":    z(row.val_hedonism),
        "STIMULATION": z(row.val_stimulation)
    }

    # 2. Social Values (With Fallback Logic)
    # Check if MFD is missing (i.e., Security is 0.0)
    mfd_present = (z(row.val_security) + z(row.val_universalism)) > 0.001

    if mfd_present:
        # Use High-Quality MFD signals
        scores["UNIVERSALISM"] = z(row.val_universalism)
        scores["BENEVOLENCE"]  = z(row.val_benevolence)
        scores["TRADITION"]    = z(row.val_tradition)
        scores["CONFORMITY"]   = z(row.val_conformity)
        scores["SECURITY"]     = z(row.val_security)
    else:
        # FALLBACK: Use LIWC proxies if MFD is broken (Common in 2015)
        # print("Warning: Using LIWC Fallback for date", date_str)
        scores["UNIVERSALISM"] = z(row.backup_social) * 0.5
        scores["BENEVOLENCE"]  = z(row.backup_social) * 0.5
        scores["TRADITION"]    = z(row.backup_anger) * 0.5
        scores["CONFORMITY"]   = z(row.backup_anger) * 0.8
        scores["SECURITY"]     = z(row.backup_risk)

    # 3. Normalization (Softmax or Standard Ratio)
    # Standard Ratio is safer for "Prompt Injection" readability
    total = sum(scores.values())
    
    if total == 0:
        return get_flat_prior()
        
    return {k: round(v/total, 3) for k, v in scores.items()}

def get_flat_prior():
    """Returns equal probability if data fails completely"""
    return {
        "POWER": 0.11, "ACHIEVEMENT": 0.11, "HEDONISM": 0.11,
        "STIMULATION": 0.11, "UNIVERSALISM": 0.11, "BENEVOLENCE": 0.11,
        "TRADITION": 0.11, "CONFORMITY": 0.11, "SECURITY": 0.11
    }



def batch_process_users_with_window(user_data_list, client):
    """
    1. Identifies all necessary dates (target +/- 3 days).
    2. Queries GDELT once for that superset of dates.
    3. reconstructs the 7-day average for each user locally.
    """
    
    # --- STEP 1: Calculate the "Superset" of dates needed ---
    all_needed_dates = set()
    
    for user in user_data_list:
        target = pd.to_datetime(user['date'])
        # Add target and its neighbors to the set
        for i in range(-3, 4): # -3 to +3
            d = target + pd.Timedelta(days=i)
            all_needed_dates.add(d.strftime('%Y-%m-%d'))
            
    print(f"User Count: {len(user_data_list)}")
    print(f"Unique Daily Partitions to Scan: {len(all_needed_dates)}")

    # Format for SQL
    date_sql_string = ", ".join([f"DATE('{d}')" for d in all_needed_dates])

    # --- STEP 2: The Batch Query (Fetches Daily Granularity) ---
    sql = f"""
    SELECT
        _PARTITIONDATE as day,
        -- EGO VALUES
        AVG(SAFE_CAST(REGEXP_EXTRACT(GCAM, r'c2.156:([0-9.]+)') AS FLOAT64)) as val_power,
        AVG(SAFE_CAST(REGEXP_EXTRACT(GCAM, r'c5.7:([0-9.]+)') AS FLOAT64))   as val_achievement,
        AVG(SAFE_CAST(REGEXP_EXTRACT(GCAM, r'c2.152:([0-9.]+)') AS FLOAT64)) as val_hedonism,
        AVG(SAFE_CAST(REGEXP_EXTRACT(GCAM, r'c2.17:([0-9.]+)') AS FLOAT64))  as val_stimulation,
        
        -- SOCIAL VALUES
        AVG(SAFE_CAST(REGEXP_EXTRACT(GCAM, r'c25.4:([0-9.]+)') AS FLOAT64))  as val_universalism, 
        AVG(SAFE_CAST(REGEXP_EXTRACT(GCAM, r'c25.6:([0-9.]+)') AS FLOAT64))  as val_benevolence,  
        AVG(SAFE_CAST(REGEXP_EXTRACT(GCAM, r'c25.10:([0-9.]+)') AS FLOAT64)) as val_tradition,    
        AVG(SAFE_CAST(REGEXP_EXTRACT(GCAM, r'c25.8:([0-9.]+)') AS FLOAT64))  as val_conformity,   
        AVG(SAFE_CAST(REGEXP_EXTRACT(GCAM, r'c25.2:([0-9.]+)') AS FLOAT64))  as val_security,

        -- BACKUP (LIWC)
        AVG(SAFE_CAST(REGEXP_EXTRACT(GCAM, r'c5.31:([0-9.]+)') AS FLOAT64))  as backup_social,
        AVG(SAFE_CAST(REGEXP_EXTRACT(GCAM, r'c5.46:([0-9.]+)') AS FLOAT64))  as backup_risk,
        AVG(SAFE_CAST(REGEXP_EXTRACT(GCAM, r'c5.10:([0-9.]+)') AS FLOAT64))  as backup_anger

    FROM `gdelt-bq.gdeltv2.gkg_partitioned`
    WHERE _PARTITIONDATE IN ({date_sql_string})
      AND V2Locations LIKE '%#US#%'
      AND GCAM IS NOT NULL
    GROUP BY day
    """

    job_config = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)

    # Start the query, passing in the extra configuration.
    query_job = client.query(
    (
        sql
    ),
    job_config=job_config,
    )  # Make an API request.

# A dry run query completes immediately.
    print("This query will process {} bytes.".format(query_job.total_bytes_processed))

    return 
    
    # Download everything to a Pandas DataFrame
    # Note: The 'day' column will be type 'db_date', convert to datetime64
    df_daily = client.query(sql).to_dataframe()
    df_daily['day'] = pd.to_datetime(df_daily['day'])
    
    # Index by date for O(1) lookups
    df_daily = df_daily.set_index('day')

    # --- STEP 3: The Reconstruction Loop (Python Side) ---
    final_output = []
    
    for user in user_data_list:
        target_date = pd.to_datetime(user['date'])
        start_win = target_date - pd.Timedelta(days=3)
        end_win = target_date + pd.Timedelta(days=3)
        
        # Slice the dataframe for this user's window
        # .loc[] includes the end index, so this grabs the 7 rows we need
        # We use reindex() to handle missing days (e.g. if GDELT missed a day) gracefully
        window_dates = pd.date_range(start=start_win, end=end_win)
        user_window = df_daily.reindex(window_dates) 
        
        # Calculate the Mean of the 7-day window (ignoring NaNs)
        avg_row = user_window.mean(skipna=True)
        
        # Check if we have valid data (if all days were NaN, avg_row is all NaN)
        if pd.isna(avg_row['val_power']):
            # Complete failure for this week -> Flat Prior
            user['psych_vector'] = {k: 0.11 for k in ["POWER", "SECURITY", "HEDONISM"]} # abbreviated
        else:
            # --- Apply Normalization/Fallback Logic on the AVERAGED Row ---
            def z(x): return 0.0 if pd.isna(x) else float(x)

            scores = {
                "POWER":       z(avg_row['val_power']),
                "ACHIEVEMENT": z(avg_row['val_achievement']),
                "HEDONISM":    z(avg_row['val_hedonism']),
                "STIMULATION": z(avg_row['val_stimulation'])
            }

            # Fallback Check (Did the Moral Foundations dictionary run this week?)
            if (z(avg_row['val_security']) + z(avg_row['val_universalism'])) > 0.001:
                scores.update({
                    "UNIVERSALISM": z(avg_row['val_universalism']), "BENEVOLENCE": z(avg_row['val_benevolence']),
                    "TRADITION": z(avg_row['val_tradition']), "CONFORMITY": z(avg_row['val_conformity']),
                    "SECURITY": z(avg_row['val_security'])
                })
            else:
                # LIWC Fallback
                scores.update({
                    "UNIVERSALISM": z(avg_row['backup_social'])*0.5, "BENEVOLENCE": z(avg_row['backup_social'])*0.5,
                    "TRADITION": z(avg_row['backup_anger'])*0.5, "CONFORMITY": z(avg_row['backup_anger'])*0.8,
                    "SECURITY": z(avg_row['backup_risk'])
                })

            # Final Normalize
            total = sum(scores.values())
            vector = {k: round(v/total, 3) for k,v in scores.items()} if total > 0 else None
            user['psych_vector'] = vector

        final_output.append(user)
        
    return final_output


CACHE_FILE = "gdelt_cache.json"

def load_cache():
    """Load cached date -> vector mappings."""
    try:
        with open(CACHE_FILE, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def save_cache(cache):
    """Save date -> vector mappings to cache file."""
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)
    print(f"Saved {len(cache)} vectors to {CACHE_FILE}")


def batch_process_with_cache(user_data_list, client, cache):
    """
    Like batch_process_users_with_window but uses cache.
    Only queries GDELT for dates not in cache.
    Returns updated cache and enriched users.
    """



# Construct a BigQuery client object.

    

    # Separate users into cached vs needs-query
    users_needing_query = []

    for user in user_data_list:
        date_str = user['date']
        if date_str in cache:
            # Use cached vector
            user['psych_vector'] = cache[date_str]
        else:
            users_needing_query.append(user)

    print(f"Users with cached vectors: {len(user_data_list) - len(users_needing_query)}")
    print(f"Users needing GDELT query: {len(users_needing_query)}")

    if users_needing_query:
        # Query GDELT for uncached users
        enriched = batch_process_users_with_window(users_needing_query, client)

        # Update cache with new vectors
        for user in enriched:
            date_str = user['date']
            if 'psych_vector' in user:
                cache[date_str] = user['psych_vector']

    return user_data_list, cache




def batch_process_users_lean(user_data_list, client, cache):
    """
    LEAN MODE: Queries ONLY the specific purchase date for each user.
    This fits within the 1TB Free Tier by scanning ~85% less data than the 7-day window.

    
    """

    users_needing_query = []

    for user in user_data_list:
        date_str = user['date']
        if date_str in cache:
            # Use cached vector
            user['psych_vector'] = cache[date_str]
        else:
            users_needing_query.append(user)

    print(f"Users with cached vectors: {len(user_data_list) - len(users_needing_query)}")
    print(f"Users needing GDELT query: {len(users_needing_query)}")
    


    # --- STEP 1: Get EXACT dates only (No Window) ---
    # We only care about the specific day the user made the purchase.
    unique_dates = list(set([u['date'] for u in users_needing_query]))
    
    print(f"User Count: {len(user_data_list)}")
    print(f"Lean Mode - Unique Daily Partitions to Scan: {len(unique_dates)}")

    # Format for SQL
    date_sql_string = ", ".join([f"DATE('{d}')" for d in unique_dates])

    # --- STEP 2: The Batch Query ---
    sql = f"""
    SELECT
        _PARTITIONDATE as day,
        -- EGO VALUES
        AVG(SAFE_CAST(REGEXP_EXTRACT(GCAM, r'c2.156:([0-9.]+)') AS FLOAT64)) as val_power,
        AVG(SAFE_CAST(REGEXP_EXTRACT(GCAM, r'c5.7:([0-9.]+)') AS FLOAT64))   as val_achievement,
        AVG(SAFE_CAST(REGEXP_EXTRACT(GCAM, r'c2.152:([0-9.]+)') AS FLOAT64)) as val_hedonism,
        AVG(SAFE_CAST(REGEXP_EXTRACT(GCAM, r'c2.17:([0-9.]+)') AS FLOAT64))  as val_stimulation,
        
        -- SOCIAL VALUES
        AVG(SAFE_CAST(REGEXP_EXTRACT(GCAM, r'c25.4:([0-9.]+)') AS FLOAT64))  as val_universalism, 
        AVG(SAFE_CAST(REGEXP_EXTRACT(GCAM, r'c25.6:([0-9.]+)') AS FLOAT64))  as val_benevolence,  
        AVG(SAFE_CAST(REGEXP_EXTRACT(GCAM, r'c25.10:([0-9.]+)') AS FLOAT64)) as val_tradition,    
        AVG(SAFE_CAST(REGEXP_EXTRACT(GCAM, r'c25.8:([0-9.]+)') AS FLOAT64))  as val_conformity,   
        AVG(SAFE_CAST(REGEXP_EXTRACT(GCAM, r'c25.2:([0-9.]+)') AS FLOAT64))  as val_security,

        -- BACKUP (LIWC)
        AVG(SAFE_CAST(REGEXP_EXTRACT(GCAM, r'c5.31:([0-9.]+)') AS FLOAT64))  as backup_social,
        AVG(SAFE_CAST(REGEXP_EXTRACT(GCAM, r'c5.46:([0-9.]+)') AS FLOAT64))  as backup_risk,
        AVG(SAFE_CAST(REGEXP_EXTRACT(GCAM, r'c5.10:([0-9.]+)') AS FLOAT64))  as backup_anger

    FROM `gdelt-bq.gdeltv2.gkg_partitioned`
    WHERE _PARTITIONDATE IN ({date_sql_string})
      AND V2Locations LIKE '%#US#%'
      AND GCAM IS NOT NULL
    GROUP BY day
    """

    # job_config = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)

    # # Start the query, passing in the extra configuration.
    # query_job = client.query(
    # (
    #     sql
    # ),
    # job_config=job_config,
    # )  # Make an API request.

# # A dry run query completes immediately.
#     print("This query will process {} bytes.".format(query_job.total_bytes_processed))

#     return 
    
    backup_file = "gdelt_raw_backup.csv"

    # Check which dates we already have in backup vs what we need
    dates_to_query = set(unique_dates)

    if os.path.exists(backup_file):
        print("Loading existing GDELT backup...")
        df_backup = pd.read_csv(backup_file)
        df_backup['day'] = pd.to_datetime(df_backup['day'])

        # Check which dates are already in backup
        existing_dates = set(df_backup['day'].dt.strftime('%Y-%m-%d').tolist())
        dates_to_query = dates_to_query - existing_dates

        print(f"Backup has {len(existing_dates)} dates, need {len(dates_to_query)} more")

        if dates_to_query:
            # Query only the missing dates
            date_sql_string_new = ", ".join([f"DATE('{d}')" for d in dates_to_query])
            sql_new = f"""
            SELECT
                _PARTITIONDATE as day,
                AVG(SAFE_CAST(REGEXP_EXTRACT(GCAM, r'c2.156:([0-9.]+)') AS FLOAT64)) as val_power,
                AVG(SAFE_CAST(REGEXP_EXTRACT(GCAM, r'c5.7:([0-9.]+)') AS FLOAT64))   as val_achievement,
                AVG(SAFE_CAST(REGEXP_EXTRACT(GCAM, r'c2.152:([0-9.]+)') AS FLOAT64)) as val_hedonism,
                AVG(SAFE_CAST(REGEXP_EXTRACT(GCAM, r'c2.17:([0-9.]+)') AS FLOAT64))  as val_stimulation,
                AVG(SAFE_CAST(REGEXP_EXTRACT(GCAM, r'c25.4:([0-9.]+)') AS FLOAT64))  as val_universalism,
                AVG(SAFE_CAST(REGEXP_EXTRACT(GCAM, r'c25.6:([0-9.]+)') AS FLOAT64))  as val_benevolence,
                AVG(SAFE_CAST(REGEXP_EXTRACT(GCAM, r'c25.10:([0-9.]+)') AS FLOAT64)) as val_tradition,
                AVG(SAFE_CAST(REGEXP_EXTRACT(GCAM, r'c25.8:([0-9.]+)') AS FLOAT64))  as val_conformity,
                AVG(SAFE_CAST(REGEXP_EXTRACT(GCAM, r'c25.2:([0-9.]+)') AS FLOAT64))  as val_security,
                AVG(SAFE_CAST(REGEXP_EXTRACT(GCAM, r'c5.31:([0-9.]+)') AS FLOAT64))  as backup_social,
                AVG(SAFE_CAST(REGEXP_EXTRACT(GCAM, r'c5.46:([0-9.]+)') AS FLOAT64))  as backup_risk,
                AVG(SAFE_CAST(REGEXP_EXTRACT(GCAM, r'c5.10:([0-9.]+)') AS FLOAT64))  as backup_anger
            FROM `gdelt-bq.gdeltv2.gkg_partitioned`
            WHERE _PARTITIONDATE IN ({date_sql_string_new})
              AND V2Locations LIKE '%#US#%'
              AND GCAM IS NOT NULL
            GROUP BY day
            """

            try:
                print(f"Querying GDELT for {len(dates_to_query)} new dates...")
                df_new = client.query(sql_new).to_dataframe()

                # Merge with existing backup
                df_results = pd.concat([df_backup, df_new], ignore_index=True)

                # Save updated backup
                df_results.to_csv(backup_file, index=False)
                print(f"Updated backup with {len(df_new)} new dates!")
            except Exception as e:
                print(f"CRITICAL QUERY FAILURE: {e}")
                return user_data_list, cache
        else:
            # All dates already in backup
            df_results = df_backup
            print("All needed dates found in backup!")
    else:
        print(f"No backup found, querying GDELT for {len(unique_dates)} unique dates...")
        try:
            print("Executing Query...")
            df_results = client.query(sql).to_dataframe()
            df_results.to_csv(backup_file, index=False)
            print(f"Query complete! Saved {len(df_results)} daily aggregates to {backup_file}")
        except Exception as e:
            print(f"CRITICAL QUERY FAILURE: {e}")
            return user_data_list, cache

    # --- STEP 3: Map Results Back to Users ---
    # Create a lookup map: Date String -> Vector
    date_vector_map = {}
    
    # Helper for safe float conversion
    def z(x): return 0.0 if pd.isna(x) else float(x)

    for _, row in df_results.iterrows():
        # 1. Build the Raw Score Dictionary
        scores = {
            "POWER":       z(row.val_power),
            "ACHIEVEMENT": z(row.val_achievement),
            "HEDONISM":    z(row.val_hedonism),
            "STIMULATION": z(row.val_stimulation)
        }
        
        # 2. Apply Fallback Logic (Same as before)
        if (z(row.val_security) + z(row.val_universalism)) > 0.001:
            scores.update({
                "UNIVERSALISM": z(row.val_universalism), "BENEVOLENCE": z(row.val_benevolence),
                "TRADITION": z(row.val_tradition), "CONFORMITY": z(row.val_conformity),
                "SECURITY": z(row.val_security)
            })
        else:
            # LIWC Fallback
            scores.update({
                "UNIVERSALISM": z(row.backup_social)*0.5, "BENEVOLENCE": z(row.backup_social)*0.5,
                "TRADITION": z(row.backup_anger)*0.5, "CONFORMITY": z(row.backup_anger)*0.8,
                "SECURITY": z(row.backup_risk)
            })

        # 3. Normalize
        total = sum(scores.values())
        final_vector = {k: round(v/total, 3) for k,v in scores.items()} if total > 0 else None
        
        # 4. Save to Map
        d_str = str(row['day'])
        if isinstance(row['day'], (pd.Timestamp, datetime)):
            d_str = row['day'].strftime('%Y-%m-%d')
        date_vector_map[d_str] = final_vector

    # --- STEP 4: Update cache with new vectors ---
    for date_str, vector in date_vector_map.items():
        if vector is not None:
            cache[date_str] = vector

    # --- STEP 5: Attach to User Objects ---
    final_output = []
    for user in user_data_list:
        u_date = user['date']
        vector = date_vector_map.get(u_date, cache.get(u_date, None))

        # Handle Missing Data (Flat Prior)
        if vector is None:
            vector = {k: 0.11 for k in ["POWER", "SECURITY", "HEDONISM", "ACHIEVEMENT", "STIMULATION", "UNIVERSALISM", "BENEVOLENCE", "TRADITION", "CONFORMITY"]}

        user['psych_vector'] = vector
        final_output.append(user)

    return final_output, cache


def enrich_files_with_gdelt(files, client):
    """
    Read all users, batch query GDELT (with caching), write back with psych_vector.
    Also saves vectors to gdelt_cache.json for future use.
    """

    # Load existing cache
    cache = load_cache()
    print(f"Loaded {len(cache)} cached vectors from {CACHE_FILE}")

    # Step 1: Read all users, track which file they came from
    all_users = []
    file_boundaries = []  # [(filename, start_idx, end_idx), ...]

    for filename in files:
        start_idx = len(all_users)
        with open(filename, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                user = json.loads(line)
                # Rename selection_month -> date for the function
                user['date'] = user['selection_month']
                all_users.append(user)
        end_idx = len(all_users)
        file_boundaries.append((filename, start_idx, end_idx))
        print(f"Loaded {end_idx - start_idx} users from {filename}")

    print(f"\nTotal users to process: {len(all_users)}")

    # Step 2: Batch process with caching (only queries GDELT for uncached dates)
    enriched_users, cache = batch_process_users_lean(all_users, client, cache)

    # Step 3: Save updated cache
    save_cache(cache)

    # Step 4: Write back to files
    for filename, start_idx, end_idx in file_boundaries:
        users_for_file = enriched_users[start_idx:end_idx]

        # Clean up the temporary 'date' key we added
        for user in users_for_file:
            if 'date' in user:
                del user['date']

        out_filename = filename.replace(".jsonl", "_enriched.jsonl")
        with open(out_filename, "w") as f:
            for user in users_for_file:
                f.write(json.dumps(user) + "\n")
        print(f"Wrote {len(users_for_file)} users to {out_filename}")

# Run it

if __name__ == "__main__":
    files = ["validationgdelt.jsonl", "testgdelt.jsonl", "traininggdelt.jsonl"]
    enrich_files_with_gdelt(files, client)