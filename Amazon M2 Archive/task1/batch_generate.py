"""
Batch Persona Generation Script
================================
Generate personas for sessions in batches with checkpointing and resume capability.

Usage:
    python generate_personas_batch.py --start 0 --end 100
    python generate_personas_batch.py --start 100 --end 200
    # Or process all:
    python generate_personas_batch.py --start 0 --end -1
"""

import pandas as pd
import argparse
import os
import json
import time
from datetime import datetime
from smolagents.models import OpenAIServerModel
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

#go to Amazon M2 Archive/task1/ since persona_dataset.csv is there
os.chdir("Amazon M2 Archive/task1")

# Configuration
INPUT_CSV = "persona_dataset.csv"
OUTPUT_CSV = "persona_dataset_with_personas.csv"
CHECKPOINT_DIR = "persona_checkpoints"
LOG_FILE = "persona_generation.log"
CHECKPOINT_FREQUENCY = 100  # Save every N rows
RATE_LIMIT_DELAY = 0.1  # Seconds between API calls (adjust as needed)

# Initialize model
persona_model = OpenAIServerModel(
    model_id="gemma3",
    api_base="https://ellm.nrp-nautilus.io/v1",
    api_key=os.getenv("NAUT_API_KEY"),
)

log_lock = Lock()


gepa_prompt = """
You are an expert Consumer Psychologist. Your goal is to infer a user\'s detailed **Shopping Persona** based ONLY on their confirmed Purchase History.\n\nSince you only have purchase data (no views/clicks), you must use **Deductive Reasoning** to fill in the gaps. You must look for what is *missing* just as much as what is *present*.\n\n=== EXAMPLE ANALYSIS (Use this as a guide for Logic and Output Style) ===\n\n**Input Purchase History:**\n- Hair Dryer Blow Dryer, 180000 RPM High-Speed Brushless Motor (None)\n- License Plate Screws with Rustproof Finish - Stainless Steel (4-Pack, Black) (None)\n- AIRROBO Robot Vacuum and Mop, 3000Pa Powerful Suction (None)\n- NADALY D200 Robot Vacuum and Mop Combo, Lidar Navigation (None)\n\n**Psychological Analysis (Internal Reasoning):**\n1. **Brand Detective:** The user bought "AIRROBO" and "NADALY" vacuums. These are not famous "default" brands like Roomba or Dyson. They are high-spec, value-priced, online-native brands.\n   * *Inference:* This implies the user is **spec-conscious** and relies heavily on **reading detailed reviews** to find hidden gems, rather than trusting marketing or brand recognition. They are cautious about overpaying for big names.\n2. **Inference by Omission:** The list is 100% "Hard Goods" (Hardware, Electronics, Tools). There are no clothes, food, or consumables.\n   * *Inference:* This strongly suggests they categorize Amazon as a "Toolbox/Hardware Store" and likely handle groceries and clothing through offline channels or other specific retailers.\n3. **Micro-Optimization:** Buying specific "Rustproof Black License Plate Screws" and a "180000 RPM" dryer indicates a high attention to detail. They prioritize **functionality, durability, and specific fit** over generic solutions.\n4. **Strategic Synthesis:** The user is a researcher. They compare highly positive and negative reviews to ensure the "unknown" brands (Nadaly) are safe.\n\n**Final Persona Description (Clean Output):**\n<persona_description>\n[Participant] prefers shopping for certain categories like home essentials and specialized tools online but tends to buy groceries and clothes offline. They read reviews, especially for unfamiliar products, focusing on detailed reviews and images to assess product quality and fit, such as ease of assembly or actual appearance. [Participant] compares both highly positive and negative reviews to get a balanced perspective. They are cautious about sponsored products, often avoiding them due to concerns over biased promotion, and prefer to check non-sponsored listings to ensure a more genuine assessment.\n</persona_description>\n\n=== END EXAMPLE ===\n\n**YOUR TASK:**\nAnalyze the following PURCHASE HISTORY for the current user.\n\n**Purchase History:**\n{product_list_str}\n\n**Instructions:**\n1. **Analyze Brand Tier:** Are these "Famous Brands," "Value Brands," or "High-Spec Unknowns"? \n   - *Insight:* Buying obscure high-spec brands implies the user **reads reviews** and cares about specs.\n   \n2. **Analyze "Inference by Omission":** - If they buy durable goods but NO food/clothes, you **MUST** infer: *"Likely handles groceries and clothing through offline channels."*\n\n3. **Construct the Persona:**\n   - Write 3-6 sentences describing the user\'s strategy and psychology.\n   - **CRITICAL RULE:** Do NOT cite specific items in the final description (e.g., do not say "evidenced by the vacuum"). Just state the trait (e.g., "They prioritize home automation").\n   - **CRITICAL RULE:** Do NOT mention "insufficient data." Use the deductions above to form a complete picture.\n\n**Output Format:**\n**1. Brand & Tier Analysis:** [Your deductive reasoning]\n**2. Strategic Omissions:** [What are they NOT buying?]\n**3. Behavioral Conclusion:** [Synthesize the traits]\n\n<persona_description>\n[Your clean final paragraph]\n</persona_description>
"""

def setup_logging():
    """Initialize logging and checkpoint directory."""
    Path(CHECKPOINT_DIR).mkdir(exist_ok=True)
    
def log_message(message: str):
    """Log a message to both console and file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    with log_lock:
        print(log_entry)
        with open(LOG_FILE, "a") as f:
            f.write(log_entry + "\n")

def generate_persona(product_details: str, session_id: int) -> dict:
    """
    Generate a persona for a single session.
    
    Returns:
        dict with keys: 'persona', 'success', 'error'
    """
    try:
        if not product_details or pd.isna(product_details):
            return {
                'persona': '',
                'success': False,
                'error': 'Empty product details'
            }
        
        # Format prompt
        prompt = gepa_prompt.format(product_list_str=product_details)
        
        # Call LLM
        response = persona_model([{"role": "user", "content": prompt}])
        raw_output = response.content if hasattr(response, 'content') else str(response)
        
        # Extract persona from tags
        if "<persona_description>" in raw_output and "</persona_description>" in raw_output:
            persona = raw_output.split("<persona_description>")[1].split("</persona_description>")[0].strip()
        else:
            # Fallback: use full output
            persona = raw_output.strip()
            log_message(f"Warning: Session {session_id} - No tags found, using full output")
        
        return {
            'persona': persona,
            'success': True,
            'error': None
        }
    
    except Exception as e:
        error_msg = f"Error generating persona for session {session_id}: {str(e)}"
        log_message(error_msg)
        return {
            'persona': '',
            'success': False,
            'error': str(e)
        }

def load_checkpoint(start_idx: int, end_idx: int) -> pd.DataFrame | None:
    """Load the most recent checkpoint for this range."""
    checkpoint_pattern = f"checkpoint_{start_idx}_{end_idx}_*.csv"
    checkpoints = list(Path(CHECKPOINT_DIR).glob(checkpoint_pattern.replace('*', '[0-9]*')))
    
    if checkpoints:
        latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
        log_message(f"Loading checkpoint: {latest}")
        return pd.read_csv(latest)
    return None

def process_row(idx, row, force_rerun):
    """Process a single row (thread-safe wrapper)."""
    # Skip if already has persona
    persona_str = str(row['persona']).strip().lower()
    if not force_rerun and row['persona'] and not pd.isna(row['persona']) and persona_str and persona_str != 'nan':
        return idx, '', 'skipped'
    
    # Generate persona
    result = generate_persona(row['prev_items_details'], row['session_id'])
    
    if result['success']:
        return idx, result['persona'], 'success'
    else:
        return idx, result['persona'], 'error'


def save_checkpoint(df: pd.DataFrame, start_idx: int, end_idx: int, current_idx: int):
    """Save a checkpoint of current progress."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_file = f"{CHECKPOINT_DIR}/checkpoint_{start_idx}_{end_idx}_{current_idx}_{timestamp}.csv"
    df.to_csv(checkpoint_file, index=False)
    log_message(f"Checkpoint saved: {checkpoint_file}")

def process_batch(start_idx: int, end_idx: int, force_rerun: bool = False, num_threads: int = 1):
    """
    Process a batch of sessions from start_idx to end_idx.
    
    Args:
        start_idx: Starting row index (inclusive)
        end_idx: Ending row index (exclusive), -1 for end of file
        force_rerun: If True, regenerate even if persona exists
    """
    setup_logging()
    
    log_message("="*80)
    log_message(f"Starting batch processing: rows {start_idx} to {end_idx if end_idx != -1 else 'END'}")
    if num_threads > 1:
        log_message(f"Using {num_threads} threads for processing")
    log_message("="*80)
    
    # Load dataset
    log_message(f"Loading dataset from {INPUT_CSV}...")
    
    # Load in chunks to handle large file
    chunk_size = 10000
    chunks = []
    rows_to_skip = start_idx
    rows_to_read = None if end_idx == -1 else (end_idx - start_idx)
    
    for chunk in pd.read_csv(INPUT_CSV, chunksize=chunk_size, skiprows=range(1, rows_to_skip + 1) if rows_to_skip > 0 else None):
        chunks.append(chunk)
        if rows_to_read and sum(len(c) for c in chunks) >= rows_to_read:
            break
    
    df = pd.concat(chunks, ignore_index=True)
    
    if rows_to_read:
        df = df.head(rows_to_read)
    
    log_message(f"Loaded {len(df)} rows")
    
    # Try to load checkpoint
    if not force_rerun:
        checkpoint_df = load_checkpoint(start_idx, end_idx)
        if checkpoint_df is not None:
            # Merge with checkpoint
            df = checkpoint_df
            
            log_message("Resumed from checkpoint")
    
    # Ensure persona column exists
    if 'persona' not in df.columns:
        df['persona'] = ''
    else:
        df['persona'] = df['persona'].astype(str).fillna('') 
    
    # Process each row
    processed = 0
    skipped = 0
    errors = 0
    start_time = time.time()

    if num_threads > 1:

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = {}
            for idx, row in df.iterrows():
                future = executor.submit(process_row, idx, row, force_rerun)
                futures[future] = idx

            completed = 0
            for future in as_completed(futures):
                idx, persona, status = future.result()

                if status == 'success':
                    df.loc[idx, 'persona'] = persona
                    processed += 1
                elif status == 'skipped':
                    skipped += 1
                else:
                    df.loc[idx, 'persona'] = persona
                    errors += 1

                completed += 1
                if completed % 10 == 0:
                    elapsed = time.time() - start_time
                    rate = completed / elapsed
                    remaining = len(df) - completed
                    eta_seconds = remaining / rate if rate > 0 else 0

                    log_message(f"Progress: {completed}/{len(df)} "
                                f"({completed/len(df)*100:.1f}%) | "
                                f"Processed: {processed} | Skipped: {skipped} | Errors: {errors} | "
                                f"Rate: {rate:.2f} rows/sec | ETA: {eta_seconds/60:.1f} min")

                if completed % CHECKPOINT_FREQUENCY == 0 and completed > 0:
                    save_checkpoint(df, start_idx, end_idx, start_idx + completed)

    else:
        # SINGLE-THREADED PROCESSING
        for idx, row in df.iterrows():
            # Skip if already has persona
            if not force_rerun and row['persona'] and not pd.isna(row['persona']) and str(row['persona']).strip():
                skipped += 1
                continue
            
            # Generate persona (note: row['session_id'], not idx)
            result = generate_persona(row['prev_items_details'], row['session_id'])
            
            if result['success']:
                df.loc[idx, 'persona'] = result['persona']
                processed += 1
            else:
                df.loc[idx, 'persona'] = result['persona']
                errors += 1

            time.sleep(RATE_LIMIT_DELAY)

            total_processed = processed + skipped
            if total_processed > 0 and total_processed % 10 == 0:
                elapsed = time.time() - start_time
                rate = total_processed / elapsed
                remaining = len(df) - total_processed
                eta_seconds = remaining / rate if rate > 0 else 0
                
                log_message(f"Progress: {total_processed}/{len(df)} "
                        f"({total_processed/len(df)*100:.1f}%) | "
                        f"Processed: {processed} | Skipped: {skipped} | Errors: {errors} | "
                        f"Rate: {rate:.2f} rows/sec | ETA: {eta_seconds/60:.1f} min")
            
            # Checkpoint
            if processed % CHECKPOINT_FREQUENCY == 0 and processed > 0:
                save_checkpoint(df, start_idx, end_idx, start_idx + total_processed)
    log_message("="*80)
    log_message(f"Batch complete! Processed: {processed}, Skipped: {skipped}, Errors: {errors}")
    log_message("="*80)
    
    # Save final output
    if end_idx == -1:
        output_file = OUTPUT_CSV
    else:
        output_file = f"persona_dataset_batch_{start_idx}_{end_idx}.csv"
    
    df.to_csv(output_file, index=False)
    log_message(f"Saved to {output_file}")
    
    return df
        





def merge_batches(batch_files: list[str], output_file: str = OUTPUT_CSV):
    """Merge multiple batch output files into one."""
    log_message(f"Merging {len(batch_files)} batch files...")
    
    dfs = []
    for file in batch_files:
        dfs.append(pd.read_csv(file))
    
    merged = pd.concat(dfs, ignore_index=True)
    merged.to_csv(output_file, index=False)
    log_message(f"Merged file saved to {output_file}")
    
    return merged

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate personas in batches")
    parser.add_argument("--start", type=int, required=True, help="Start index (inclusive)")
    parser.add_argument("--end", type=int, required=True, help="End index (exclusive), use -1 for end of file")
    parser.add_argument("--force", action="store_true", help="Force regenerate even if persona exists")
    parser.add_argument("--rate-limit", type=float, default=0.1, help="Delay between API calls in seconds")
    parser.add_argument("--threads", type=int, default=1, help="Number of concurrent threads (default: 1)")


    
    args = parser.parse_args()
    
    RATE_LIMIT_DELAY = args.rate_limit
    num_threads = args.threads
    
    process_batch(args.start, args.end, args.force, num_threads)