import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.preprocessing import StandardScaler

# 1. LOAD DATA FROM ENRICHED JSONL
features = [
    "POWER",
    "ACHIEVEMENT",
    "HEDONISM",
    "STIMULATION",
    "SELF_DIRECTION",
    "UNIVERSALISM",
    "BENEVOLENCE",
    "TRADITION",
    "CONFORMITY",
    "SECURITY",
]

records = []
full_by_user_id = {}
with open("thousand_reddit_enriched.jsonl", "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        target_vector = dict(obj.get("target_vector") or {})

        # Normalize key naming inconsistencies
        if "SELF-DIRECTION" in target_vector and "SELF_DIRECTION" not in target_vector:
            target_vector["SELF_DIRECTION"] = target_vector.pop("SELF-DIRECTION")

        # Only keep users with all required Schwartz values
        if not all(k in target_vector for k in features):
            continue

        uid = obj.get("user_id")
        record = {"user_id": uid}
        record.update({k: target_vector[k] for k in features})
        record["subreddits"] = obj.get("subreddits", [])
        records.append(record)
        # Full row for pipeline JSONL (needs history + same schema as rawreddit.jsonl)
        full_by_user_id[uid] = {
            "user_id": uid,
            "target_vector": target_vector,
            "subreddits": obj.get("subreddits", []),
            "history": obj.get("history", []),
        }

df = pd.DataFrame(records)

# 2. PREPROCESSING & CLUSTERING
X = df[features].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Set K=10 for your 250-person sample. Use K=20 when you hit 1000 users.
k = 200 
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_scaled)
df['cluster'] = kmeans.labels_.astype(str)

# FIND ARCHETYPES (Users closest to the cluster centroids)
closest_indices, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, X_scaled)
archetypes = df.iloc[closest_indices].copy()

# 3. PCA FOR VISUALIZATION (Reducing 10D to 2D)
pca = PCA(n_components=2)
components = pca.fit_transform(X_scaled)
df['pca1'] = components[:, 0]
df['pca2'] = components[:, 1]
archetypes['pca1'] = df.iloc[closest_indices]['pca1'].values
archetypes['pca2'] = df.iloc[closest_indices]['pca2'].values

# 4. PLOT 1: CLUSTER SCATTER PLOT
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='pca1', y='pca2', hue='cluster', palette='tab10', alpha=0.7)
plt.scatter(archetypes['pca1'], archetypes['pca2'], color='black', marker='X', s=150, label='Archetypes')
plt.title(f'Persona Clusters (n={len(df)}, k={k})\nBlack X marks the representative archetype for each cluster')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('persona_clusters_pca.png')

# 5. PLOT 2: VALUE FINGERPRINTS (Compare top 4 archetypes)
plt.figure(figsize=(12, 6))
subset_archetypes = archetypes.iloc[:4] # Compare first 4 for visual clarity
plot_data = subset_archetypes.melt(id_vars='user_id', value_vars=features, var_name='Value', value_name='Score')
sns.barplot(data=plot_data, x='Value', y='Score', hue='user_id')
plt.title('Schwartz Value Fingerprints of Representative Archetypes')
plt.ylim(0, 1.1)
plt.xticks(rotation=45)
plt.legend(title='Archetype User', loc='upper right')
plt.tight_layout()
plt.savefig('archetype_profiles.png')

# 6. SAVE RESULTS
archetypes.to_csv("archetypes_to_simulate.csv", index=False)

archetype_jsonl_path = os.path.join("ClusteredPersonas", "archetypes_to_simulate.jsonl")
os.makedirs(os.path.dirname(archetype_jsonl_path), exist_ok=True)
missing = []
with open(archetype_jsonl_path, "w", encoding="utf-8") as jf:
    for uid in archetypes["user_id"]:
        row = full_by_user_id.get(uid)
        if row is None:
            missing.append(uid)
            continue
        jf.write(json.dumps(row, ensure_ascii=False) + "\n")
if missing:
    print(f"Warning: {len(missing)} archetype user_ids missing from enriched source: {missing[:5]}...")

print(f"Success! Identified {len(archetypes)} archetypes.")
print(
    "Saved: persona_clusters_pca.png, archetype_profiles.png, archetypes_to_simulate.csv, "
    f"{archetype_jsonl_path}"
)