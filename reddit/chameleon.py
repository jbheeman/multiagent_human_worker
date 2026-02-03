import os
import requests
import zipfile
from datasets import load_dataset

# 1. Define URL (The direct link to the zip file in the HF Repo)
# Note: We use the "resolve/main" link to get the raw file
url = "https://huggingface.co/datasets/webis/tldr-17/resolve/main/data/corpus-webis-tldr-17.zip"
zip_path = "corpus-webis-tldr-17.zip"
json_path = "corpus-webis-tldr-17.json"

def download_file(url, filename):
    print(f"Downloading {filename}...")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    print("Download complete.")

# 2. Download if missing
if not os.path.exists(json_path):
    if not os.path.exists(zip_path):
        download_file(url, zip_path)
    
    # 3. Unzip
    print("Unzipping...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(".")
    print("Unzip complete.")

# 4. Load with 'datasets' (The Standard Way)
print("Loading dataset into memory...")
ds = load_dataset("json", data_files=json_path, split="train")

# 5. Verify
print(f"Success! Loaded {len(ds)} posts.")
print("Sample Post:", ds[0])