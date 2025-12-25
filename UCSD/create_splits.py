import random
from datasets import load_dataset
import json
from dataclasses import dataclass
from typing import Optional, List, Any
from collections import defaultdict
import re

SIGNAL_PATTERNS = [
    # justification / contrast
    r"\b(because|since|so|therefore|however|but|although|though|despite|yet)\b",
    # decision / outcome
    r"\b(returned|return|refund|kept|repurchase|buy again|won't buy|wouldn't buy|recommend|recommended)\b",
    # value / price
    r"\b(price|expensive|cheap|worth|value|overpriced)\b",
    # common product aspects (expand per category later)
    r"\b(quality|smell|scent|texture|fit|size|battery|durable|comfortable|easy|hard)\b",
    # strong sentiment
    r"\b(love|hate|amazing|terrible|awful|great|perfect|disappointed)\b",
    # numbers often indicate specifics
    r"\b\d+(\.\d+)?\b",
]

def split_sentences(text: str) -> List[str]:
    # Simple sentence split on ., !, ? while keeping content
    # (Not perfect, but much better than split("."))
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p.strip() for p in parts if p.strip()]

def score_sentence(s: str) -> int:
    s_low = s.lower()
    score = 0
    for pat in SIGNAL_PATTERNS:
        if re.search(pat, s_low):
            score += 2
    # Slight bonus for longer (but cap)
    score += min(len(s), 200) // 80  # 0..2
    return score

def generate_high_signal_excerpt(review_text: str, max_chars: int = 900, min_sentences: int = 2) -> str:
    if not review_text or not review_text.strip():
        return ""

    sents = split_sentences(review_text)
    if len(sents) <= min_sentences:
        return review_text[:max_chars]

    scored = [(score_sentence(s), i, s) for i, s in enumerate(sents)]
    scored.sort(reverse=True)

    chosen = []
    used_chars = 0

    # pick top sentences, then output in original order for coherence
    for _, _, s in scored:
        if used_chars + len(s) + 1 > max_chars:
            continue
        chosen.append(s)
        used_chars += len(s) + 1
        if used_chars >= max_chars * 0.7 and len(chosen) >= min_sentences:
            break

    if not chosen:
        return review_text[:max_chars]

    # preserve original order
    chosen_set = set(chosen)
    ordered = [s for s in sents if s in chosen_set]
    excerpt = " ".join(ordered).strip()
    return excerpt[:max_chars]

def reservoir_sample(stream, k, seed=0, max_items=None):
    
    rng = random.Random(seed)
    sample = []
    for i, ex in enumerate(stream, start=1):
        if max_items and i > max_items:
            break
        if i <= k:
            sample.append(ex)
        else:
            j = rng.randint(1, i)
            if j <= k:
                sample[j-1] = ex
    return sample

review_path = "hf://datasets/McAuley-Lab/Amazon-Reviews-2023/raw/review_categories/All_Beauty.jsonl"
meta_path = "hf://datasets/McAuley-Lab/Amazon-Reviews-2023/raw/meta_categories/meta_All_Beauty.jsonl"
ds = load_dataset("json", data_files=review_path, split="train", streaming=True)
meta_ds = load_dataset("json", data_files=meta_path, split="train", streaming=True)


sample = reservoir_sample(ds, k=5000, seed=42, max_items=10)  # cap scan if you want

user_review_counts = defaultdict(int)
for review in ds:  # Stream through reviews
    user_id = review['user_id']
    user_review_counts[user_id] += 1



qualified_users = {uid: count for uid, count in user_review_counts.items() if count > 6}
print(f"Found {len(qualified_users)} users with >6 reviews")

# goal_for_category = int(len(qualified_users) * 0.1) # 10% of users
goal_for_category = 30
print(f"Goal for category: {goal_for_category}")


qualified_reviews = []
for review in ds:
    if review['user_id'] in qualified_users:
        qualified_reviews.append(review)
        if len(qualified_reviews) >= goal_for_category:
            break
        # Could break early if you have enough




# print("Review:================================================")
# print(sample[0]['text'])
# print("========================================================")
# print(generate_high_signal_excerpt(sample[0]['text']))



# def main():
#     print(generate_high_signal_excerpt("This is perfect for my between salon visits. I have been using this now twice a week for over a month and I absolutely love it! My skin looks amazing and feels super smooth and silky. This is also super easy to use (just follow instructions). I can see already that I will begin expanding the time between visits which will definitely help me save money in the long run. Highly recommend!"))

# if __name__ == "__main__":
#     main()



# #save the sample to a jsonl file    
# with open("sample.jsonl", "w") as f:
#     for review in sample:
#         f.write(json.dumps(review) + "\n")