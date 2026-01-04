import json
import math
from pathlib import Path
from typing import Dict, Any, List, Tuple
from collections import Counter

import numpy as np
from sentence_transformers import SentenceTransformer
import mord
from sklearn.metrics import accuracy_score, mean_absolute_error


# -----------------------------
# 1) I/O helpers
# -----------------------------
def read_jsonl(path: str, persona_json_path: str) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    users_seen = set()
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                user = json.loads(line)
                user_id = user["user_id"]
                if user_id in users_seen: #added to avoid duplicates - but this is wrong because we have userA with a revew in category1 and userA with a review in category2 - we want both
                    continue
                users_seen.add(user_id)
                rows.append(user)
    with open (persona_json_path, "r", encoding="utf-8") as f:
        persona_rows = {}
        for line in f:
            line = line.strip()
            if line:
                persona = json.loads(line)
                user_id = persona["user_id"]
                persona_rows[user_id] = persona["persona"]
    
    return rows , persona_rows


def format_product_text(prod: Dict[str, Any]) -> str:
    # prod is user["heldout"]["product"]
    title = (prod.get("title") or "").strip()
    cat = (prod.get("main_category") or "").strip()
    brand = (prod.get("brand") or "").strip() if prod.get("brand") else ""
    price = prod.get("price", None)

    price_str = "unknown" if price is None else str(price)
    # Keep it compact and consistent
    return f"Title: {title} | Category: {cat} | Brand: {brand or 'unknown'} | Price: {price_str}"


def get_persona_text(user_row: Dict[str, Any], persona_rows: Dict[str, Dict[str, Any]]) -> str:
    """
    Adjust this depending on where you stored personas.

    Option A: you appended persona into the row:
      user_row["persona"]["persona_description"]
    Option B: you appended directly:
      user_row["persona_description"]

    Change this function to match your data.
    """
    if "persona_description" in user_row:
        return user_row["persona_description"]
    if "persona" in user_row and isinstance(user_row["persona"], dict):
        return user_row["persona"].get("persona_description", "")
    raise KeyError("No persona_description found in row")


def rating_to_class(rating: float) -> int:
    # Map 1.0..5.0 -> 0..4
    r = int(round(float(rating)))
    r = min(5, max(1, r))
    return r - 1


# -----------------------------
# 2) Build dataset
# -----------------------------
def build_examples(rows: List[Dict[str, Any]], persona_rows: Dict[str, Dict[str, Any]]) -> Tuple[List[str], List[str], np.ndarray]:
    persona_texts = []
    item_texts = []
    y = []

    for row in rows:
        user_id = row["user_id"]
        persona_for_id = persona_rows.get(user_id, {})
        # persona = (get_persona_text(row, persona_rows) or "").strip()
        heldout = row["heldout"]
        item_text = format_product_text(heldout["product"])

        # print(f"User ID: {user_id}")
        # print(f"Persona: {persona_for_id}")
        # print(f"Item: {item_text}")
        

        if not persona_for_id:
            # Skip or handle with a placeholder
            continue

        persona_texts.append(persona_for_id)
        item_texts.append(item_text)
        y.append(rating_to_class(heldout["rating"]))

    return persona_texts, item_texts, np.array(y, dtype=np.int64)


# -----------------------------
# 3) Embedding + feature construction
# -----------------------------
def batched_encode(model: SentenceTransformer, texts: List[str], batch_size: int = 64) -> np.ndarray:
    # Returns (N, D) float32
    embs = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # makes dot-products behave nicely
    )
    return embs.astype(np.float32)


def make_features(persona_emb: np.ndarray, item_emb: np.ndarray) -> np.ndarray:
    # Simple and effective:
    # [p, i, p*i]
    #trying p*i only
    return np.concatenate([ persona_emb * item_emb], axis=1)


# -----------------------------
# 4) Train/eval
# -----------------------------
def eval_model(model, X: np.ndarray, y: np.ndarray, name: str):
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    # convert to 1..5 stars for human-readable MAE
    mae_stars = mae  # in class units; same scale as stars
    print(f"[{name}] acc={acc:.4f}  MAE(stars)={mae_stars:.4f}")
    return acc, mae


def main():
    train_path = "training.jsonl"
    training_personas_path = "util_score_model/training_personas.jsonl"

    val_path   = "../UCSD/validation.jsonl"
    val_personas_path   = "util_score_model/validation_personas.jsonl"

    test_path  = "../UCSD/test.jsonl"
    test_personas_path  = "util_score_model/test_personas.jsonl"

    # 1) Load split rows (each row must include persona_description)
    train_rows, train_persona_rows = read_jsonl(train_path, training_personas_path)
    print(f"Loaded {len(train_rows)} train rows, {len(train_persona_rows)} persona rows")
    


    val_rows, val_persona_rows   = read_jsonl(val_path, val_personas_path)
  

    print(f"Loaded {len(val_rows)} val rows, {len(val_persona_rows)} persona rows")

    test_rows, test_persona_rows  = read_jsonl(test_path, test_personas_path)

    print(f"Loaded {len(test_rows)} test rows, {len(test_persona_rows)} persona rows")

    # 2) Build text pairs + labels
    tr_p, tr_i, y_tr = build_examples(train_rows, train_persona_rows)


    va_p, va_i, y_va = build_examples(val_rows, val_persona_rows)
    te_p, te_i, y_te = build_examples(test_rows, test_persona_rows)
    # 3) Embed (batch) and cache (optional but recommended)
    emb_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    tr_p_emb = batched_encode(emb_model, tr_p, batch_size=64)
    tr_i_emb = batched_encode(emb_model, tr_i, batch_size=64)
    va_p_emb = batched_encode(emb_model, va_p, batch_size=64)
    va_i_emb = batched_encode(emb_model, va_i, batch_size=64)
    te_p_emb = batched_encode(emb_model, te_p, batch_size=64)
    te_i_emb = batched_encode(emb_model, te_i, batch_size=64)

    X_tr = make_features(tr_p_emb, tr_i_emb)
    X_va = make_features(va_p_emb, va_i_emb)
    X_te = make_features(te_p_emb, te_i_emb)

    
    print("y_tr dist:", Counter(y_tr))
    print("y_va dist:", Counter(y_va))
    print("y_te dist:", Counter(y_te))

    print(f"Feature shapes: X_tr={X_tr.shape}, X_va={X_va.shape}, X_te={X_te.shape}")
    

    # 4) Train ordinal model
    # y is 0..4 ordinal classes
    alphas = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    best = None
    for a in alphas:
        ord_model = mord.LogisticAT(alpha=a)
        ord_model.fit(X_tr, y_tr)


        tr_acc, tr_mae = eval_model(ord_model, X_tr, y_tr, "train (for alpha tuning) alpha="+str(a))
        va_acc, va_mae = eval_model(ord_model, X_va, y_va, "val (for alpha tuning) alpha="+str(a))
        print(f"alpha={a:<7}  train acc={tr_acc:.3f} mae={tr_mae:.3f} | val acc={va_acc:.3f} mae={va_mae:.3f}")

        if best is None or va_mae < best["va_mae"]:
            best = {"alpha": a, "model": ord_model, "va_mae": va_mae, "va_acc": va_acc}

    print("\nBest on val:", best["alpha"], "val_mae=", best["va_mae"], "val_acc=", best["va_acc"])

    te_acc, te_mae = eval_model(best["model"], X_te, y_te, "test")
    print(f"Test set results: acc={te_acc:.4f}  MAE(stars)={te_mae:.4f}")






    best_model = best["model"]

    va_p_blank = [""] * len(va_p)
    va_p_blank_emb = batched_encode(emb_model, va_p_blank, batch_size=64)
    X_va_blank = make_features(va_p_blank_emb, va_i_emb)  # or just p*i case

    y_pred_blank = best["model"].predict(X_va_blank)
    print("VAL acc (blank persona):", accuracy_score(y_va, y_pred_blank))
    print("VAL mae (blank persona):", mean_absolute_error(y_va, y_pred_blank))


    # # 5) Evaluate
    # eval_model(ord_model, X_tr, y_tr, "train")
    # eval_model(ord_model, X_va, y_va, "val")
    # eval_model(ord_model, X_te, y_te, "test")

    # 6) Save model + embeddings if you want (pickle ord_model, np.save embeddings)
    # (mord models are sklearn-like; you can pickle them)
    model_path = Path("util_score_model/ordinal_model.pkl")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    import pickle
    with open(model_path, "wb") as f:
        pickle.dump(best_model, f)
    

if __name__ == "__main__":
    main()
