"""Offline logic smoke test. Stubs unavailable third-party deps so the pure
adapter logic (schema load, render_prompt, distractor pool, reflective dataset,
proposal validation) runs without gepa/smolagents/sentence-transformers/tau_bench.
MOCK_LLM=1 ensures no real model/network calls."""
import os
import sys
import types

os.environ["MOCK_LLM"] = "1"
os.environ["GEPA_ARM"] = "full"


def _mod(name):
    m = types.ModuleType(name)
    sys.modules[name] = m
    return m


# --- gepa ---
gepa = _mod("gepa")
gepa.optimize = lambda **kw: None
gepa_core = _mod("gepa.core")
gepa_adapter = _mod("gepa.core.adapter")


class _GEPAAdapter:
    def __class_getitem__(cls, item):
        return cls


class EvaluationBatch:
    def __init__(self, outputs=None, scores=None, trajectories=None):
        self.outputs = outputs
        self.scores = scores
        self.trajectories = trajectories

    def __class_getitem__(cls, item):
        return cls


gepa_adapter.GEPAAdapter = _GEPAAdapter
gepa_adapter.EvaluationBatch = EvaluationBatch

# --- smolagents.models ---
smol = _mod("smolagents")
smol_models = _mod("smolagents.models")
smol.models = smol_models


class _Resp:
    content = ("I am blunt and impatient and I do not suffer fools. I guard my "
               "time and my money, and I will walk away the moment support wastes either.")


class OpenAIServerModel:
    def __init__(self, *a, **k):
        pass

    def __call__(self, *a, **k):
        # Only persona GENERATION reaches here under MOCK_LLM (scoring is stubbed).
        return _Resp()


smol_models.OpenAIServerModel = OpenAIServerModel

# --- sentence_transformers ---
st = _mod("sentence_transformers")
st_ce = _mod("sentence_transformers.cross_encoder")
st.SentenceTransformer = lambda *a, **k: None
st_ce.CrossEncoder = lambda *a, **k: None
st.cross_encoder = st_ce

# --- tau_bench.run_gepa_eval ---
tb = _mod("tau_bench")
tb_run = _mod("tau_bench.run_gepa_eval")
tb_run.run_evaluation = lambda *a, **k: {}
tb_run.clean_transcript_for_judge = lambda *a, **k: ""
tb.run_gepa_eval = tb_run

import personaAdapter as pa  # noqa: E402

train_file = "selected_users_pvq_gepa_train_k50.jsonl"
ds = pa.load_persona_dataset(train_file)
print(f"loaded {len(ds)} instances from {train_file}")
d0 = ds[0]
print("user_id:", d0.user_id, "| subreddits:", d0.subreddits)
print("posts (USER turns):", len(d0.posts), "| heldout:", len(d0.heldout_post.split()), "tokens")

# 1) render_prompt: assertion passes, no unfilled placeholders, corpus injected
rendered = pa.render_prompt(pa.GEPA_PARAGRAPH_PROMPT, d0)
assert "{history_str}" not in rendered and "{psych_vector_str}" not in rendered
assert "{{" not in rendered
assert d0.posts[0][:40] in rendered
assert "[USER]" in rendered  # speaker-attribution honored
print("render_prompt OK (filled, corpus + speaker labels injected, no leftovers)")

# 2) distractor pool
adapter = pa.PersonaGEPAAdapter()
adapter.build_distractor_index({"train": ds})
dist = adapter._sample_distractors(d0)
print(f"distractors: {len(dist)} (K={pa.UTILITY_K}); tier usage: {adapter._distractor_tier_counts}")
assert len(dist) == pa.UTILITY_K
assert all(t != d0.heldout_post for t in dist)

# 3) evaluate small batch (MOCK signals) + reflective dataset
batch = ds[:6]
eb = adapter.evaluate(batch, {"persona_prompt": pa.GEPA_PARAGRAPH_PROMPT}, capture_traces=True)
print("scores:", [round(s, 3) for s in eb.scores])
assert len(eb.outputs) == len(batch) == len(eb.trajectories)
assert all(t.valid for t in eb.trajectories)
rd = adapter.make_reflective_dataset({"persona_prompt": pa.GEPA_PARAGRAPH_PROMPT}, eb, ["persona_prompt"])
recs = rd["persona_prompt"]
print(f"reflective records: {len(recs)} (expect 4 bottom + 2 top = 6); rec0 keys: {sorted(recs[0].keys())}")
assert len(recs) == 6
assert "corpus_snippet" in recs[0]["Inputs"]
assert "[BOTTOM]" in recs[0]["Feedback"]
assert any("[TOP]" in r["Feedback"] for r in recs)
assert "Tau Result" in recs[0]  # full arm has tau active

# 4) proposal validation helpers
assert pa._valid_proposal("x {history_str} y {psych_vector_str} z")
assert not pa._valid_proposal("no placeholders")
assert pa._strip_fences("```\nhello\n```") == "hello"
assert pa._strip_fences("```text\nhi\nthere\n```") == "hi\nthere"
print("proposal validation + fence strip OK")

# 5) arm switch wiring
print("ARM:", pa.ARM, "| weights:", pa.ARM_CONFIG)
assert (pa.W_ALIGN, pa.W_TAU, pa.W_UTILITY) == (0.4, 0.3, 0.3)

# 6) value_only arm omits behavioral fields from reflection
os.environ["GEPA_ARM"] = "value_only"
import importlib  # noqa: E402
importlib.reload(pa)
adapter2 = pa.PersonaGEPAAdapter()
adapter2.build_distractor_index({"train": ds})
eb2 = adapter2.evaluate(ds[:6], {"persona_prompt": pa.GEPA_PARAGRAPH_PROMPT}, capture_traces=True)
rd2 = adapter2.make_reflective_dataset({"persona_prompt": pa.GEPA_PARAGRAPH_PROMPT}, eb2, ["persona_prompt"])
r2 = rd2["persona_prompt"][0]
assert "Tau Result" not in r2, "value_only must omit Tau from reflection"
assert "Tau" not in r2["Feedback"] or "Grounding" in r2["Feedback"]
print("value_only arm omits behavioral fields OK; weights:", pa.ARM_CONFIG)

print("\nALL SMOKE CHECKS PASSED")
