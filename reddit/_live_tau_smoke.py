"""Live smoke test for the tau2 path before an overnight GEPA run.

Run in the SAME shell where NAUT_API_KEY is exported (needs real network):

    python _live_tau_smoke.py

It exercises exactly the layers that never ran live yet:
  1. Raw tau2 sim through litellm  -> catches SSL / `openai/kimi` routing errors
     with a clear traceback (isolated from the adapter's broad try/except).
  2. One full real single-instance eval (persona gen -> grounding -> PVQ align ->
     utility -> YAML compile -> tau2 sim -> behavioral judge).

If both sections print without an SSL/auth/routing error and you see a non-empty
transcript + a tau score, the overnight run is safe. This writes NO artifacts.
"""

import os

assert not os.getenv("MOCK_LLM"), "unset MOCK_LLM: this must be a LIVE test"
assert os.getenv("NAUT_API_KEY"), "export NAUT_API_KEY first"
# Force the tau sim to run even on a low-grounding persona so we actually test it.
os.environ["GROUNDING_SKIP"] = "0.0"

import personaAdapter as pa  # noqa: E402
import run_gepa_eval as r  # noqa: E402

SAMPLE_YAML = """persona_profile:
  communication_style:
    formality: Low
    sentence_structure: short, clipped
    vocabulary_and_lexicon: blunt, impatient
    punctuation_and_formatting: minimal
    example_utterances:
      - "Just fix it, I don't have all day."
      - "Why is this so complicated?"
  interaction_policy:
    authority_challenge: Active
    escalation_trigger: being asked to repeat myself
    gratification_delay_tolerance: Low
    policy_friction_tolerance: Low
    verification_patience: Low
  state_transition_rules:
    - "IF stonewalled THEN escalate"
    - "IF asked for data THEN push back"
    - "IF looped twice THEN demand a manager"
  termination_success: "got what I came for"
  termination_abandonment: "gave up and left angry"
"""

print("=" * 60)
print(f"[1] Raw tau2 sim  (domain={r.TAU_DOMAIN}, model={r.TAU_AGENT_LLM})")
print("=" * 60)
sim = r.run_evaluation(SAMPLE_YAML, user_id="smoke_test_user")
transcript = r.clean_transcript_for_judge(sim)
print(f"termination_reason: {getattr(sim, 'termination_reason', '?')}")
print(f"transcript chars: {len(transcript)}")
print(transcript[:1200] or "<EMPTY TRANSCRIPT>")

print("\n" + "=" * 60)
print("[2] Full single-instance eval (real pipeline)")
print("=" * 60)
ds = pa.load_persona_dataset("selected_users_pvq_gepa_train_k50.jsonl")
adapter = pa.PersonaGEPAAdapter()
adapter.build_distractor_index({"train": ds, "val": ds})  # _run_dir stays None -> no logging
traj = adapter._evaluate_one(ds[0], pa.GEPA_PARAGRAPH_PROMPT)

print("\n--- RESULT ---")
print("valid:      ", traj.valid)
print("alignment:  ", traj.schwartz_alignment_score)
print("grounding:  ", traj.grounding_score)
print("utility:    ", traj.utility_score)
print("tau_result: ", traj.tau_result)
print("combined:   ", traj.combined_score)
if not traj.valid:
    print("\nWARNING: trajectory invalid -> an ACTIVE signal hard-failed. "
          "Inspect the logs above before launching overnight.")
else:
    print("\nOK: full pipeline ran end-to-end. Safe to launch the overnight run.")
