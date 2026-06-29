# Handoff Addendum — Reddit Persona Fixes (from preview output)

> Companion to `/home/yash/multiagent_human_worker/reddit/handoff.md` + `/home/yash/multiagent_human_worker/reddit/persona_pipeline_datadesigner.py`.
> This addendum is based on inspecting two real preview personas: one GEPA/Reddit-seeded
> (user `drunkentune`, a combative philosophy grad student) and one Nemotron baseline
> (`Hassan Fields`, 19, Dallas). The pipeline runs end-to-end; these are the fixes the
> preview surfaced before scaling to N. Apply to the **Reddit/GEPA arm only** — the
> Nemotron arm is fine as-is.

---
 
## What the two personas told us
 
**Nemotron (the reference for what's right *and* wrong):**
- Role is CORRECT — utterances are customer-side ("Hey, I need help figuring out why my
  account isn't letting me log in"). **Match this.**
- Behaviorally HOLLOW — `authority_challenge: Neutral`, polite escalation, emojis,
  "expressing thanks." Pleasant by construction. Schwartz vector is decorative (no
  behavioral source). **Do NOT match this — preserving edge is the whole thesis.**
- Demographics richly sampled (their strength). We don't compete here.
- (Strip the fake PII — SSN/email/phone — we don't need it.)
**GEPA/Reddit (what we're fixing):** behaviorally rich *source* but four defects below.
Net: once fixed, GEPA should look as customer-shaped as Nemotron **but keep the abrasive
register Nemotron flattens away.** That contrast is the paper's argument.
 
---
 
## Fixes (apply in this order; first two gate scaling)
 
### 1. Role inversion — persona is written as the support AGENT, not the customer
**Evidence:** paragraph says "into my work as a support specialist," "I see each ticket as
a little puzzle"; example utterances are troubleshooter lines ("What exactly did you see?
Which step triggered the error?"); rules/escalation are framed from the agent's side.
 
**Fix — in the paragraph column prompt AND the structured-compile prompt**, state the role
explicitly:
```
This persona is the CUSTOMER / END-USER who CONTACTS a support agent to get a task done
(refund, booking change, account fix). They are NEVER the agent or a support worker.
Write them as the help-SEEKER reacting to an agent.
- example_utterances are things the CUSTOMER says to the agent (requests, complaints,
  reactions) — never troubleshooting questions an agent would ask.
- interaction_policy + state_transition_rules are from the USER's POV:
  "IF the agent <does X>, THEN I <react Y>".
- escalation_trigger is what makes THIS CUSTOMER demand a human.
```
Add a cheap guard: reject/regenerate if `example_utterances` read as agent-side (e.g.,
contain "let me explain", "which step triggered", or ask the other party for error
details).
 
### 2. Provenance contamination — quoted/mocked content treated as the user's own voice
**Evidence:** grounding judge flagged that the r/badphilosophy posts (the "invisible
prince" ramble, the Vegas/miracle-drug/probation post, the "baby-dick/PCP" line) are the
user **quoting and mocking others**, not their own voice — and that the *gender* inference
traced to one of those misattributed quotes.
 
**Confirmed root cause (not a logic bug — the cleaning was never implemented):**
- `enrich_users.py` *documents* steps 2a/2b in its header but `enrich()` only does: hold out
  one post, Schwartz inference on **raw** history, demographics inference on **raw** history.
- `build_dd_seed.py:flatten_history()` joins the **raw** `history` straight into `user_corpus`,
  so even a correct enrich step wouldn't reach the seed.
- Result: 96/200 users in `seed_eval.jsonl` carry bare `>` interlocutor quotes in `user_corpus`,
  and the persona generator receives contaminated text. (The other 104 are NOT necessarily
  clean — the worst contamination is whole pasted blocks with no `>` marker.)
**Principle: re-attribution, not deletion.** What a user chooses to quote/mock is personality
signal, and quote-and-rebut threads lose meaning if you delete the quoted line (the rebuttal
dangles). Represent the corpus as **speaker-attributed turns**; never bare-delete.
 
**Implement this per-user sequence in `enrich_users.py` (replaces "Schwartz on raw history"):**
 
1. **Deterministic pre-parse (no LLM).** Split posts into lines; mark `^\s*>` lines as
   `INTERLOCUTOR` turns (reliable Reddit reply-quote syntax) — *labeled, kept, not deleted*.
   Strip URLs. **Do NOT use a `QUOTE_HEAVY_SUBS` exclude list** — whole-sub exclusion is
   wrong-granularity deletion, discards first-person framing, is arbitrary to reviewers, and
   misses the unmarked pasted blocks anyway. (Subreddit name may be passed to step 2 as a hint.)
2. **LLM attribution pass (one call per user)** over the pre-parsed corpus. Labels only the
   semantic part rules can't: `FIRST_PERSON` (own voice/views/reactions/framing — KEEP) vs
   `QUOTED` (whole pasted blocks the user reproduces/mocks). Emits:
   - `clean_corpus`: speaker-attributed turns (`USER` / `INTERLOCUTOR` / `QUOTED`), nothing deleted.
   - `mock_targets`: short summary of what they mock + stance ("ridicules pseudo-profound
     metaphysics; demands rigor") — passed into Schwartz inference as value signal.
```
   Label each segment of this user's posts. `>`-lines are already marked INTERLOCUTOR.
   - FIRST_PERSON: the user's own views, experiences, arguments, reactions, framing —
     INCLUDING how they introduce or ridicule a quote ("Good grief", "I'm going to call her
     on it"). KEEP, labeled USER.
   - QUOTED: words authored by someone else that the user is reproducing/mocking. KEEP but
     label QUOTED so extraction can ignore the body.
   Output: (1) clean_corpus with every turn labeled USER/INTERLOCUTOR/QUOTED, in order;
   (2) mock_targets — what the user chose to mock and their stance.
   Subreddit context (hint, not a rule): {{ subreddit }}
```
 
3. **Holdout — reorder to AFTER attribution.** The header runs holdout *before* attribution,
   which can select a QUOTED block as the behavioral target (someone else's words). Select the
   held-out post only from `USER`/`FIRST_PERSON` posts.
4. **Schwartz re-inference** on `USER` turns of `clean_corpus` (heldout removed), with
   `mock_targets` as additional input.
5. **Demographics: drop** (see fix #3). Don't infer them.
   Persist per user: `clean_corpus`, `mock_targets`, `heldout_post`, `target_vector`.
**Plumbing fix in `build_dd_seed.py`:** point `flatten_history` at `clean_corpus` (the labeled
turns), not raw `history`, and carry `mock_targets` through as its own seed column. This is the
line that currently leaks the raw text past all cleaning.
 
**Data Designer prompts:** build the persona from `USER` turns; treat `INTERLOCUTOR`/`QUOTED`
turns only as context for *how* the user argues/reacts; never attribute quoted views to the user.
 
Keep the `grounding` judge as the **post-hoc auditor** (defense in depth), not the filter.
 
**Cost/validation:** one extra LLM call per user, once, offline (~200 users — negligible); stub
it under `MOCK_LLM=1`. The **96 quote-carrying users are your test set** — after re-running,
grep `seed_eval.jsonl` for bare unlabeled `>` in `user_corpus` (should be gone), and re-run the
grounding judge on a few including `drunkentune` (should stop flagging misattribution). Spot-check
that the attribution pass isn't over-stripping legitimate first-person argument as quotation —
over-eager attribution would flatten the abrasive register you're trying to keep.
 
### 3. Demographics — DROP them entirely for the Reddit arm (decided)
**Evidence:** grounding judge flagged age/gender/location as unsupported; only occupation was
grounded. **Decision: remove demographics from the Reddit/GEPA persona altogether.** They add
fabrication risk for no behavioral benefit, and "we ground behavior, not demographics" is the
cleaner framing against Nemotron (whose strength *is* sampled demographics).
 
**Implementation:**
- Make `demographics` optional in `PersonaProfile` (e.g. `Optional[str] = None`), or use a
  Reddit-arm schema variant without it. The Reddit arm omits it; the **Nemotron arm keeps it**
  (its sampled demographics are the intended baseline strength — leave that arm untouched).
- Generation now anchors on the cleaned first-person corpus + value vector + MOCK_TARGETS,
  not a demographic anchor. That's purer grounding, not a loss.
- Don't reintroduce demographics later as "inferred." If a reviewer wants them for realism,
  the answer is the Nemotron arm, not confabulation in the GEPA arm.
### 4. Register preservation — the generator launders hostile users into polite helpers
**Evidence:** source user is caustic (held-out post confirms it); persona became "a sincere
willingness to help" with tame rules. This kills the failure-surfacing signal that justifies
the whole method.
 
**Fix:**
- In the paragraph prompt, add an explicit anti-normalization instruction:
```
  Preserve the source user's affect and register faithfully. If the source is blunt,
  sarcastic, impatient, or hostile, the persona MUST read that way AS A CUSTOMER. Do not
  sand it into a polite, agreeable helper. Realistic difficulty is the point.
```
- Add a **register-grounding judge** that uses the held-out post as a tonal ground truth:
```
  Compare the persona's communication_style + example_utterances against the user's
  FIRST-PERSON corpus and this held-out post. Score 1-5: does the persona match the source's
  REGISTER and AFFECT (tone, bluntness, hostility, politeness)? A persona noticeably more
  polite/agreeable than the source scores LOW.
  HELD-OUT POST: {{ heldout_post }}
```
  For `drunkentune` this should score LOW today — that's the signal that stops the laundering.
 
### 5. Moderate-value collapse — watch, don't necessarily block
**Evidence:** value_alignment judge (4) noted Stimulation (0.45) and Universalism (0.3) are
under-represented while dominant + extreme-low values are captured. Generation may be
bimodal (nails peaks/floors, flattens the middle).
 
**Action:** track across the batch. If systematic, either nudge generation to express
mid-range dims or report it honestly as a known limitation. Don't over-engineer before you
see whether it's a one-off.
 
---
 
## Validation gate before scaling to N
 
Re-run preview on ~5–10 users **including at least one abrasive source (`drunkentune`)** and
confirm all of:
1. `example_utterances` are customer-side (help-seeking), not agent-side.
2. `grounding` judge no longer flags misattributed/quoted content as the user's own.
3. `register` judge: persona's tone matches the held-out post (abrasive sources stay abrasive).
4. Reddit personas have NO demographics field (dropped); Nemotron personas still do.
5. Schwartz vector still copied verbatim into `cognitive_profile` (don't regress fix P1).
Only scale once 1–4 hold on the abrasive case.
 
---
 
## Do NOT touch
- The Nemotron arm (role is correct; it's the intentional behaviorally-hollow baseline).
- The fixed OCEAN+Schwartz schema and deterministic YAML (working).
- The judge columns' existence (working — extend grounding with the register check, keep both).
- GEPA itself (`Persona_Adapter.py`) — offline, unchanged; these are generation/pre-processing fixes.