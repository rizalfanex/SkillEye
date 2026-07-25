# LLM-Generated Correction Paragraph — Design

Date: 2026-07-23
Status: Approved for implementation

## Problem

The quality-scoring system (`quality/score.py`) already produces a per-(phase, joint)
deviation table and a list of fixed-template coaching-tip sentences (`JOINT_TIPS`) for
flagged deviations. This works, but reads as mechanical — a fixed sentence per (joint,
phase, direction) combination, not a coherent explanation a coach would actually give.

The user wants an LLM (via NVIDIA's build.nvidia.com API, an OpenAI-compatible hosted
model catalog) to rewrite the already-computed flags into one natural paragraph per
swing. No API key exists yet — the user will supply it later via an environment
variable — so this design must work correctly once a key is set without requiring one
to build or test against today.

This is scoped separately from the previously-discussed real-time webcam pivot; that is
an independent sub-project, not part of this design.

## Approach: LLM as a rephrasing layer over already-flagged data, never a diagnosis source

The LLM is given ONLY the rows `score_clip()` already flagged (phase, joint, z-score,
deviation direction) plus the stroke type, and is instructed to combine them into one
coherent paragraph — not to inspect raw angles itself or introduce a diagnosis the
rule-based system didn't already make. This keeps hallucination risk low: the LLM's
job is narrowly "explain this data more naturally," not "figure out what's wrong,"
which is exactly the job the already-smoke-checked rule-based system does. If the API
call fails for any reason, the system falls back to the existing `JOINT_TIPS` sentences
— the LLM layer is additive polish, never a single point of failure for the demo.

## Components

### 1. Prompt construction (`quality/llm_explainer.py`: `build_explanation_prompt`)

`build_explanation_prompt(stroke: str, table: list[dict]) -> str`. Filters `table`
(the same list `score_clip()` returns) to rows where `flagged` is `True`. For each,
includes the phase, joint display name, and deviation direction (above/below expert
range, from the sign of `z`) — the same information `suggestion_text` in `score.py`
already uses, not raw z magnitudes or unflagged rows. Instructs the model explicitly:
combine these into one coherent, encouraging paragraph as a coach would say to a
student, using only the deviations listed, inventing nothing else.

If no rows are flagged, builds a different prompt asking for a short, positive
paragraph acknowledging the clip's technique matched the expert template well — this
is itself a real, grounded fact (the absence of any flag), not a fabrication.

### 2. API call (`quality/llm_explainer.py`: `generate_explanation`)

`generate_explanation(stroke: str, table: list[dict], api_key: str | None = None, timeout: float = 8.0, post_fn=requests.post) -> str`.

- Reads the API key from the `api_key` argument if given, else `os.environ["NVIDIA_API_KEY"]`.
  Missing key raises `LLMExplanationError` immediately — no network call attempted.
- Builds the prompt via `build_explanation_prompt`, POSTs to NVIDIA's OpenAI-compatible
  chat completions endpoint (`NVIDIA_API_BASE`, default model `MODEL_NAME` — both
  module-level constants, overridable) with the bounded `timeout`.
- `post_fn` is injectable (defaults to `requests.post`) specifically so tests never
  make a real network call and never need a real key.
- Any failure — missing key, request exception, non-200 response, unexpected response
  shape — raises `LLMExplanationError` with a descriptive message. Callers catch this
  one exception type, never a bare `requests` exception or a KeyError.

### 3. Demo UI integration (`app.py`)

A "Generate AI explanation" button in the existing Correction suggestions section
(below the current bullet-list of `suggestions`, which remains and is never removed).
On click: call `generate_explanation` with the current clip's `stroke`/`table`; on
success, display the paragraph; on `LLMExplanationError`, show a short inline note
("AI explanation unavailable — showing rule-based suggestions below") and leave the
existing bullet list visible, which it already is by default.

Wrapped in `st.cache_data`, keyed on `(stroke, tuple of (phase, joint, flagged, z) for
every row)` — not on the frame slider or any other widget state — so dragging the
frame slider (which reruns the whole Streamlit script) never re-fires the API call for
a clip whose score hasn't changed.

## Data flow

```
score_clip() output (existing, unmodified)
  -> table: [{phase, joint, value, z, flagged, note}, ...]
  -> build_explanation_prompt(): flagged rows only -> grounded prompt string
  -> generate_explanation(): POST to NVIDIA API -> paragraph text
       (post_fn injectable; api_key from env if not passed)
  -> app.py: st.cache_data-wrapped, button-triggered
       success -> display paragraph
       LLMExplanationError -> inline fallback note + existing suggestions list stays visible
```

## Error handling

- `NVIDIA_API_KEY` not set and no `api_key` argument: `LLMExplanationError` before any
  network call.
- Network error, timeout (bounded at 8s), non-200 status, or a 200 response whose body
  doesn't contain the expected message-content field: all normalized to
  `LLMExplanationError` with a message identifying which of these occurred, not a raw
  stack trace surfaced to the Streamlit UI.
- Zero flagged rows: handled by a distinct prompt (Component 1), not an error path —
  this is a valid, common case (a well-scoring clip), not a failure.

## Testing

No real API key or network access is available or required for any test:

- `build_explanation_prompt`: given a table with a mix of flagged and unflagged rows,
  assert the returned prompt string mentions the flagged joints/phases and does not
  mention the unflagged ones. Separately, assert the all-unflagged case produces the
  distinct "positive" prompt variant, not the correction-prompt template.
- `generate_explanation`: inject a fake `post_fn` returning a canned successful
  response object (mimicking `requests.Response`'s `.status_code`/`.json()`) to verify
  the success path extracts and returns the expected text. Inject a fake `post_fn`
  that raises (simulating a network error) and one that returns a non-200 status, both
  asserting `LLMExplanationError` is raised. Assert that calling with no `api_key` and
  no `NVIDIA_API_KEY` in the environment (monkeypatched empty) raises
  `LLMExplanationError` without `post_fn` ever being called.

## Out of scope for this iteration

- The real-time webcam capture pipeline — a separate, independent sub-project to be
  brainstormed on its own.
- Automatic (non-button) triggering of the LLM call.
- Per-joint separate paragraphs (one combined paragraph per swing was chosen instead).
- Any evaluation of generated-paragraph *quality* — no ground truth exists for that,
  consistent with the rest of the quality-scoring system's honest v1 framing; this
  feature is a communication layer over already-validated rule-based flags, not a new
  source of truth.
- Choosing/tuning the specific NVIDIA-hosted model or prompt wording beyond a
  reasonable default — the user can change `MODEL_NAME` once they have a working key
  and a preference.
