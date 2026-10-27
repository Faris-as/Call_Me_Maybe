# Call Me Maybe: Implementation PLAN v2
 
Written 2026-10-09; decisions locked the same day (section 11). It replaces nothing: your original "Implementation PLAN" is untouched. v2 keeps your stage-and-proof method but is rebuilt around the subject (v1.5 PDF), what was learned from two reference repos (AlexHysel, eepylaurie), and measurements taken on eepylaurie's code.
 
## 0. Is a chat template allowed? Yes, and it is not hardcoding.
 
**What a chat template is.** Qwen3 was fine-tuned on conversations written in a fixed text layout with special marker tokens (`<|im_start|>`, `<|im_end|>`). The template is just that wrapper around your input: a system message, the user's message, and the start of the assistant's turn. Without it the model sees raw text and has to guess what role it plays.
 
**Hardcoding vs. a template.** Page 14 forbids hardcoding *solutions based on the provided examples*: rules like "if the prompt contains 'sum' return fn_add_numbers", or a lookup from the 11 test prompts to answers. A template fixes only the **format**. The same wrapper surrounds every prompt, and the function list inside it is built at runtime from whatever `functions_definition.json` contains. No answer is stored anywhere. The same goes for the fixed JSON skeleton `{"name": "..."`: the subject requires that output shape, while names and values stay open.
 
| Subject rule (page) | What the template does |
|---|---|
| No torch / transformers / HF / outlines in your code (p8) | The template is a plain Python string you build yourself. Never call `apply_chat_template`. It lives on the HF tokenizer, which is also the SDK's private `_tokenizer`, so it is forbidden twice over. |
| Don't rely on prompting for valid JSON (p13) | You don't. The FSM and mask guarantee structure. The prompt only steers which valid option the model prefers. |
| Function chosen by the LLM, not heuristics (p8) | Still chosen by logits. The template only helps the model decide. |
| No private SDK members (p8) | `encode` is public, so it is allowed. Avoiding it is a bonus, not a requirement (p18). |
| Do not hardcode from the examples (p14) | Any few-shot examples use invented functions, never the provided tests. |
 
**Verify in Stage 3** (could not fetch the model card while writing this): (a) `encode("<|im_start|>")` returns one special-token id, not the literal characters; (b) the exact Qwen3 layout, with the empty `<think>\n\n</think>\n\n` block after `<|im_start|>assistant\n` for non-thinking mode.
 
## 1. Subject checklist (every item gets a test or a manual check)
 
| # | Requirement | Page |
|---|---|---|
| R1 | Python ≥3.10, flake8, type hints, mypy flags, PEP 257 docstrings | 7 |
| R2 | Makefile rules: install, run, debug, clean, lint (lint-strict optional) | 7-8 |
| R3 | **All classes use pydantic**; numpy and json allowed; Qwen3-0.6B | 8 |
| R4 | LLM picks the function; no private `llm_sdk` members | 8 |
| R5 | uv venv; the reviewer runs only `uv sync`; `llm_sdk/` sits beside `src/` | 8 |
| R6 | Never crash; clear error messages; handle missing or invalid input files | 8, 11 |
| R7 | `uv run python -m src [--functions_definition F] [--input F] [--output F]`; defaults `data/input/` → `data/output/` | 9 |
| R8 | Output is exactly `prompt`, `name`, `parameters`; types match; all args present; no extras | 14 |
| R9 | ≥90% accuracy, 100% valid JSON, all prompts under 5 min, robust | 15 |
| R10 | Inputs change at review: no hardcoding | 14 |
| R11 | Edge cases: empty strings, large numbers, special characters, wrong types, ambiguous prompts, multi-parameter functions | 15 |
| R12 | README sections (first line italic, plus algorithm, design, performance, challenges, testing, example usage, resources and AI usage) | 16 |
| R13 | Repo has src/, pyproject.toml, uv.lock, llm_sdk/, data/input/, README; **no output/ directory** | 19 |
| R14 | A live code modification may be requested during the defense | 19 |
 
## 2. Architecture
 
```
prompt.py ─► encode ─► ids ─► MODEL ─► logits ─► MASK ─► argmax ─► token id
                         ▲                         ▲                  │
                         │              (FSM state + vocabulary)      │
                         └────────────── append ids ◄─────────────────┘
```
 
**Terminology (two different things share the word "decode").** The SDK's `decode(ids)` only turns token ids back into text (detokenizing). The SDK has **no generate function**: `get_logits_from_input_ids` runs the network once and returns scores for ONE next token. The repeated loop that picks a token, appends it and asks again is ours. In this plan that loop is called the **generator** (`generator.py`), which is also the name in your original file structure.
 
| File | Job | Imports llm_sdk? |
|---|---|---|
| models.py | Pydantic contracts for inputs and output | no |
| io_utils.py | Load and save JSON; one `InputError` type | no |
| vocabulary.py | Token id ↔ real text; precomputed lookups | no |
| prompt.py | Compact chat-template prompt | no |
| call_fsm.py | Character-level rules for the call template | no |
| token_mask.py | FSM state + vocabulary → allowed token ids | no |
| generator.py | The token-by-token loop; model injected as callables | no |
| pipeline.py | Wires SDK → generator | **yes, only here** |
| `__main__.py` | argparse, orchestration, exit codes | via pipeline |
 
This replaces your `json_fsm.py` and `token_filter.py`. The old generic-JSON FSM validated any JSON, so `"name": "fn_banana"` would have passed. Schema-awareness is the core of the project, not a Stage 8 extra.
 
## 3. Key design decisions (be able to defend each)
 
- **D1. Template with holes, not generic JSON.** The output is always `{"name": "<one of N>", "parameters": {"<known key>": <typed value>, ...}}`. Only the name and the values are open.
- **D2. Segments.** Once the name is chosen, compile the rest into a list: `[Literal('", "parameters": {"a": '), Value(number), Literal(', "b": '), Value(number), Literal('}}')]`. Literals are forced. Values have type rules. This is a small FSM that is easy to teach, test and modify live. Build forced text with `json.dumps(key)` so odd names and keys stay valid.
- **D3. Character rules, token masks.** The FSM speaks characters. A token is a string of characters, so "is this token legal?" means simulating its characters on a copy of the state.
- **D4. Fast masks (the speed core).** Brute force costs about 1.7 s per step inside string values (measured on eepylaurie's design with a synthetic 151k vocabulary). Three accelerators, plus the brute-force version kept as the reference for tests:
  1. *Choice states (name):* legal tokens are exactly the tokens whose text is a prefix of some candidate's remaining text. That is a few dict lookups in `text → ids`, not a scan of 151k tokens.
  2. *String states:* precompute once the "plain" tokens (no `"`, no `\`, no control characters, no U+FFFD). They are always legal inside a string. Simulate only the few thousand tokens containing `"` or `\`.
  3. *Number states:* precompute the digit-only tokens. Cache masks keyed by a small state signature (sign, leading zero, has dot, at cap). The same few states repeat across steps and prompts.
- **D5. Skip the model when nothing is open.** If the FSM allows exactly one next character, don't call the model. Append `encode(forced_text)` and advance the FSM. The SDK has no KV cache, so every model call re-reads the whole context. Skipped calls are the biggest saving. Expect roughly 10-20 calls per prompt instead of about 60 (an estimate; the Stage 9 call counter confirms it).
- **D6. Termination by construction.** String values have a max length. At the cap only `"` is legal. Number values have a max digit count. After that only completing moves are legal. A global step cap is the last resort. Degenerate loops then cannot run forever.
- **D7. Parse, then coerce by schema.** `number` accepts `-?digits(.digits)?`, so the model writes the natural `265`. After `json.loads`, coerce `number` values to `float`, giving `265.0` as the subject's example shows. `integer` stays an int. This saves about 2 model calls per number compared with forcing `.d`.
- **D8. Injection.** The generator takes `logits_fn` and `encode_fn`. All logic is testable with a fake model.
- **D9. Errors.** One `InputError` type; validate item by item; write the output atomically (temp file, then `os.replace`) after every prompt, so a killed run keeps its progress.
- **D10. Prompt.** Chat template, compact signatures, no examples taken from the provided tests (R10). Prompt length drives runtime (see section 6).
 
Greedy token-level choice compares only the first token where candidate names diverge. If function-selection accuracy falls short in Stage 11, one lever is to score full names by summed log-probability. That costs N times the calls, so measure before adding it.
 
## 4. Stages
 
Each stage lists Goal, Why here, Proof, Watch-outs and a Check question.
 
### Stage 0: Environment proof (10 min)
- **Goal:** `uv sync` works and the model loads. Time the load. Test with `HF_HUB_OFFLINE=1` after the first download.
- **Why here:** everything depends on it.
- **Proof:** `uv run python -c "from llm_sdk import Small_LLM_Model; Small_LLM_Model()"` runs.
- **Watch-outs:** path dependency `llm-sdk` in `pyproject.toml`; commit `uv.lock`. On the 42 Linux machines, torch plus the ~1.5 GB model can exceed the home quota: check your free space and, if needed, point `HF_HOME` and the uv cache at a roomier folder (dev-time only; the reviewer just runs `uv sync`).
- **Check:** why must the reviewer's `uv sync` alone be enough to run the project?
 
### Stage 1: Data contracts (`models.py`)
- **Goal:** Pydantic models for FunctionDefinition, ParameterSpec, PromptEntry and FunctionCallResult.
- **Why here:** every later stage assumes these shapes.
- **Fixes to your current file:**
  - `ParamType` needs `integer` and `boolean` (the subject names number, string, boolean "etc.", and mentions integer vs float).
  - Remove the invented limits (`min_length=5` on description and name). They reject legal inputs such as an empty description.
  - Give `description` a default, make `returns` optional, and use `extra="ignore"` on input models.
  - Give the output model `extra="forbid"`.
  - Add a validator for unique function names.
- **Proof:** load all 5 provided functions and all 11 prompts, plus deliberately odd inputs: an extra key, an unknown type, a duplicate name.
- **Check:** why `dict[str, Any]` for result parameters but `dict[str, ParameterSpec]` for definitions?
 
### Stage 2: I/O layer (`io_utils.py`)
- **Goal:** `load_function_definitions`, `load_prompts`, `save_results`, all raising `InputError(msg)`.
- **Why here:** R6. It also prevents confusing failures later.
- **Fixes to your draft:** printing and returning `[]` hides the failure, so the program carries on with no functions. Your `read_func_definition` also falls through after the "file is empty" message, and `read_prompts` doesn't stop on a non-list.
- **Behaviour:**
  - Read with `utf-8-sig`, so a byte-order mark doesn't break parsing.
  - Validate item by item and report the index.
  - An empty prompt list is allowed and writes `[]`.
  - An empty function list is an error.
  - **Decided:** an unsupported parameter type skips only that function, with a warning.
  - Save: `mkdir(parents=True)`, temp file, `os.replace`, `ensure_ascii=False`.
- **Proof:** pytest with `tmp_path`: missing, empty, invalid JSON, wrong shape, bad item, unwritable output path.
- **Check:** why write to a temp file and replace, instead of writing straight to the output path?
 
### Stage 3: SDK sandbox and measurements (throwaway script, not committed)
- **Goal:** see real tokens and logits, and measure **on the 42 Linux machine**, since its hardware is unknown.
- **Why here:** later budgets and design choices rest on these numbers.
- **Do and record:**
  - encode/decode round trip.
  - Length of the logits list vs the number of entries in `vocab.json` (expected 151936 vs 151643; verify).
  - Top-5 logits with their token strings.
  - Does `encode("<|im_start|>user")` give a special id? (section 0)
  - How are digits tokenized? (expected one digit per token; verify)
  - Time of `get_logits_from_input_ids` at 50, 200, 400 and 800 tokens.
- **Proof:** a table of timings and a short list of verified facts.
- **Check:** why does a longer prompt make every generation step slower?
 
### Stage 4: Vocabulary (`vocabulary.py`)
- **Goal:** token id → real text, plus the lookups the masks need.
- **Why here:** the FSM speaks characters and the model speaks ids. This bridges them.
- **Build once at startup:**
  - Invert GPT-2's byte↔unicode table (`Ġ` is a space, `Ċ` a newline), then decode the bytes as UTF-8.
  - `None` for tokens that aren't valid UTF-8 on their own or that are special.
  - `text → [ids]` dict.
  - Plain-string and digit-only id sets as numpy arrays.
- **Pydantic note:** use `PrivateAttr` for the big containers. Validating 150k entries on init is slow (AlexHysel hit exactly this).
- **Proof:** for 2000 random ids, your text equals the SDK `decode([id])` where both are valid. Every printable ASCII character has a single-character token. This is an invariant the FSM relies on.
- **Check:** why can't `vocab.json` keys be used as text directly?
 
### Stage 5: Baseline loop (dev script, short)
- **Goal:** unconstrained argmax loop with the chat template, run on the provided prompts.
- **Why here:** it proves loop mechanics. It also gives you the "30% valid JSON" evidence for the README Challenges section.
- **Proof:** it runs; count how many outputs pass `json.loads`.
 
### Stage 6: Prompt builder (`prompt.py`)
- **Goal:** system message with compact signatures (`fn_name(a: number, b: number): description`), the user message, then `<|im_start|>assistant\n<think>\n\n</think>\n\n`. State the output shape and rules briefly ("copy values exactly; numbers as digits").
- **Why here:** prompt length drives speed, so fix the budget before tuning.
- **Proof:** print it, count tokens for 5 and 20 functions, record the per-function cost.
- **Watch-outs:** no examples copied from the provided tests (R10). Optional hardening: neutralise literal `<|...|>` text inside user prompts, since `encode` may turn it into special tokens.
- **Check:** does the prompt guarantee valid JSON? What does?
 
### Stage 7: Call FSM (`call_fsm.py`), pure, no LLM
- **Goal:** segments from D2, with value rules:
  - `string`: opening quote is part of the preceding literal; content except `"`, `\`, control characters; valid escapes; closing quote ends it; cap.
  - `number` / `integer`: optional `-`, no leading zeros, optional fraction for `number`; "complete but extendable", and the next segment's first character ends it.
  - `boolean`: choice of `true` / `false`.
  - Name: choice among candidates written as `name + '"'`, so a name that is a prefix of another still works.
- **API:** `allowed_next() -> set[str]`, `advance(ch)`, `copy()`, `is_complete()`, `forced_text()`.
- **Pydantic:** state as a small BaseModel; `model_copy` for simulation.
- **Proof, three kinds of test:**
  1. Golden strings accepted and one-character mutations rejected.
  2. **Fuzz:** 1000 seeded random walks (pick a random allowed character each step). Every finished walk must `json.loads` and match the schema (types, keys, order).
  3. Run those walks across zero-parameter functions, prefix names, odd names, and all four types.
- **Check:** why is a number "complete" but not "finished"?
 
### Stage 8: Token mask (`token_mask.py`)
- **Goal:** FSM state + vocabulary → allowed ids (D4).
- **Why here:** the FSM and the vocabulary must both be trusted first.
- **Proof:** **differential test.** The fast mask must equal the brute-force mask on a small synthetic vocabulary with tricky multi-character tokens (`"}`, `.0`, `\"`, `",`). Repeat on sampled states with the real vocabulary.
- **Watch-outs:** the mask array length is `len(logits)`, not the vocab size. Ids above the vocabulary are always illegal. If every token is masked to `-inf`, `argmax` silently returns index 0, so the code must raise instead.
- **Check:** why is simulating a whole token different from checking only its first character?
 
### Stage 9: Generator (`generator.py`)
- **Goal:** the loop:
 
```
while not fsm.done:
    forced = fsm.forced_text()
    if forced: ids += encode(forced); fsm.advance_all(forced); continue
    ids_ok = mask(fsm); logits = logits_fn(ids)
    token = argmax over ids_ok; ids.append(token); fsm.advance_all(text[token])
```
 
- **Plus:** caps (D6), a step cap raising `DecodingError`, an optional `on_step` callback for a verbose trace (chosen bonus), and a model-call counter.
- **Proof, with fake logits:** uniform, adversarial (always prefers an illegal token), random, and "prefers end-of-text". Every run must produce valid, schema-correct output. A biased fake model must steer the chosen function (shows the model decides, R4).
- **Check:** if the model's top token is illegal, what does argmax return, and why is the output still valid?
 
### Stage 10: Pipeline and CLI (`pipeline.py`, `__main__.py`)
- **Goal:** the real run.
- **Behaviour:**
  - argparse per R7.
  - Model-load failure gives a clear message (including the offline first run).
  - **Decided:** a prompt that fails is skipped with a warning and the run continues. Per-prompt try/except. Write the output after each prompt. The summary line reports how many were skipped, and the exit code is 0 once the output file is written.
  - Handle KeyboardInterrupt by saving partial results.
  - Build the result from `json.loads(raw)`, coerce `number` to float, then copy the original `prompt` verbatim.
- **Proof:** run the 11 prompts. Validate the output with an independent checker script written from the subject's validation rules (p14), not from your own models.
 
### Stage 11: Accuracy and speed harness (dev script)
- **Goal:** measure R9 honestly.
- **Dev set:** at least 25 prompts over at least 3 function sets, with unseen functions (boolean, integer, multi-string, zero-parameter) and every R11 edge case. Keep a separate held-out set you never tune on.
- **Method:** exact-match scoring on name and parameters. Change one prompt variable at a time (with/without think block, with/without two invented-function few-shot examples, description wording). Log failures.
- **Speed:** total seconds and model calls per prompt, on the 42 machine. Target 2 minutes or less for 11 prompts, to leave margin under 5.
- **Feeds:** the README Performance analysis.
 
### Stage 12: Polish and submission
- Makefile (install, run, debug, clean, lint, lint-strict).
- flake8 clean; mypy with the subject's flags, then try `--strict`.
- `.gitignore` (output, caches, .venv).
- README from the section 8 map.
- **Fresh-clone test:** clone to a temp directory, `uv sync`, `uv run python -m src`, then delete `data/output`.
- Check `uv.lock` is committed and `llm_sdk/` is untouched.
 
## 5. Durability matrix: input variation → handling
 
| Input variation | Handling |
|---|---|
| Function name is a prefix of another | Candidates carry the closing quote (Stage 7) |
| Zero-parameter function | No value slots; output `"parameters": {}` |
| Odd characters in names or keys | Forced text via `json.dumps` |
| 20+ functions | Compact signatures; measure prompt cost (Stage 6) |
| Unsupported parameter type (array, object, null) | Skip that function with a warning; one extension point per type |
| Duplicate function names | `InputError` |
| Empty, long, quoted, backslashed, multi-line, emoji prompts | Prompt copied verbatim; special-token text neutralised |
| Numbers: negative, decimal, huge, leading zeros | FSM rules, digit cap, float coercion |
| Strings: empty, escapes, accents | Escape rules; whole-character tokens; length cap closes the string |
| Ambiguous prompt, no matching function | Still returns a valid call, since the output has no "refuse" field (document it) |
| Missing, empty, invalid, wrong-shape, BOM input files | `InputError` with a clear message |
| Output directory missing or unwritable | `mkdir`; `InputError` |
| Degenerate model loops | Caps and step limit |
| One prompt fails | Skipped with a warning; the rest continue |
| Interrupted run | Partial results already written |
| cpu / cuda / mps | Same logits list from the SDK |
 
Known limit (decided): strings can only contain characters that exist as whole-character tokens. Rare characters split across byte tokens are unreachable.
 
## 6. Speed budget
 
Cost per prompt is about (model calls) × (time per call at the current context length). No KV cache means each call reads the whole prompt again. The levers, in order of payoff: skip forced calls (D5), keep the prompt short (Stage 6), compute masks cheaply (D4), and keep numbers natural (D7). Fill the real figures in after Stage 3 and Stage 11; any number above is an estimate until then.
 
## 7. Tests (pytest, not graded but you defend them)
 
- FSM: golden, mutation, fuzz (Stage 7).
- Mask: differential against brute force (Stage 8).
- Generator: fake models (Stage 9).
- I/O: `tmp_path` cases (Stage 2).
- CLI: call `main(argv)` directly.
- Harness: separate script with a `slow` marker.
 
## 8. README map (R12)
 
First line italic: *This project has been created as part of the 42 curriculum by <login>.*
 
| Section | Source material |
|---|---|
| Description | Goal; the 30% baseline from Stage 5 |
| Instructions | `make install`, `make run`, custom paths |
| Algorithm explanation | D1-D5, one worked example with real trace |
| Design decisions | D1-D10 |
| Performance analysis | Stage 11 numbers (accuracy, validity, seconds, calls) |
| Challenges faced | Your real problems and fixes |
| Testing strategy | section 7 |
| Example usage | CLI examples with input and output |
| Resources | Docs plus an honest AI-usage description |
 
## 9. Defense drills (R14)
 
Be able to do each of these in a few minutes, live:
1. Add a new value type (`null`, say).
2. Change float output formatting.
3. Change the string length cap.
4. Print the number of legal tokens per step.
5. Change the prompt template.
6. Explain why `"fn_banana"` can never be produced.
7. Explain what happens when the model's favourite token is illegal.
8. Explain where the model decides and where the code decides.
 
## 10. Bonus (p18)
 
**Chosen (simple):** verbose generation trace (Stage 9), mask caching (D4), comprehensive test suite (section 7). Not planned: multiple models, recoding the tokenizer (would use `get_path_to_merges_file`).
 
## 11. Decisions (locked 2026-10-09)
 
1. **Hardware:** 42 Linux machines, specs unknown. Measure in Stage 3 and Stage 11 on that machine; budget to 2 minutes for 11 prompts.
2. **Bonus:** simple ones only (section 10).
3. **A prompt that fails:** skip it with a warning and continue; exit code 0 once the output is written.
4. **Unsupported parameter type:** skip only that function, with a warning.
5. **Strings:** whole-character tokens only.
