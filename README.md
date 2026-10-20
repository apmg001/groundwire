# Groundwire

**A self-hosted, OpenAI-compatible LLM gateway that keeps answers available and honest.**

Groundwire sits between your application and any LLM, local or cloud. Point an
OpenAI SDK at it by changing one URL, and every call gets:

- **Failover across providers**: local llama.cpp first, cloud only if needed.
- **Retries with jittered backoff** and **per-provider circuit breakers**.
- **End-to-end deadlines**, so a slow upstream can't hang your app.
- **Grounding verification**: checks the answer against the context you supplied
  and flags or blocks unsupported claims (invented amounts, dates, IDs).
- **Prompt-injection screening and PII redaction** before any model sees the request.
- **Exact + semantic response cache** with request coalescing: repeat and reworded
  questions are answered without a model call.
- **Structured JSON logs** with request IDs, and **Prometheus metrics**.

Runs fully offline on a laptop CPU. No GPU, no cloud account required.

---

## Architecture

```mermaid
flowchart LR
    App[Your app<br/>OpenAI SDK] -->|POST /v1/chat/completions| API

    subgraph Groundwire
        API[API layer<br/>validation · SSE · errors] --> SVC[ChatService<br/>deadline · policy]
        SVC --> RT[Router]
        RT --> CB1{{Breaker}} --> R1[Retry] --> P1[llama.cpp]
        RT --> CB2{{Breaker}} --> R2[Retry] --> P2[OpenAI / Groq]
        RT --> CB3{{Breaker}} --> R3[Retry] --> P3[Anthropic]
        SVC --> SG[Safety guard<br/>injection · PII]
        SVC --> RC[(Response cache<br/>exact · semantic)]
        SVC --> GV[Grounding verifier<br/>lexical · NLI · token]
        SVC -.-> M[(Prometheus metrics)]
    end
```

**Layering (ports and adapters):**

| Layer | Package | Knows about |
|---|---|---|
| HTTP | `api/` | FastAPI, SSE framing, status codes |
| Use case | `services/` | Deadlines, grounding policy, metrics |
| Routing | `routing/` | Fallback chains, breakers, retries |
| Resilience | `resilience/` | Backoff math, breaker state machine |
| Guardrails | `guardrails/` | Claim splitting, support scoring |
| Safety | `safety/` | Injection rules/classifier, PII detection and vault |
| Cache | `cache/` | Keys, embeddings, store port + in-memory adapter, coalescing |
| Adapters | `providers/` | Vendor wire formats, error classification |
| Core | `core/`, `domain/` | Config, errors, logging, models |

Dependencies point inward. Adding a vendor means writing one adapter and one
line in `providers/factory.py`.

---

## Quick start

```bash
git clone https://github.com/apmg001/groundwire.git && cd groundwire
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

make check     # ruff + mypy --strict + pytest with coverage
make run       # gateway on http://127.0.0.1:8000
```

Try the built-in `demo` route (two in-process mock providers, one of which fails 50% of the time):

```bash
curl -s localhost:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model": "demo", "messages": [{"role": "user", "content": "hi"}]}'
```

The response includes a `groundwire` block showing which provider answered and every attempt made.

### With a real local model (llama.cpp, CPU)

```bash
# Download a small quantized model, e.g. Qwen2.5-3B-Instruct Q4_K_M (~2 GB)
llama-server -m qwen2.5-3b-instruct-q4_k_m.gguf --port 8080 -t 6
make run
curl -s localhost:8000/v1/chat/completions -H 'content-type: application/json' \
  -d '{"messages": [{"role": "user", "content": "Explain circuit breakers in one line."}]}'
```

To add cloud fallbacks, export the key and set `enabled: true` for that provider in `config/groundwire.yaml`.

### Docker

```bash
docker compose up --build     # gateway :8000, Prometheus :9090
```

---

## Using it from an OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")

context = ["Order 7781 was refunded 49.99 USD on 2 May."]
response = client.chat.completions.create(
    model="default",  # a route alias from the config
    messages=[
        {"role": "system", "content": "Answer only from this context:\n" + context[0]},
        {"role": "user", "content": "How much was refunded for order 7781?"},
    ],
    extra_body={"groundwire": {"grounding": {"context": context, "mode": "block"}}},
)
print(response.choices[0].message.content)
```

With `mode: "block"`, an answer that cites an amount not present in the context
is refused with HTTP 422 and a per-claim report. With `mode: "flag"` (default),
the answer is returned with the report attached.

---

## How the fault tolerance works

| Failure | Behaviour |
|---|---|
| Timeout, connection error, 5xx, 529, 429 | Retried on the same provider with full-jitter backoff (honours `Retry-After`, capped), then fail over |
| 401 / 403 | No retry (won't fix itself); fail over; counts against the breaker |
| 400 / 404 / 422 from upstream | Surfaced to the client as 400; **no** failover (the request itself is bad) |
| Provider keeps failing | Circuit opens after N consecutive failures; calls skip it instantly (no timeout cost) until a half-open probe succeeds |
| Every provider down | HTTP 503 with a record of each attempt |
| Too slow overall | HTTP 504 when `request_timeout_s` expires, across all retries and failover |
| Client disconnects | Upstream call cancelled; breaker probe slot released without a verdict |
| Grounding scorer crashes | `fail_open` → answer marked `unverified`; `fail_closed` → 503 |

**Streaming:** failover happens only before the first chunk is sent. After
that, a provider failure is delivered in-band as an SSE `error` payload rather
than silently switching models mid-sentence.

---

## Grounding scorers

| Scorer | Install | Catches | Misses |
|---|---|---|---|
| `lexical` (default) | nothing | Invented numbers, amounts, dates, times and IDs; claims with little overlap with the context | Paraphrased contradictions ("did" vs "did not") |
| `nli` | `make install-nli` | Semantic entailment and contradiction (DeBERTa cross-encoder) | Weak on exact numbers; tens of ms per claim on CPU |
| `hybrid` | `make install-nli` | Both: a numeric gate rejects wrong numbers without running the model, NLI judges the rest | Slower than lexical |
| `token` | `make install-nli` | Reads the whole prompt and answer at once and marks each answer word as supported or made up. Default model: [LettuceDetect](https://github.com/KRLabsOrg/LettuceDetect) (ModernBERT, MIT), fine-tuned on RAGTruth | Slowest; trained on RAG-style prompts, so other domains may need checking |

Numbers are normalised before comparison (`1,200.00` = `1200`), and so are clock times (`5 pm` = `5:00 PM` = `17:0`).
Scoring is CPU-bound, so it runs in a worker thread behind a concurrency limit and never blocks the event loop.

---

## Safety: prompt injection and PII

Two checks run on every request **before any model sees it** (configured under `safety:`).

**Prompt-injection screening.** User messages (direct attacks) and grounding documents
(indirect attacks hidden in retrieved content) are scanned. `block` returns HTTP 400
`prompt_injection_detected` and **no provider is called**; `flag` lets the request through and
reports findings in `groundwire.safety`.

| Detector | How | Trade-off |
|---|---|---|
| `heuristic` (default) | Weighted rules for instruction override, role hijack, system-prompt extraction, fake chat-template markers, data exfiltration via links, and text addressed to "the AI". Text is normalised first (Unicode NFKC, zero-width characters removed), so `ig\u200bnore previous instructions` still matches. | ~1 ms, explainable, but only knows the patterns it was written for |
| `classifier` | [`protectai/deberta-v3-base-prompt-injection-v2`](https://huggingface.co/protectai/deberta-v3-base-prompt-injection-v2) (Apache-2.0), long texts scored in overlapping windows | Generalises better; English only, model card advises against scanning system prompts |
| `both` | Maximum of the two | Best coverage |

**PII redaction.** Emails, phone numbers, card numbers (Luhn-checked), Aadhaar (Verhoeff-checked),
PAN and IPv4 addresses are replaced with placeholders in *every* message sent to providers, then
restored in the answer, including while streaming, where a placeholder can arrive split across chunks:

```
client  : "Refund rahul@example.com, card 4111 1111 1111 1111"
provider: "Refund [EMAIL_1], card [CARD_1]"
answer  : "Refund sent to [EMAIL_1]."  ->  client sees "Refund sent to rahul@example.com."
```

The same value always gets the same placeholder within a request, raw values are held only in
memory for that request, and only counts are logged.

**Limits, stated plainly:**

- **No filter prevents prompt injection.** Screening is one layer. The ones that matter more are
  architectural: keep decisions (refunds, permissions) in deterministic code, give the model least
  privilege, never put secrets in prompts, and check outputs (the grounding check above blocks
  unsupported claims an injected document tries to plant).
- **Regex PII detection covers structured identifiers only.** It does not reliably find names,
  addresses or business secrets. For data that must never leave your servers, route to local
  models only (`llama-local`, `ollama-local`); redaction is for when cloud models are allowed.

---

## Response cache: exact, semantic, coalesced

Configured under `cache:`. A lookup runs after PII redaction and before any provider call:

1. **Exact**: hash of the normalised question (case, spacing and trailing punctuation ignored).
   Always on and needs no extra dependencies.
2. **Semantic** (`semantic: true`, needs `make install-nli`): the question is embedded with
   [`all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
   (Apache-2.0, 22M params, CPU) and matched against earlier questions by cosine similarity.
3. **Coalescing**: identical requests that arrive while the first is still running wait for
   its answer instead of each calling the provider (no cache stampede).

```json
"groundwire": {
  "provider": "cache",
  "cache": {"status": "hit", "match": "semantic", "similarity": 0.9712, "age_s": 12.4, "tokens_saved": 21}
}
```

**When two requests may share an answer.** Only when the model would have seen the same input:
same route, sampling parameters, conversation history, grounding documents and `namespace` tag
(and the same `user`, with `partition_by_user: true`). Only the latest message is matched
semantically, and a semantic match is rejected if the two questions mention **different
numbers** (`order 7781` vs `order 7782` embed almost identically) or **different PII
placeholders**.

**Safety properties:**

- **No PII in the cache.** Keys and answers are stored in redacted form (`[EMAIL_1]`); each
  request restores its own values. Two customers asking the same question share the answer, not
  each other's data.
- **No cache poisoning via injection.** A prompt with any injection finding is never cached.
- **No caching of bad answers.** Answers that fail grounding, were cut off
  (`finish_reason: length`), or are empty are not stored.
- **Fail-open.** A broken store, a crashed embedder, or a lookup slower than
  `lookup_timeout_s` is logged and counted, and the request goes to the provider.

**Per-request control** (like HTTP `Cache-Control`): `"groundwire": {"cache": {"mode": "bypass"}}`
skips the cache, and `"refresh"` skips the read and overwrites the entry. Requests hotter than
`max_temperature` are not cached, because the caller asked for variety.

**Scaling out.** `cache/store.py` defines a `CacheStore` port. The in-memory adapter is bounded
(LRU, `max_entries`), expiring (`ttl_s`) and partitioned by namespace. For several replicas
sharing one cache, implement the port on Redis or pgvector; nothing else changes.

**Why the guards exist (measured on a laptop with MiniLM):** a question about a *different invoice*
("...INV-1043?") scored **0.964** against a cached "...INV-1042?", *higher* than a genuine paraphrase
of the same question (0.921). A threshold alone would have served one customer's invoice to another.

### Benchmark: does the semantic cache serve wrong answers?

`make bench-cache` runs the cache's own normalisation, embedder and guards on
[Quora Question Pairs](https://huggingface.co/datasets/nyu-mll/glue) (GLUE release): real Quora
questions paired and labelled by humans as *same question* or *different question*, many of them
deliberately close ("learn Python" vs "learn Java"). Each pair is treated as "question 1 is cached,
question 2 arrives". The threshold is **tuned on 20,000 train pairs** for a wrong-answer budget and
**reported on all 40,430 validation pairs**.

| Wrong-answer budget | Threshold (tuned on train) | Rewordings reused (hit rate) | Different questions answered from cache (false-hit rate) |
|---|---|---|---|
| ≤ 5% | 0.90 | 38.1% | 5.0% |
| **≤ 1% (shipped default)** | **0.96** | **14.9%** | **0.9%** |
| ≤ 0.1% | 0.99 | 3.0% | 0.1% |

`all-MiniLM-L6-v2` on a laptop CPU (i5-1235U): similarity ROC-AUC **0.874**, **14 ms p50 / 36 ms
p95** per lookup. The budget tuned on train held on validation (1% asked, 0.9% measured).

**What the numbers say, stated plainly:**

- **The first default (0.92) was too loose.** It served a cached answer to 3.5% of *different*
  questions. The shipped default is now 0.96, chosen from the data.
- **Semantic caching saves less than people assume.** At a 1% wrong-answer budget, it reuses about
  1 in 7 rewordings. Exact matching (always on) carries no such risk.
- **Small embedding models ignore word order.** "Shifting from Tirupur *to* Chennai" vs "from
  Chennai *to* Tirupur" scored 0.997. No threshold fixes that, which is why semantic matching is
  opt-in.
- **The number guard helps modestly on QQP** (wrong hits 970 → 891 at 0.92, costing 72 correct
  hits), because few Quora questions hinge on numbers. On invoice- or order-style traffic it
  matters more (the INV-1042/1043 case above).
- **The error rate is pessimistic.** Several "wrong" matches are QQP label noise ("How *can* I
  analyze a short story?" vs "How *do* I…" are labelled different).
- QQP is general Q&A, not your traffic. Re-run with your model and confirm on your own logs before
  lowering the threshold.

---

## Benchmark: does the fact-checker actually work?

Measured on **[RAGTruth](https://github.com/ParticleMedia/RAGTruth)**: 2,700 real LLM answers to
retrieval-augmented prompts (question answering, news summaries, and descriptions written from
business data), each labelled by human annotators as hallucinated or not.

The benchmark runs the **same verifier the gateway uses**. A response is flagged when its answer
score (claim scores combined by the configured aggregation) falls below a threshold. Thresholds are
either tuned on the *train* split and applied unchanged to *test*, or fixed in advance; never tuned
on the numbers they report.

### Headline: full RAGTruth test set (2,700 answers)

| Scorer | F1 | Precision | Recall | ROC-AUC | Good answers blocked | Latency p50 (laptop CPU) |
|---|---|---|---|---|---|---|
| `lexical` (rules, no model) | 64.4% | 52.8% | 82.6% | 0.783 | 39.7% | 3 ms |
| **`token`** (LettuceDetect-base) | **76.1%** | **76.6%** | 75.5% | **0.889** | **12.3%** | 975 ms |
| *Flag everything (baseline)* | *51.8%* | *34.9%* | *100%* | *0.5* | *100%* | – |

**Switching to the token scorer cut wrongly blocked good answers from 39.7% to 12.3%** while raising
F1 from 64.4% to 76.1%. It costs about a second per answer on a laptop CPU (Intel i5-1235U).

**Reproduction check.** LettuceDetect's paper reports 76.07% F1 for this model on the RAGTruth test
set (QA 65.52%, Data2txt 87.89%, Summary 50.52%). Groundwire, running it through its own
verifier pipeline, measured **76.1%** (QA 65.5%, Data2txt 87.9%, Summary 50.5%), matching to the first
decimal. That confirms the integration is faithful, not just "inspired by" the model.

> Fairness note: LettuceDetect was fine-tuned on the RAGTruth *train* split, so this is an
> in-distribution result; the test split is held out. Its benchmark uses the model's native 0.5
> threshold with no tuning. Expect lower accuracy on very different domains until measured there.

### Token scorer by task

| Task | Responses | Hallucinated | Precision | Recall | F1 | F1 if flagging everything | ROC-AUC | Good answers blocked | Latency p50 / p95 |
|---|---|---|---|---|---|---|---|---|---|
| **All tasks** | 2700 | 34.9% | 76.6% | 75.5% | **76.1%** | 51.8% | **0.889** | 12.3% | 975 / 2472 ms |
| QA | 900 | 17.8% | 60.6% | 71.2% | 65.5% | 30.2% | 0.901 | 10.0% | 725 / 1178 ms |
| Summary | 900 | 22.7% | 53.9% | 47.5% | 50.5% | 37.0% | 0.756 | 11.9% | 917 / 3380 ms |
| Data2txt | 900 | 64.3% | 89.3% | 86.5% | 87.9% | 78.3% | 0.904 | 18.7% | 1327 / 2324 ms |

Summaries remain the hardest case: it catches fewer than half of hallucinated summaries.

### Lexical scorer by task

Answer score = average of the two weakest claims (`lowest_two`), threshold tuned for best F1 on train:

| Task | Responses | Hallucinated | Precision | Recall | F1 | F1 if flagging everything | ROC-AUC | Good answers blocked | Latency p50 |
|---|---|---|---|---|---|---|---|---|---|
| **All tasks** | 2700 | 34.9% | 52.8% | 82.6% | **64.4%** | 51.8% | **0.783** | 39.7% | 3 ms |
| QA | 900 | 17.8% | 35.8% | 61.3% | 45.2% | 30.2% | 0.785 | 23.8% | 2 ms |
| Summary | 900 | 22.7% | 33.9% | 51.5% | 40.9% | 37.0% | 0.667 | 29.4% | 3 ms |
| Data2txt | 900 | 64.3% | 64.6% | 99.5% | 78.3% | 78.3% | 0.592 | 98.4% | 11 ms |

`--max-good-blocked` tunes the lexical threshold for a budget instead: at ≤ 20% good answers blocked
(15.4% on test) it catches 51.6% of hallucinations with 64.3% precision. Use it when latency
matters more than accuracy.

### How we got here

| Step | ROC-AUC | F1 | Good answers blocked | What changed |
|---|---|---|---|---|
| 1. Lexical, weakest claim decides | 0.735 | 63.5% | 34.8% | First measurement |
| 2. + clock-time fix, `lowest_two` | 0.783 | 64.4% | 39.7% (tunable) | Error analysis + better aggregation |
| 3. General NLI (`hybrid`), 300-answer sample | 0.708 | 61.6% | 50.0% | Worse, and ~1,000× slower |
| 4. General NLI, evidence-first, 300-answer sample | 0.677 | 56.3% | 78.6% | 10× faster than step 3, still worse |
| **5. Task-trained token classifier** | **0.889** | **76.1%** | **12.3%** | Model trained for exactly this job |

- **Judging by the weakest claim alone over-penalised long answers.** Averaging the two weakest was
  chosen because it scored best on a separate sample of 1,200 *train* answers, not by looking at test.
- **Error analysis found a real bug**: correct opening hours were rejected because `11 am` didn't
  match `11:0` in the source. Clock times are now normalised.
- **A general-purpose NLI model was the wrong tool.** `nli-deberta-v3-small` is trained on short
  sentence pairs; against long evidence it gave near-zero entailment to almost every claim, true
  or not (tuned thresholds collapsed to 0.0009 and 0.012). Evidence-first pairing fixed the speed
  but not the accuracy.
- **What worked was a model trained on this task**, which reads every passage and the whole answer
  at once and labels each answer token.

### Reproduce

```bash
make bench            # lexical, full test set, about a minute
make install-nli      # once: CPU-only torch + transformers
make bench-nli        # NLI, 100 answers per task, evidence-first
make bench-hybrid     # numeric gate + NLI, 100 answers per task, evidence-first
make bench-token      # LettuceDetect token classifier, full test set (slow on CPU)

# Tune for a "good answers blocked" budget instead of best F1
python -m groundwire.evals --per-task 0 --dev-per-task 200 --max-good-blocked 0.2
```

Each run writes `results/<timestamp>-<scorer>/` containing `report.md` (the table above),
`summary.json`, and `predictions.jsonl` with the weakest claim of every answer for error analysis.
See `python -m groundwire.evals --help` for sampling, label policy and threshold options.

---

## Quality gates: CI fails if accuracy drops

The benchmark numbers above are enforced, not just reported. `config/quality_gates.yaml` defines
gates; each runs a **fixed, seeded sample with a fixed threshold**, so its numbers only move when
code, models or data change, never between runs.

| Suite | Runs | Gates | Cost |
|---|---|---|---|
| `fast` | every push and PR (`ci.yml`) | lexical fact-checker on 450 RAGTruth answers | ~5 s, no ML deps |
| `full` | `main`, weekly, PRs touching `guardrails/`, `cache/`, `evals/` (`quality-full.yml`) | LettuceDetect on 150 answers; semantic cache on 5,000 QQP pairs | minutes, CPU torch |

```bash
make gate                         # what CI runs on every push
make gate-full                    # model-based gates (needs make install-nli)
make gate-calibrate SUITE=full    # measure and print suggested bounds
```

The result appears as a table on the GitHub Actions run page:

```
## Quality gate `fast`: ❌ FAILED
| Metric           | Measured | Required  | Result   |
| roc_auc          | 0.7792   | >= 0.753  | pass     |
| false_block_rate | 0.4364   | <= 0.433  | **FAIL** |
```

That failure is real: it came from a simulated bug (word matching made case-sensitive) that the unit
tests mostly missed.

**Design rules:**

- **Fail-closed.** A gate that crashes, can't find its data, or doesn't report a checked metric counts
  as failed. `scorer_errors: {max: 0}` fails the build if the model errors on any example.
- **Bounds move only by review.** `--calibrate` *suggests* bounds (measured ± 0.02); a person commits
  them. Lowering a bound for a deliberate trade-off happens in the same PR, visible to reviewers.
- **Weekly runs catch drift** from upstream model or library updates even when nobody pushes.

**Limit, stated plainly:** a gate catches *large* regressions, not one-point changes. Disabling the
lexical number check moved the 450-answer sample's AUC from 0.774 to 0.796, but on the full 2,700
answers the difference vanished (0.7829 vs 0.7831). Small effects need the full benchmark, and
specific behaviours (wrong invoice numbers) are protected by unit tests.

---

## Operations

| Endpoint | Purpose |
|---|---|
| `GET /healthz` | Liveness: process is up. Never touches providers. |
| `GET /readyz` | Readiness: 503 if any route has every circuit open. |
| `GET /metrics` | Prometheus: requests, latency, attempts, failovers, circuit state, tokens, grounding verdicts, injection checks, PII redactions, cache hits/misses, tokens saved, coalesced requests. |
| `GET /v1/models` | Route aliases and their provider chains. |

Every response carries `X-Request-ID` (your own value is reused if you send
one) and `X-Groundwire-Provider`. All log lines for a request share its ID.

---

## Chaos demo

One command (starts a temporary gateway, runs the test, stops the gateway):

```bash
make demo
```

Or, against a gateway you started yourself:

```bash
make run          # terminal 1: keep this running
make chaos        # terminal 2
```

`make chaos` is only a client. If no gateway is running it stops immediately
with a hint instead of sending traffic.

Example run (in-process mocks; numbers measure gateway overhead, not model speed):

```
requests:       500 in 0.86s (583 req/s)
status codes:   {200: 500}
served by:      {'mock-flaky': 60, 'mock-stable': 440}
latency:        p50=64.8ms  p95=234.0ms  p99=324.3ms
circuit states: {'mock-flaky': 'open', 'mock-stable': 'closed'}
```

The primary fails 50% of the time, yet every request succeeds: retries absorb
some failures, the breaker opens and routes around the rest.

---

## Project layout

```
src/groundwire/
  api/            routes, ASGI middleware (request ID, body limit), error mapping
  services/       ChatService: safety, cache, deadline, grounding policy, metrics
  routing/        Router: fallback chains, failover, streaming handoff
  resilience/     RetryPolicy, CircuitBreaker
  guardrails/     text utils, scorers (lexical, NLI, hybrid), token detector, GroundingVerifier
  safety/         prompt-injection detectors, PII redaction/restoration, SafetyGuard
  cache/          cache keys, embeddings, store port + LRU/TTL adapter, single-flight, policy
  evals/          RAGTruth benchmark (loader, metrics, runner, report, CLI);
                  QQP loader + cachebench/ (semantic-cache benchmark)
  providers/      base port, OpenAI-compatible, Anthropic, mock, factory
  observability/  Prometheus metrics
  core/           config, errors, logging, request context
  domain/         OpenAI-compatible models
  container.py    composition root
  main.py         app factory and entrypoint
tests/            unit + integration (no network needed)
config/           gateway YAML (local and Docker variants)
scripts/          chaos demo
```

## Roadmap

- [x] Gateway core, streaming, failover, retries, circuit breakers, deadlines
- [x] Grounding verification (lexical, NLI, hybrid), flag/block policies
- [x] Reproducible RAGTruth benchmark with train-tuned thresholds
- [x] Metrics, structured logs, Docker, CI
- [x] Prompt-injection screening (heuristic + optional classifier) and PII redaction with restoration
- [x] NLI and hybrid benchmark results (negative result, documented)
- [x] Token-classifier (LettuceDetect) benchmark: 76.1% F1, reproduces the published result
- [x] Exact + semantic response cache with coalescing, PII-safe keys, fail-open
- [x] Semantic-cache benchmark harness (QQP, threshold tuned for a false-hit budget)
- [ ] Redis/pgvector cache adapter for multi-replica deployments
- [x] Benchmark quality gates in CI (fast on every push, model-based weekly/on main)
- [ ] OpenTelemetry tracing and Grafana dashboard
- [ ] Helm chart; shared (Redis-backed) breaker state across replicas

## License

MIT
