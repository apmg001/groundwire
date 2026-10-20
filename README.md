# Groundwire

**A self-hosted, OpenAI-compatible LLM gateway that keeps answers available and honest.**

Groundwire sits between your application and any LLM, local or cloud. Point an
OpenAI SDK at it by changing one URL, and every call gets:

- **Failover across providers**: local llama.cpp first, cloud only if needed.
- **Retries with jittered backoff** and **per-provider circuit breakers**.
- **End-to-end deadlines**, so a slow upstream can't hang your app.
- **Grounding verification**: checks the answer against the context you supplied
  and flags or blocks unsupported claims (invented amounts, dates, IDs).
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
        SVC --> GV[Grounding verifier<br/>lexical · NLI]
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
| `lexical` (default) | nothing | Invented numbers/amounts/dates/IDs; claims with little overlap with the context | Paraphrased contradictions ("did" vs "did not") |
| `nli` | `pip install 'groundwire[nli]'` | Semantic entailment and contradiction (DeBERTa cross-encoder) | Costs ~tens of ms per claim on CPU |

Scoring is CPU-bound, so it runs in a worker thread behind a concurrency limit and never blocks the event loop.

---

## Operations

| Endpoint | Purpose |
|---|---|
| `GET /healthz` | Liveness: process is up. Never touches providers. |
| `GET /readyz` | Readiness: 503 if any route has every circuit open. |
| `GET /metrics` | Prometheus: requests, latency, attempts, failovers, circuit state, tokens, grounding verdicts. |
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
  services/       ChatService: deadline, grounding policy, metrics
  routing/        Router: fallback chains, failover, streaming handoff
  resilience/     RetryPolicy, CircuitBreaker
  guardrails/     text utils, scorers (lexical, NLI), GroundingVerifier
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
- [x] Grounding verification (lexical + NLI), flag/block policies
- [x] Metrics, structured logs, Docker, CI
- [ ] Semantic cache (embeddings + Redis/pgvector)
- [ ] Prompt-injection detection and PII redaction
- [ ] Eval harness with a CI regression gate; RAGTruth/HaluEval benchmark
- [ ] OpenTelemetry tracing and Grafana dashboard
- [ ] Helm chart; shared (Redis-backed) breaker state across replicas

## License

MIT
