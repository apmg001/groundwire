"""Run a Groundwire load test end to end and write a report.

One command starts two fake upstreams and the gateway, optionally measures a
direct-to-upstream baseline with the same traffic, runs a Locust profile
against the gateway while sampling the gateway's memory and CPU, injects chaos
on schedule, then writes ``results/loadtest/<timestamp>-<profile>/``:

* ``report.md``: numbers, per-stage tables and pass/fail checks;
* ``summary.json``: the same, machine-readable;
* Locust's CSVs and HTML report, the gateway log, resource samples.

Profiles (see ``loadtest/locustfile.py`` for the stages):

* ``load``: a plateau of ``--users``; with ``--baseline`` the gateway's own
  overhead is measured against the same traffic sent directly upstream;
* ``stress``: steps up to 600 users, to find where latency or errors break;
* ``spike``: 20 → 300 → 20 users, to check survival and recovery;
* ``chaos``: a plateau while the primary upstream first fails every request,
  then becomes slower than the provider timeout;
* ``soak``: a long plateau, to check memory stays flat.

Exit code: 0 all checks passed, 1 a check failed, 2 setup error.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import statistics
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import psutil

from loadtest.profiles import FIXED, PROFILES

logger = logging.getLogger("loadtest")


ROOT = Path(__file__).resolve().parents[1]
LOADTEST_DIR = ROOT / "loadtest"
CONFIG = ROOT / "config" / "loadtest.yaml"
UPSTREAM_PORTS = (9101, 9102)


class SetupError(RuntimeError):
    """A process did not start or a port is busy."""


# --------------------------------------------------------------------------- #
# Processes
# --------------------------------------------------------------------------- #


def _wait_healthy(url: str, timeout_s: float = 60.0) -> None:
    """Poll ``url`` until it answers 200.

    Raises:
        SetupError: If it does not within ``timeout_s``.
    """
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            if httpx.get(url, timeout=1.0).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.25)
    raise SetupError(f"{url} did not become healthy within {timeout_s:.0f}s")


def _port_free(port: int) -> bool:
    """True if nothing listens on ``port`` on localhost."""
    return not any(c.laddr and c.laddr.port == port and c.status == psutil.CONN_LISTEN
                   for c in psutil.net_connections(kind="tcp"))


@contextmanager
def stack(out: Path, gateway_port: int, latency_ms: float) -> Iterator[psutil.Process]:
    """Start two fake upstreams and the gateway; stop them on exit.

    Args:
        out: Directory for process logs.
        gateway_port: Port for the gateway.
        latency_ms: Fake model time to first token.

    Yields:
        The gateway process (for resource sampling).

    Raises:
        SetupError: A port is busy or a process failed to start.
    """
    for port in (*UPSTREAM_PORTS, gateway_port):
        if not _port_free(port):
            raise SetupError(f"port {port} is busy; stop whatever is using it first")
    procs: list[subprocess.Popen[bytes]] = []
    logs = []
    try:
        for port in UPSTREAM_PORTS:
            log = (out / f"upstream-{port}.log").open("wb")
            logs.append(log)
            procs.append(subprocess.Popen(  # noqa: S603 - fixed argv, our own module
                [sys.executable, "-m", "loadtest.fake_upstream", "--port", str(port),
                 "--latency-ms", str(latency_ms)], cwd=ROOT, stdout=log, stderr=log))
        env = {**os.environ, "GROUNDWIRE_CONFIG_PATH": str(CONFIG),
               "GROUNDWIRE_LOG_LEVEL": "WARNING", "GROUNDWIRE_LOG_JSON": "false"}
        env.pop("WEB_CONCURRENCY", None)
        log = (out / "gateway.log").open("wb")
        logs.append(log)
        gateway = subprocess.Popen(  # noqa: S603 - fixed argv
            [sys.executable, "-m", "uvicorn", "groundwire.main:app_factory", "--factory",
             "--port", str(gateway_port), "--no-access-log", "--log-level", "warning"],
            cwd=ROOT, env=env, stdout=log, stderr=log)
        procs.append(gateway)
        for port in UPSTREAM_PORTS:
            _wait_healthy(f"http://127.0.0.1:{port}/healthz")
        _wait_healthy(f"http://127.0.0.1:{gateway_port}/healthz")
        yield psutil.Process(gateway.pid)
    finally:
        for proc in procs:
            proc.terminate()
        for proc in procs:
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
        for log in logs:
            log.close()


# --------------------------------------------------------------------------- #
# Sampling and chaos
# --------------------------------------------------------------------------- #


@dataclass
class Sample:
    """One resource sample of the gateway process."""

    t: float
    rss_mb: float
    cpu_percent: float


class ResourceSampler(threading.Thread):
    """Samples the gateway's memory and CPU once a second in the background."""

    def __init__(self, process: psutil.Process) -> None:
        """Create the sampler.

        Args:
            process: Gateway process.
        """
        super().__init__(daemon=True)
        self._process = process
        self._stop_event = threading.Event()
        self.samples: list[Sample] = []
        self._start = time.monotonic()

    def run(self) -> None:
        """Sample until stopped or the process exits."""
        self._process.cpu_percent(None)  # prime the CPU counter
        while not self._stop_event.wait(1.0):
            try:
                rss = self._process.memory_info().rss / 2**20
                cpu = self._process.cpu_percent(None)
            except psutil.Error:
                return
            self.samples.append(Sample(time.monotonic() - self._start, rss, cpu))

    def stop(self) -> None:
        """Stop sampling."""
        self._stop_event.set()
        self.join(timeout=5)


@dataclass
class ChaosEvent:
    """A behaviour change applied to an upstream at a time into the test."""

    at_fraction: float
    port: int
    change: dict[str, float]
    label: str
    applied_at_s: float | None = None


def chaos_plan() -> list[ChaosEvent]:
    """Primary fails everything, recovers, then turns slower than the provider timeout."""
    a = UPSTREAM_PORTS[0]
    return [ChaosEvent(0.20, a, {"error_rate": 1.0}, "primary returns 503 on every request"),
            ChaosEvent(0.45, a, {"error_rate": 0.0}, "primary recovers"),
            ChaosEvent(0.60, a, {"extra_latency_ms": 12_000.0},
                       "primary 12 s slower than normal (provider timeout is 10 s)"),
            ChaosEvent(0.85, a, {"extra_latency_ms": 0.0}, "primary back to normal")]


def run_chaos(events: list[ChaosEvent], duration_s: float, started: float) -> None:
    """Apply chaos events on schedule (runs in a thread during the test)."""
    for event in events:
        delay = started + event.at_fraction * duration_s - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        try:
            httpx.post(f"http://127.0.0.1:{event.port}/admin/chaos", json=event.change,
                       timeout=5).raise_for_status()
            event.applied_at_s = time.monotonic() - started
            logger.info("chaos at %.0fs: %s", event.applied_at_s, event.label)
        except httpx.HTTPError as exc:
            logger.error("chaos event failed: %s (%s)", event.label, exc)


# --------------------------------------------------------------------------- #
# Locust
# --------------------------------------------------------------------------- #


def run_locust(user_class: str, host: str, profile: str, prefix: Path, *,
               users: int, duration_s: float) -> None:
    """Run Locust headless and write CSV + HTML under ``prefix``.

    Raises:
        SetupError: If Locust cannot start.
    """
    env = {**os.environ, "LOAD_PROFILE": profile, "LOAD_DURATION": str(duration_s),
           "LOAD_USERS": str(users),
           "PYTHONPATH": os.pathsep.join(filter(None, [str(ROOT), os.environ.get("PYTHONPATH")]))}
    cmd = [sys.executable, "-m", "locust", "-f", str(LOADTEST_DIR / "locustfile.py"),
           user_class, "--headless", "--host", host, "--csv", str(prefix),
           "--csv-full-history", "--html", f"{prefix}.html", "--only-summary",
           "--loglevel", "WARNING"]
    with (prefix.parent / f"{prefix.name}.locust.log").open("wb") as log:
        code = subprocess.call(cmd, cwd=ROOT, env=env,  # noqa: S603 - fixed argv
                               stdout=log, stderr=log)
    if not (prefix.parent / f"{prefix.name}_stats.csv").exists():
        raise SetupError(f"locust produced no results (exit {code}); see {prefix}.locust.log")


def read_stats(prefix: Path) -> dict[str, dict[str, float]]:
    """Per-request-type totals from ``<prefix>_stats.csv``."""
    rows: dict[str, dict[str, float]] = {}
    with (prefix.parent / f"{prefix.name}_stats.csv").open() as handle:
        for row in csv.DictReader(handle):
            count = float(row["Request Count"] or 0)
            rows[row["Name"]] = {
                "requests": count, "failures": float(row["Failure Count"] or 0),
                "rps": float(row["Requests/s"] or 0),
                "p50": float(row["50%"] or 0), "p95": float(row["95%"] or 0),
                "p99": float(row["99%"] or 0), "max": float(row["Max Response Time"] or 0),
                "mean": float(row["Average Response Time"] or 0)}
    return rows


def _num(value: str) -> float:
    """Parse a Locust CSV number (``N/A`` before the first response counts as 0)."""
    try:
        return float(value)
    except ValueError:
        return 0.0


def read_history(prefix: Path) -> list[dict[str, Any]]:
    """Per-second, per-request-type history from ``<prefix>_stats_history.csv``.

    Returns:
        Rows with ``t`` (seconds since start), ``name``, ``users``, ``rps``,
        ``fails`` and the current-window percentiles ``p50``/``p95``/``p99``.
    """
    path = prefix.parent / f"{prefix.name}_stats_history.csv"
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return []
    t0 = float(rows[0]["Timestamp"])
    return [{"t": float(r["Timestamp"]) - t0, "name": r["Name"], "users": _num(r["User Count"]),
             "rps": _num(r["Requests/s"]), "fails": _num(r["Failures/s"]),
             "p50": _num(r["50%"]), "p95": _num(r["95%"]), "p99": _num(r["99%"])}
            for r in rows]


# --------------------------------------------------------------------------- #
# Analysis
# --------------------------------------------------------------------------- #


@dataclass
class Check:
    """One pass/fail criterion."""

    name: str
    measured: float
    bound: str
    passed: bool


@dataclass
class Report:
    """Everything one run measured."""

    profile: str
    machine: str
    users: int
    duration_s: float
    latency_ms: float
    totals: dict[str, dict[str, float]] = field(default_factory=dict)
    baseline: dict[str, dict[str, float]] = field(default_factory=dict)
    stages: list[dict[str, float]] = field(default_factory=list)
    resources: dict[str, float] = field(default_factory=dict)
    chaos: list[dict[str, Any]] = field(default_factory=list)
    checks: list[Check] = field(default_factory=list)
    provider_counts: dict[str, float] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        """True if every check passed."""
        return all(c.passed for c in self.checks)


SHED = "shed (503)"  # must match loadtest.locustfile.SHED_NAME
LATENCY_REQUEST = "chat"  # the main request type (half the traffic) for per-stage latency


def failure_rate(stats: dict[str, float]) -> float:
    """Failures / requests (0 when there were none)."""
    return stats["failures"] / stats["requests"] if stats.get("requests") else 0.0


def served(totals: dict[str, dict[str, float]]) -> dict[str, float]:
    """Requests answered by the gateway (every type except refusals), summed.

    Returns:
        ``requests``, ``failures`` and ``shed`` counts plus ``shed_rate``
        (refusals / all requests).
    """
    rows = [v for k, v in totals.items() if k not in ("Aggregated", SHED)]
    requests = sum(r["requests"] for r in rows)
    failures = sum(r["failures"] for r in rows)
    shed = totals.get(SHED, {}).get("requests", 0.0)
    return {"requests": requests, "failures": failures, "shed": shed,
            "shed_rate": shed / (requests + shed) if requests + shed else 0.0}


def by_stage(history: list[dict[str, Any]],
             stages: list[tuple[float, int, float]]) -> list[dict[str, float]]:
    """Summarise each load stage, skipping its first quarter (ramp-up).

    Args:
        history: Output of :func:`read_history`.
        stages: ``(end_s, users, spawn_rate)`` from the locustfile.

    Returns:
        Per stage: users, served and shed requests/s, failures/s, and the
        ``chat`` request's median p50 and worst p95/p99 over the stage.
    """
    summary: list[dict[str, float]] = []
    start = 0.0
    for end, users, _ in stages:
        lo = start + (end - start) / 4
        window = [r for r in history if lo <= r["t"] < end]
        seconds = sorted({r["t"] for r in window})
        start = end
        if not seconds:
            continue

        def per_second(pick: Callable[[dict[str, Any]], bool], key: str,
                       rows: list[dict[str, Any]] = window, n: int = len(seconds)) -> float:
            return float(sum(r[key] for r in rows if pick(r))) / n

        chat = [r for r in window if r["name"] == LATENCY_REQUEST and r["p50"]]
        summary.append({
            "users": users,
            "served_rps": round(per_second(lambda r: r["name"] not in ("Aggregated", SHED),
                                           "rps"), 1),
            "shed_rps": round(per_second(lambda r: r["name"] == SHED, "rps"), 1),
            "fail_per_s": round(per_second(lambda r: r["name"] == "Aggregated", "fails"), 2),
            "p50_ms": statistics.median(r["p50"] for r in chat) if chat else 0.0,
            "p95_ms": max((r["p95"] for r in chat), default=0.0),
            "p99_ms": max((r["p99"] for r in chat), default=0.0)})
    return summary


def provider_counts(port: int) -> dict[str, float]:
    """Requests served per provider, from the gateway's Prometheus metrics."""
    counts: dict[str, float] = {}
    try:
        text = httpx.get(f"http://127.0.0.1:{port}/metrics", timeout=5).text
    except httpx.HTTPError:
        return counts
    for line in text.splitlines():
        if line.startswith("groundwire_requests_total{") and 'status="success"' in line:
            provider = line.split('provider="', 1)[1].split('"', 1)[0]
            counts[provider] = counts.get(provider, 0.0) + float(line.rsplit(" ", 1)[1])
    return counts


def evaluate(report: Report, history: list[dict[str, Any]]) -> None:
    """Fill in the pass/fail checks for the profile.

    Bounds were written down before the runs they judge, not fitted to the
    results. "Errors" are real failures; load-shedding refusals are counted
    separately (refusing excess load is the intended behaviour).
    """
    done = served(report.totals)
    errors = done["failures"] / done["requests"] if done["requests"] else 0.0
    checks = report.checks
    if report.profile in ("load", "soak"):
        checks.append(Check("error rate", errors, "< 0.1%", errors < 0.001))
        checks.append(Check("requests shed at normal load", done["shed_rate"], "< 0.1%",
                            done["shed_rate"] < 0.001))
    if report.profile == "load" and report.baseline:
        # Means, not medians: Locust rounds percentiles above 100 ms to 10 ms.
        ours = report.totals.get(LATENCY_REQUEST, {}).get("mean", 0)
        direct = report.baseline.get(LATENCY_REQUEST, {}).get("mean", 0)
        checks.append(Check("gateway overhead, mean (ms)", ours - direct, "< 25",
                            ours - direct < 25))
    if report.profile in ("stress", "spike"):
        checks.append(Check("error rate (refusals excluded)", errors, "< 1%", errors < 0.01))
        worst = max((s["p99_ms"] for s in report.stages), default=0.0)
        checks.append(Check("worst stage p99 of served chat (ms)", worst, "< 2500",
                            worst < 2500))
    if report.profile == "spike":
        after = sorted({r["t"] for r in history if r["t"] >= 75})
        calm = [r["p95"] for r in history
                if r["name"] == LATENCY_REQUEST and 10 <= r["t"] < 30 and r["p95"]]
        normal = statistics.median(calm) if calm else 0.0
        recovered = next((t - 75 for t in after
                          if any(r["t"] == t and r["name"] == LATENCY_REQUEST and r["p95"]
                                 and r["p95"] <= 1.2 * normal + 20 for r in history)), None)
        value = recovered if recovered is not None else float("inf")
        checks.append(Check("recovery after spike (s)", value, "<= 30", value <= 30))
    if report.profile == "chaos":
        checks.append(Check("success rate during chaos", 1 - errors, ">= 99%", errors <= 0.01))
        slowest = max((v["max"] for k, v in report.totals.items() if k != SHED), default=0.0)
        checks.append(Check("slowest request (ms)", slowest, "< 30000 (deadline)",
                            slowest < 30_000))
    if report.profile == "soak" and report.resources.get("rss_growth_pct") is not None:
        growth = report.resources["rss_growth_pct"]
        checks.append(Check("memory growth, second half of run (%)", growth, "< 5",
                            growth < 5))


def summarise_resources(samples: list[Sample]) -> dict[str, float]:
    """Peak and steady-state memory and CPU of the gateway.

    Leak check: memory growth over the *second half* of the run. The first
    half is excluded because bounded structures (the response cache, up to
    ``max_entries``) legitimately fill up first; in the 10-minute soak the
    cache filled by ~300 s and memory was flat after that. A leak keeps
    growing in the second half; a full cache does not.

    Args:
        samples: One-second samples from :class:`ResourceSampler`.

    Returns:
        Memory (MB) at start, middle and end, second-half growth (%), peak
        memory, and mean/peak CPU (% of one core). Empty without samples.
    """
    if not samples:
        return {}
    window = max(1, len(samples) // 20)  # average 5% of the run at each point

    def rss_at(start: int) -> float:
        return statistics.mean(s.rss_mb for s in samples[start:start + window])

    start_rss, mid_rss = rss_at(0), rss_at(len(samples) // 2)
    end_rss = rss_at(len(samples) - window)
    warm = samples[len(samples) // 5:] or samples
    return {"rss_start_mb": round(start_rss, 1), "rss_mid_mb": round(mid_rss, 1),
            "rss_end_mb": round(end_rss, 1),
            "rss_growth_pct": round(100 * (end_rss - mid_rss) / mid_rss, 2),
            "rss_peak_mb": round(max(s.rss_mb for s in samples), 1),
            "cpu_mean_pct": round(statistics.mean(s.cpu_percent for s in warm), 1),
            "cpu_peak_pct": round(max(s.cpu_percent for s in samples), 1)}


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #


def render(report: Report) -> str:
    """Markdown report."""
    lines = [f"# Load test `{report.profile}`: {'PASSED' if report.passed else 'FAILED'}", "",
             f"- Machine: {report.machine}",
             f"- Fake model latency: {report.latency_ms:.0f} ms to first token",
             ("- Stages (end s, users): "
              + ", ".join(f"{end:.0f}s→{users}" for end, users, _ in FIXED[report.profile])
              if report.profile in FIXED else
              f"- Users: {report.users}, duration: {report.duration_s:.0f} s"), ""]
    if report.checks:
        lines += ["| Check | Measured | Required | Result |", "|---|---|---|---|"]
        lines += [f"| {c.name} | {c.measured:.4g} | {c.bound} | "
                  f"{'pass' if c.passed else '**FAIL**'} |" for c in report.checks]
        lines.append("")
    lines += ["## Totals (ms)", "",
              "| Request | Requests | Failures | req/s | p50 | p95 | p99 | max |",
              "|---|---|---|---|---|---|---|---|"]
    for name, s in report.totals.items():
        lines.append(f"| {name} | {s['requests']:.0f} | {s['failures']:.0f} | {s['rps']:.1f} | "
                     f"{s['p50']:.0f} | {s['p95']:.0f} | {s['p99']:.0f} | {s['max']:.0f} |")
    if report.baseline:
        lines += ["", "## Baseline: same traffic straight to the fake model (ms)", "",
                  "| Request | mean | p50 | p95 | p99 | gateway overhead (mean) |",
                  "|---|---|---|---|---|---|"]
        lines += [f"| {n} | {s['mean']:.1f} | {s['p50']:.0f} | {s['p95']:.0f} | "
                  f"{s['p99']:.0f} | {report.totals.get(n, {}).get('mean', 0) - s['mean']:+.1f} |"
                  for n, s in report.baseline.items() if n not in ("Aggregated", "chat (cached)")]
    if report.stages:
        lines += ["", "## By stage", "",
                  "First quarter of each stage skipped as ramp-up. Latency is the `chat` "
                  "request (served requests only).", "",
                  "| Users | served req/s | shed req/s | failures/s | p50 ms | p95 ms | p99 ms |",
                  "|---|---|---|---|---|---|---|"]
        lines += [f"| {s['users']:.0f} | {s['served_rps']} | {s['shed_rps']} | "
                  f"{s['fail_per_s']} | {s['p50_ms']:.0f} | {s['p95_ms']:.0f} | "
                  f"{s['p99_ms']:.0f} |" for s in report.stages]
    if report.chaos:
        lines += ["", "## Chaos events", ""]
        lines += [f"- {e['applied_at_s']:.0f} s: {e['label']}" for e in report.chaos
                  if e.get("applied_at_s") is not None]
    if report.provider_counts:
        lines += ["", "Requests served per provider: "
                  + ", ".join(f"{k} {v:.0f}" for k, v in report.provider_counts.items())]
    if report.resources:
        r = report.resources
        lines += ["", f"Gateway process: memory {r['rss_start_mb']} → {r['rss_mid_mb']} → "
                  f"{r['rss_end_mb']} MB (start, middle, end; {r['rss_growth_pct']:+}% over "
                  f"the second half), peak {r['rss_peak_mb']} MB; "
                  f"CPU mean {r['cpu_mean_pct']}%, peak {r['cpu_peak_pct']}% "
                  "(100% = one core)."]
    return "\n".join(lines) + "\n"


def machine() -> str:
    """Short description of the machine the numbers come from."""
    cpu = psutil.cpu_count(logical=True)
    mem = psutil.virtual_memory().total / 2**30
    return f"{cpu} logical CPUs, {mem:.0f} GiB RAM, Python {sys.version.split()[0]}"


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: Arguments (defaults to ``sys.argv[1:]``).

    Returns:
        Exit code.
    """
    parser = argparse.ArgumentParser(description="Run a Groundwire load test.")
    parser.add_argument("profile", choices=PROFILES)
    parser.add_argument("--users", type=int, default=100)
    parser.add_argument("--duration", type=float, default=60.0, help="seconds (plateau profiles)")
    parser.add_argument("--latency-ms", type=float, default=300.0, help="fake model latency")
    parser.add_argument("--baseline", action="store_true",
                        help="also run the same traffic directly upstream (load profile)")
    parser.add_argument("--port", type=int, default=8020, help="gateway port")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "loadtest")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # no line per health poll

    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    out = args.out / f"{stamp}-{args.profile}"
    out.mkdir(parents=True, exist_ok=True)
    report = Report(profile=args.profile, machine=machine(), users=args.users,
                    duration_s=args.duration, latency_ms=args.latency_ms)
    try:
        with stack(out, args.port, args.latency_ms) as gateway:
            if args.baseline:
                logger.info("baseline: same traffic straight to the fake model")
                run_locust("DirectUser", f"http://127.0.0.1:{UPSTREAM_PORTS[0]}", args.profile,
                           out / "baseline", users=args.users, duration_s=args.duration)
                report.baseline = read_stats(out / "baseline")
            sampler = ResourceSampler(gateway)
            sampler.start()
            started = time.monotonic()
            events = chaos_plan() if args.profile == "chaos" else []
            chaos_thread = threading.Thread(target=run_chaos, daemon=True,
                                            args=(events, args.duration, started))
            if events:
                chaos_thread.start()
            logger.info("running %s against the gateway", args.profile)
            run_locust("GatewayUser", f"http://127.0.0.1:{args.port}", args.profile,
                       out / "gateway", users=args.users, duration_s=args.duration)
            sampler.stop()
            report.provider_counts = provider_counts(args.port)
    except (SetupError, OSError, psutil.Error) as exc:
        logger.error("load test could not run: %s", exc)
        return 2

    history = read_history(out / "gateway")
    report.totals = read_stats(out / "gateway")
    report.stages = (by_stage(history, FIXED[args.profile])
                     if args.profile in ("stress", "spike") else [])
    report.resources = summarise_resources(sampler.samples)
    report.chaos = [asdict(e) for e in events]
    evaluate(report, history)
    (out / "resources.json").write_text(json.dumps([asdict(s) for s in sampler.samples]))
    (out / "summary.json").write_text(json.dumps(asdict(report) | {"passed": report.passed},
                                                 indent=2, default=str))
    (out / "report.md").write_text(render(report))
    logger.info("report: %s", out / "report.md")
    sys.stdout.write(render(report))
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
