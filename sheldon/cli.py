"""Command line. Five verbs.

    sheldon ls                       what benchmarks and methods exist
    sheldon check                    what data is missing and how to get it
    sheldon smoke                    full pipeline, offline, ~1s
    sheldon run -b arc-agi-1 -m sheldon,direct -n 20
    sheldon report                   the table
    sheldon failures <run_id>        the worst tasks, with trace paths
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from sheldon import report as report_mod
from sheldon.benchmarks import benchmarks
from sheldon.benchmarks import get as get_benchmark
from sheldon.core.config import Config
from sheldon.core.io import load_dotenv
from sheldon.core.llm import credentials_status
from sheldon.engines import engines as engine_registry
from sheldon.engines import get as get_engine
from sheldon.methods import get as get_method
from sheldon.methods import methods
from sheldon.runner import run_cells, run_dir, run_experiment


def _split_list(value: str | None) -> list[str]:
    return [v.strip() for v in (value or "").split(",") if v.strip()]


def _parse_params(pairs: list[str]) -> dict[str, Any]:
    """-p key=value, with JSON values so `-p critic=false` is a bool.

    A dotted key nests: `-p benchmark_opts.report_dir=runs/x` sets
    `params["benchmark_opts"]["report_dir"]`. Without this it set a key called
    `"benchmark_opts.report_dir"` instead, which nothing reads -- so the flag was
    accepted, ignored, and two arms of an A/B silently shared one output path and
    graded each other's files. A flag that cannot work should not parse.
    """
    out: dict[str, Any] = {}
    for pair in pairs:
        if "=" not in pair:
            raise SystemExit(f"bad -p {pair!r}, expected key=value")
        k, v = pair.split("=", 1)
        try:
            value = json.loads(v)
        except json.JSONDecodeError:
            value = v
        target = out
        *parents, leaf = [part for part in k.strip().split(".") if part]
        for part in parents:
            existing = target.get(part)
            if not isinstance(existing, dict):
                existing = {}
                target[part] = existing
            target = existing
        target[leaf] = value
    return out


# -- commands -------------------------------------------------------------


_DOTENV_LOADED: list[str] = []


def engine_names() -> list[str]:
    return engine_registry.names()


def cmd_ls(args: argparse.Namespace) -> int:
    print("benchmarks:")
    for name in benchmarks.names():
        try:
            print("  " + get_benchmark(name).describe())
        except Exception as e:
            print(f"  {name:<18} [error: {e}]")
    print("\nmethods:")
    for name in methods.names():
        print("  " + get_method(name).describe())
    print("\nengines:")
    for name in engine_names():
        print("  " + get_engine(name).describe())
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    names = _split_list(args.benchmark) or benchmarks.names()
    bad = 0
    for name in names:
        problems = get_benchmark(name).check()
        if problems:
            bad += 1
            print(f"✗ {name}")
            for p in problems:
                print(f"    {p}")
        else:
            print(f"✓ {name}")
    print(f"\ncredentials: {credentials_status()}")
    shadowed = [n[:-1] for n in _DOTENV_LOADED if n.endswith("*")]
    if shadowed:
        print(f"  (.env overrode a different value already in the environment: {shadowed})")
    return 1 if bad else 0


def cmd_smoke(args: argparse.Namespace) -> int:
    from sheldon import smoke

    return smoke.main(verbose=args.echo)


def cmd_run(args: argparse.Namespace) -> int:
    overrides = {
        "split": args.split,
        "limit": args.limit,
        "tasks": args.tasks,
        "seed": args.seed,
        "model": args.model,
        "effort": args.effort,
        "engine": args.engine,
        "provider": args.provider,
        "max_rounds": args.rounds,
        "max_usd": args.max_usd,
        "max_calls": args.max_calls,
        "max_seconds": args.max_seconds,
        "concurrency": args.jobs,
        "call_concurrency": args.call_jobs,
        "tag": args.tag,
        # empty dict must not override a config file's params
        "params": _parse_params(args.param) or None,
    }
    if args.no_cache:
        overrides["cache"] = False
    if args.no_resume:
        overrides["resume"] = False

    base = Config.load(args.config, **{k: v for k, v in overrides.items() if v is not None})
    bms = _split_list(args.benchmark) or [base.benchmark]
    mts = _split_list(args.method) or [base.method]

    cells = [
        Config(**{**base.to_dict(), "benchmark": b, "method": m}) for b in bms for m in mts
    ]

    if args.dry_run:
        for c in cells:
            print(f"{c.run_id}\n  -> {run_dir(c)}")
            print("  " + json.dumps(c.to_dict(), default=str))
        return 0

    failed = 0

    def report_cell(cfg, summary, error):
        head = f"=== {cfg.benchmark} / {cfg.method} ({cfg.split}, limit={cfg.limit}) ==="
        if error is not None:
            print(f"\n{head}\n  FAILED: {type(error).__name__}: {error}",
                  file=sys.stderr, flush=True)
            return
        print(
            f"\n{head}\n"
            f"  score={summary['score']:.3f} ±{summary['stderr']:.3f}  n={summary['n']}  "
            f"errors={summary['errors']}  ${summary['usage']['cost_usd']:.2f}  "
            f"cache_hits={summary.get('cache', {}).get('hits', 0)}  "
            f"{report_mod.fmt_duration(summary.get('wall_s', 0))}\n"
            f"  {run_dir(cfg)}",
            flush=True,
        )

    if args.sequential or len(cells) == 1:
        for c in cells:
            print(f"\n=== {c.benchmark} / {c.method} ({c.split}, limit={c.limit}) ===",
                  flush=True)
            try:
                summary = asyncio.run(run_experiment(c, echo=args.echo))
            except Exception as e:
                failed += 1
                print(f"  FAILED: {type(e).__name__}: {e}", file=sys.stderr)
                continue
            print(
                f"  score={summary['score']:.3f} ±{summary['stderr']:.3f}  n={summary['n']}  "
                f"errors={summary['errors']}  ${summary['usage']['cost_usd']:.2f}  "
                f"cache_hits={summary.get('cache', {}).get('hits', 0)}"
            )
            print(f"  {run_dir(c)}")
    else:
        # Cells share one concurrency budget, so `-j` still describes the load
        # on the endpoint rather than the load per cell.
        done = asyncio.run(run_cells(cells, echo=args.echo, on_cell=report_cell))
        failed = len(cells) - len(done)

    if len(cells) > 1 and not failed:
        print("\n" + report_mod.markdown(report_mod.collect(tag=cells[0].tag)))
    return 1 if failed else 0


def cmd_report(args: argparse.Namespace) -> int:
    if args.refresh:
        for run_id in refresh_summaries():
            print(f"refreshed {run_id}", file=sys.stderr)
    rows = report_mod.collect(benchmark=args.benchmark, method=args.method, tag=args.tag)
    print(report_mod.to_csv(rows) if args.csv else report_mod.markdown(rows))
    return 0


def refresh_summaries() -> list[str]:
    """Rebuild every summary.json from the raw per-task records.

    `predictions.jsonl` and `scores.jsonl` are the primary data; summary.json is
    a derived view. When the view gains a field — or a scorer changes — this
    recomputes it from what is already on disk, with no model calls and no cost.
    """
    from sheldon.core.config import Config  # noqa: PLC0415
    from sheldon.core.io import runs_root, write_json  # noqa: PLC0415
    from sheldon.runner import summarize  # noqa: PLC0415

    done: list[str] = []
    for cfg_path in sorted(runs_root().glob("*/config.json")):
        out = cfg_path.parent
        try:
            cfg = Config(**json.loads(cfg_path.read_text(encoding="utf-8"))["config"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as e:
            print(f"skipped {out.name}: {e}", file=sys.stderr)
            continue
        write_json(out / "summary.json", summarize(cfg, out))
        done.append(out.name)
    return done


def cmd_show(args: argparse.Namespace) -> int:
    print(report_mod.render_trace(args.run_id, args.task_id, width=args.width))
    return 0


def cmd_ui(args: argparse.Namespace) -> int:
    from sheldon.ui import console as ui_mod  # noqa: PLC0415

    # Served, traces are fetched per task, so the page carries the index alone.
    # Written to a file, they have to be embedded — a `file://` page cannot
    # fetch, and neither can a published artifact.
    data = ui_mod.collect(
        benchmark=args.benchmark, tag=args.tag, max_mb=args.max_mb,
        embed=args.serve is None,
    )
    if not data["runs"]:
        print("no completed runs found — try `sheldon run` first", file=sys.stderr)
        return 1
    out = ui_mod.write(Path(args.out), data)
    tasks = sum(len(r["tasks"]) for r in data["runs"])
    size = out.stat().st_size / 1024 / 1024
    print(f"{out}  ({len(data['runs'])} runs, {tasks} tasks, {size:.1f} MB)")
    if data["omitted"]:
        print(f"  {len(data['omitted'])} run(s) had traces trimmed for size; "
              f"raise --max-mb to include them", file=sys.stderr)
    if args.serve is not None:
        _serve(out, args.serve)
        return 0

    if args.open:
        import webbrowser  # noqa: PLC0415

        webbrowser.open(out.resolve().as_uri())
    return 0


def _serve(path: Path, port: int, routes: dict[str, Any] | None = None) -> None:
    """Serve the page on localhost until interrupted.

    For the case this is usually run in: the repo is on a remote box and you
    are reading it from a laptop. Bound to 127.0.0.1 deliberately — an SSH
    tunnel or a VS Code forwarded port reaches it, and nothing else on the
    network does.
    """
    import functools  # noqa: PLC0415
    from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer  # noqa: PLC0415

    # Prefix -> resolver. Each console serves its bodies on its own route, so
    # one server implementation covers both rather than being copied.
    resolvers = dict(routes or {})

    class Handler(SimpleHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            for prefix, resolve in resolvers.items():
                if self.path.startswith(prefix):
                    return self._json(prefix, resolve)
            if self.path.startswith("/task/"):
                return self._task()
            if self.path in ("/", "/index.html"):
                self.path = "/" + path.name
            super().do_GET()

        def _json(self, prefix: str, resolve: Any) -> None:
            from urllib.parse import unquote  # noqa: PLC0415

            parts = [unquote(x) for x in self.path[len(prefix):].split("/") if x]
            try:
                payload = resolve(*parts) if len(parts) == 2 else None
            except Exception:
                payload = None
            if payload is None:
                self.send_error(404, "not found")
                return
            body = json.dumps(payload, default=str).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _task(self) -> None:
            """One task's trace, read from disk when it is opened.

            The whole index is about 0.2 MB against roughly 149 MB of traces, so
            embedding everything means embedding almost nothing — eight tasks of
            one run, in practice. Served, the page holds every run and reads a
            trace only when someone asks for it.
            """
            from urllib.parse import unquote  # noqa: PLC0415

            from sheldon.ui import console as ui_mod  # noqa: PLC0415

            parts = [unquote(x) for x in self.path[len("/task/"):].split("/")]
            payload = (
                ui_mod.task_payload(parts[0], parts[1]) if len(parts) == 2 else None
            )
            if payload is None:
                self.send_error(404, "no such task")
                return
            body = json.dumps(payload, default=str).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args: object) -> None:
            pass  # one line per asset is noise, not information

        def handle_one_request(self) -> None:
            # A browser that navigates away mid-transfer resets the connection.
            # That is routine on a 2 MB page and is not worth a traceback.
            try:
                super().handle_one_request()
            except (BrokenPipeError, ConnectionResetError):
                self.close_connection = True

    handler = functools.partial(Handler, directory=str(path.parent.resolve()))
    with ThreadingHTTPServer(("127.0.0.1", port), handler) as httpd:
        print(f"serving {path.name} at http://localhost:{port}  (ctrl-c to stop)", flush=True)
        print("  over ssh:  ssh -N -L "
              f"{port}:localhost:{port} <this-host>   then open the same URL locally",
              flush=True)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print()


def _pick_run(args: argparse.Namespace) -> Path | None:
    """The newest run matching the filters, or the one named by --run."""
    from sheldon.core.io import runs_root  # noqa: PLC0415

    root = runs_root()
    if getattr(args, "run", None):
        d = root / args.run
        if not (d / "summary.json").exists():
            print(f"no such run: {d}", file=sys.stderr)
            return None
        return d
    best = None
    for sp in sorted(root.glob("*/summary.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            sm = json.loads(sp.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if not sm.get("n"):
            continue
        if getattr(args, "tag", None) and sm.get("tag") != args.tag:
            continue
        if getattr(args, "benchmark", None) and sm.get("benchmark") != args.benchmark:
            continue
        if getattr(args, "method", None) and sm.get("method") != args.method:
            continue
        best = sp.parent
        break
    if best is None:
        print("no matching run found", file=sys.stderr)
    return best


def cmd_log(args: argparse.Namespace) -> int:
    """Every model call of every run, with both sides of the exchange."""
    from sheldon.ui import log as log_ui  # noqa: PLC0415

    data = log_ui.collect(benchmark=args.benchmark, tag=args.tag)
    if not data["runs"]:
        print("no completed runs found - try `sheldon run` first", file=sys.stderr)
        return 1
    out = log_ui.write(Path(args.out), data)
    calls = sum(r["calls"] for r in data["runs"])
    print(f"{out}  ({len(data['runs'])} runs, {calls} calls indexed)")
    if args.serve is None:
        # Bodies come from the server, so a file:// page can only ever show the
        # index. Say so here rather than letting every task fail to open.
        print("  call bodies are served on demand: use --serve to read them",
              file=sys.stderr)
        if args.open:
            import webbrowser  # noqa: PLC0415

            webbrowser.open(out.resolve().as_uri())
        return 0
    _serve(out, args.serve, routes={"/calls/": log_ui.calls_payload})
    return 0


def cmd_failures(args: argparse.Namespace) -> int:
    for row in report_mod.failures(args.run_id, limit=args.limit):
        print(json.dumps(row, indent=2, default=str))
    return 0


# -- parser ---------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="sheldon", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("ls", help="list benchmarks and methods")
    s.set_defaults(fn=cmd_ls)

    s = sub.add_parser("check", help="report missing data")
    s.add_argument("-b", "--benchmark", help="comma-separated; default all")
    s.set_defaults(fn=cmd_check)

    s = sub.add_parser("smoke", help="offline end-to-end test, no API key")
    s.add_argument("--echo", action="store_true", help="print every trace event")
    s.set_defaults(fn=cmd_smoke)

    s = sub.add_parser("run", help="run one or more (benchmark, method) cells")
    s.add_argument("-c", "--config", help="YAML config; CLI flags override it")
    s.add_argument("-b", "--benchmark", help="comma-separated")
    s.add_argument("-m", "--method", help="comma-separated")
    s.add_argument("-s", "--split", default=None)
    s.add_argument("-n", "--limit", type=int, default=None, help="tasks per cell")
    s.add_argument("--task", default=None, dest="tasks", metavar="ID[,ID...]",
                   help="run only these task ids, ignoring --limit")
    s.add_argument("-j", "--jobs", type=int, default=None,
                   help="concurrent tasks, summed across cells")
    s.add_argument("--call-jobs", type=int, default=None,
                   help="concurrent API calls (default 4x --jobs)")
    s.add_argument("--sequential", action="store_true",
                   help="run cells one at a time instead of together")
    s.add_argument("--seed", type=int, default=None)
    s.add_argument("--model", default=None)
    s.add_argument("--effort", default=None, choices=["low", "medium", "high", "xhigh", "max"])
    s.add_argument("-e", "--engine", default=None, help="toolloop | simple | react | codex")
    s.add_argument("--provider", default=None, choices=["auto", "anthropic", "openai"])
    s.add_argument("--rounds", type=int, default=None, help="max loop rounds per task")
    s.add_argument("--max-usd", type=float, default=None, dest="max_usd", help="budget per task")
    s.add_argument("--max-calls", type=int, default=None, dest="max_calls",
                   help="model calls per task — the cap that binds a loop long "
                        "before spend does")
    s.add_argument("--max-seconds", type=float, default=None, dest="max_seconds",
                   help="wall-clock seconds per task; stages stop starting once it is spent")
    s.add_argument("-p", "--param", action="append", default=[], metavar="K=V",
                   help="method param, JSON value (repeatable)")
    s.add_argument("--tag", default=None, help="label this run in the report")
    s.add_argument("--no-cache", action="store_true")
    s.add_argument("--no-resume", action="store_true")
    s.add_argument("--echo", action="store_true")
    s.add_argument("--dry-run", action="store_true", help="print resolved configs and exit")
    s.set_defaults(fn=cmd_run)

    s = sub.add_parser("report", help="aggregate finished runs")
    s.add_argument("-b", "--benchmark")
    s.add_argument("-m", "--method")
    s.add_argument("--tag")
    s.add_argument("--csv", action="store_true")
    s.add_argument("--refresh", action="store_true",
                   help="rebuild every summary.json from predictions/scores first")
    s.set_defaults(fn=cmd_report)

    s = sub.add_parser("show", help="one task's trace, rendered to read")
    s.add_argument("run_id")
    s.add_argument("task_id", nargs="?", help="default: every task in the run")
    s.add_argument("-w", "--width", type=int, default=2000,
                   help="truncate each block at this many chars")
    s.set_defaults(fn=cmd_show)

    s = sub.add_parser("ui", help="browsable HTML view of every run and trace")
    s.add_argument("-o", "--out", default="runs/ui.html")
    s.add_argument("-b", "--benchmark")
    s.add_argument("--tag")
    s.add_argument("--max-mb", type=float, default=12.0,
                   help="cap on embedded trace data (default 12)")
    s.add_argument("--open", action="store_true", help="open it in a browser")
    s.add_argument("--serve", type=int, nargs="?", const=8000, metavar="PORT",
                   help="serve it on localhost:PORT (default 8000) instead of opening it")
    s.set_defaults(fn=cmd_ui)

    s = sub.add_parser("log", help="browsable log of every model call, prompt and reply")
    s.add_argument("-o", "--out", default="runs/log.html")
    s.add_argument("-b", "--benchmark")
    s.add_argument("--tag")
    s.add_argument("--open", action="store_true", help="open it in a browser")
    s.add_argument("--serve", type=int, nargs="?", const=8001, metavar="PORT",
                   help="serve it on localhost:PORT (default 8001); required to read bodies")
    s.set_defaults(fn=cmd_log)

    s = sub.add_parser("failures", help="worst-scoring tasks in a run, with trace paths")
    s.add_argument("run_id")
    s.add_argument("-n", "--limit", type=int, default=10)
    s.set_defaults(fn=cmd_failures)

    return p


def main(argv: list[str] | None = None) -> int:
    global _DOTENV_LOADED
    _DOTENV_LOADED = load_dotenv()  # before anything reads os.environ
    args = build_parser().parse_args(argv)
    return int(args.fn(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
