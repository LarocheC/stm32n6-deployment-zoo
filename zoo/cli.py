"""`zoo` command-line entry point."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from zoo.config import RUNS_DIR, ConfigError, load_toolchain

_GLYPH = {"ok": "✓", "warn": "!", "fail": "✗"}
_COLOUR = {"ok": "\033[32m", "warn": "\033[33m", "fail": "\033[31m"}
_RESET = "\033[0m"


def _emit(status: str, name: str, detail: str, *, colour: bool) -> None:
    glyph = _GLYPH.get(status, "?")
    if colour:
        glyph = f"{_COLOUR.get(status, '')}{glyph}{_RESET}"
    line = f"  {glyph} {name}"
    if detail:
        line += f"\n      {detail}" if len(detail) > 60 else f"  — {detail}"
    print(line)


def cmd_doctor(args: argparse.Namespace) -> int:
    from zoo.st import toolchain as tcmod

    colour = sys.stdout.isatty() and not args.no_colour
    try:
        tc = load_toolchain()
    except ConfigError as exc:
        print(f"\nconfiguration error:\n  {exc}\n", file=sys.stderr)
        return 2

    workdir = RUNS_DIR / "_doctor"
    print(f"\nzoo doctor — level: {args.level}\n")

    checks = tcmod.probe_tools(tc, workdir)
    if args.level in ("board", "full"):
        print("  (board-level checks land in P5; running tools level only)\n")

    for check in checks:
        _emit(check.status, check.name, check.detail, colour=colour)

    n_fail = sum(1 for c in checks if c.status == "fail")
    n_warn = sum(1 for c in checks if c.status == "warn")
    print(f"\n  {len(checks)} checks — {n_fail} failed, {n_warn} warnings\n")
    return 1 if n_fail else 0


def cmd_profiles(args: argparse.Namespace) -> int:
    from zoo.st import profiles as pmod

    tc = load_toolchain()
    path = pmod.materialise(tc.core_tag, force=args.force)
    loaded = pmod.load(tc.core_tag)
    print(f"\nmaterialised: {path}\n")
    width = max(len(n) for n in loaded)
    for name in sorted(loaded):
        prof = loaded[name]
        mm = pmod.read_mpool(prof.mpool)
        flash = mm.octoflash
        print(
            f"  {name:<{width}}  on-chip {mm.onchip_bytes // 1024:>5} KB  "
            f"flash {(flash.size_bytes // 1024 // 1024) if flash else 0:>4} MB"
            f"{f' @ 0x{flash.offset:08X}' if flash and flash.size_bytes else '':>14}"
        )
    print()
    return 0


def cmd_ops(args: argparse.Namespace) -> int:
    from zoo.graph import ops as omod

    tc = load_toolchain()
    table = omod.load(tc, refresh=args.refresh)

    if args.query:
        for op in args.query:
            info = table.info(op)
            tier = table.tier(op)
            line = f"\n  {op}  →  {tier}"
            if info and info.unlocked_by:
                line += f"   ({table.tier(op, unlocked=True)} with {info.unlocked_by})"
            print(line)
            if info and info.requires_constant_input:
                print("      hardware only when the named operand is a constant/initializer")
            if op in omod.HARD_BLOCKED:
                print(f"      {omod.HARD_BLOCKED[op]}")
            elif info and info.comment:
                print(f"      {info.comment}")
            elif not info:
                print(
                    "      not in ST's accelerator mapping table"
                    + (" — the importer accepts it, but NPU behaviour is undocumented"
                       if op in table.frontend else "")
                )
        print()
        return 0

    counts: dict[str, int] = {}
    for op in set(table.mapping) | table.frontend:
        counts[table.tier(op)] = counts.get(table.tier(op), 0) + 1
    print(f"\nop oracle — ST Edge AI Core {table.core_version}")
    print(f"  accelerator mapping table : {len(table.mapping)} ops")
    print(f"  front-end parser vocabulary: {len(table.frontend)} ops")
    print("\n  tiers:")
    for tier in (omod.HW, omod.MIXED, omod.SW, omod.SW_INT, omod.SW_FLOAT,
                 omod.PLUMBING, omod.FRONTEND_ONLY, omod.UNSUPPORTED):
        if counts.get(tier):
            print(f"    {tier:<16} {counts[tier]:>4}")

    gated = {n: i for n, i in table.mapping.items() if i.unlocked_by and i.fallback_tier}
    if gated:
        print("\n  hardware gated behind a compiler flag (off in every stock profile):")
        for name, info in sorted(gated.items()):
            print(f"    {name:<12} {info.fallback_tier} → {info.tier} via {info.unlocked_by}")

    cond = {n: i for n, i in table.mapping.items() if i.requires_constant_input}
    if cond:
        print("\n  hardware conditional on a constant operand (resolve against the graph):")
        for name in sorted(cond):
            print(f"    {name}")
    print()
    return 0


def cmd_init(args: argparse.Namespace) -> int:
    from zoo import draft as dmod
    from zoo.config import ROOT
    from zoo.graph import ops as omod

    result = dmod.draft(args.repo, revision=args.revision, max_mb=args.max_mb)
    drafted = result.graphs

    print(f"\n{args.repo}: {len(drafted)} graph(s), domain={result.domain}\n")
    table = None
    try:
        table = omod.load(load_toolchain())
    except Exception:  # noqa: BLE001 - drafting must work without a toolchain
        pass

    for item in drafted:
        size = f"{item.size / 1e6:.1f} MB" if item.size else "?"
        print(f"  {item.graph_id}  ({size})")
        if item.probe is None:
            print(f"      skipped: {item.skip_reason}")
            continue
        p = item.probe
        print(f"      ir={p.ir_version} opset={p.default_opset} "
              f"nodes={sum(p.op_types.values())} max_dim={p.max_tensor_dim} rank={p.max_rank}")
        if table:
            counts = {
                tier: sum(g.values())
                for tier, g in table.census(dict(p.op_types)).items()
            }
            print(f"      op tiers: {counts}")
            blocked = table.census(dict(p.op_types)).get(omod.UNSUPPORTED, {})
            if blocked:
                print(f"      BLOCKED: {blocked}")
        dyn = p.dynamic_conditional_ops
        if dyn:
            print(f"      {len(dyn)} MatMul/Gemm/Conv with a dynamic operand → M55 fallback")
        if p.initializer_inputs:
            print(f"      {len(p.initializer_inputs)} initializers declared as graph inputs "
                  "(old IR-v3 export) → run the simplifier")
        unpinned = sorted({d for s in p.inputs for d in s.symbolic_dims})
        if unpinned:
            print(f"      symbolic dims to pin: {unpinned}")

    if args.stdout:
        print("\n" + "-" * 72)
        print(result.text)
        return 0

    out = Path(args.out) if args.out else result.path_in(ROOT / "models")
    if out.exists() and not args.force:
        print(f"\n  refusing to overwrite {out.relative_to(ROOT)} (use --force)\n")
        return 1
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(result.text)
    print(f"\n  wrote {out.relative_to(ROOT)}  —  {result.todo_count} TODO(s) to resolve\n")
    return 0


_SEV_MARK = {"error": "✗", "warn": "!", "info": "·"}


def cmd_lint(args: argparse.Namespace) -> int:
    from zoo.config import ROOT, load_policy
    from zoo.graph import lint as lmod
    from zoo.graph import ops as omod
    from zoo.graph import probe as pmod

    tc = load_toolchain()
    table = omod.load(tc)
    policy = load_policy()

    targets: list[tuple[str, Path, dict]] = []
    for target in args.target:
        path = Path(target)
        if path.suffix == ".onnx" and path.is_file():
            targets.append((path.name, path, {}))
            continue
        # Otherwise treat it as a recipe id or path.
        from zoo import fetch as fmod
        from zoo import recipe as recmod

        rpath = path if path.is_file() else None
        if rpath is None:
            matches = [p for p in (ROOT / "models").rglob("*.toml") if p.stem == target]
            if not matches:
                print(f"  no recipe or ONNX file matching {target!r}", file=sys.stderr)
                return 2
            rpath = matches[0]
        rec = recmod.load(rpath)
        for graph in rec.graphs:
            if not graph.enabled:
                print(f"  ⚪ {rec.id}/{graph.id}: skipped — {graph.skip_reason}")
                continue
            remotes = [
                f for f in fmod.list_hf_onnx(rec.source, rec.revision)
                if f.filename == graph.file
            ]
            if not remotes:
                print(f"  ✗ {rec.id}/{graph.id}: {graph.file} not found in {rec.source}",
                      file=sys.stderr)
                return 2
            local = fmod.download(remotes[0], rec.revision)
            targets.append((f"{rec.id}/{graph.id}", local, graph.pin))

    worst = 0
    for label, path, pin in targets:
        probe = pmod.probe_file(path)
        result = lmod.lint(probe, table, policy, pinned=pin, unlocked=args.unlocked)

        status = "PASS" if result.ok else "FAIL"
        print(f"\n  {label}  [{status}]")
        counts = {t: sum(g.values()) for t, g in result.census.items()}
        print(f"      {result.node_count} nodes · tiers {counts}")
        print(f"      Cortex-M55 instances: {result.sw_instances}"
              f" (+{result.dynamic_conditional} dynamic MatMul/Gemm)"
              f" → {result.sw_instances_unlocked} with recognition passes")

        for v in result.violations:
            if v.severity == "info" and not args.verbose:
                continue
            print(f"      {_SEV_MARK.get(v.severity, '?')} [{v.rule}] {v.message}")
            if v.remedy:
                print(f"          → {v.remedy}")
        if result.suggested_patches:
            print(f"      patches that would help: {', '.join(result.suggested_patches)}")
        worst = max(worst, 1 if not result.ok else 0)

    print()
    return worst


def cmd_screen(args: argparse.Namespace) -> int:
    import subprocess

    from zoo import funnel as fmod
    from zoo import recipe as recmod
    from zoo.config import ROOT, load_policy
    from zoo.graph import ops as omod
    from zoo.store.events import EventLog
    from zoo.store.schema import Status

    tc = load_toolchain()
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
            capture_output=True, text=True, check=False,
        ).stdout.strip()
    except Exception:  # noqa: BLE001
        commit = ""

    ctx = fmod.Context(
        toolchain=tc,
        table=omod.load(tc),
        policy=load_policy(),
        log=EventLog(),
        zoo_commit=commit,
    )

    recipes = recmod.discover(ROOT / "models")
    if args.only:
        wanted = set(args.only)
        recipes = [r for r in recipes if r.id in wanted]
        if not recipes:
            print(f"  no recipe matching {sorted(wanted)}", file=sys.stderr)
            return 2

    print(f"\nscreening {len(recipes)} recipe(s) — run {ctx.run_id}\n")
    failed = 0
    for rec in recipes:
        for graph in rec.graphs:
            events = fmod.screen_graph(rec, graph, ctx, unlocked=args.unlocked)
            last = events[-1] if events else None
            reached = last.stage if last else "—"
            ok = last is not None and last.status == Status.PASS
            mark = "✓" if ok else ("⚪" if last and last.status == Status.SKIP else "✗")
            print(f"  {mark} {rec.id}/{graph.id:<24} reached {reached}")
            for ev in events:
                if ev.status != Status.PASS and ev.error:
                    print(f"        [{ev.stage}] {ev.failure_class or ''} {ev.error[:150]}")
            if not ok and (not last or last.status != Status.SKIP):
                failed += 1

    print(f"\n  {len(ctx.log.read_all())} events in {EventLog().path.relative_to(ROOT)}")
    print("  run `zoo report` to fold them into RESULTS.md\n")
    return 1 if failed else 0


def cmd_measure(args: argparse.Namespace) -> int:
    import subprocess

    from zoo import funnel as fmod
    from zoo import recipe as recmod
    from zoo.config import ROOT, load_policy
    from zoo.graph import ops as omod
    from zoo.store.events import EventLog
    from zoo.store.schema import Status

    tc = load_toolchain()
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                            capture_output=True, text=True, check=False).stdout.strip()
    ctx = fmod.Context(toolchain=tc, table=omod.load(tc), policy=load_policy(),
                       log=EventLog(), zoo_commit=commit)

    all_recipes = recmod.discover(ROOT / "models")
    recipes = all_recipes
    if args.only:
        recipes = [r for r in recipes if r.id in set(args.only)]
    if not recipes:
        print("  no matching recipe", file=sys.stderr)
        return 2

    # Policy overrides, so a quick sanity pass does not have to pretend to be
    # evidence. Whatever is used lands in the event, so a row measured with
    # --loads 1 is permanently distinguishable from one that met the bar.
    if args.loads is not None:
        ctx.policy.setdefault("measure", {})["min_loads"] = args.loads
    if args.invokes is not None:
        ctx.policy.setdefault("measure", {})["invokes_per_load"] = args.invokes

    canary = None
    if not args.no_canary and not args.no_board:
        canary = fmod.build_canary(ctx, all_recipes)
        wanted = ctx.policy.get("measure", {}).get("canary_model")
        if canary is None and wanted:
            print(f"  ! no canary available ({wanted}); rows will record no bench reference")
        elif canary is not None:
            print(f"  canary: {wanted} — read once before each row")

    failed = 0
    cfg = ctx.policy.get("measure", {})
    print(f"\nmeasuring {len(recipes)} recipe(s) — run {ctx.run_id} — "
          f"{cfg.get('min_loads', 3)} load(s) x {cfg.get('invokes_per_load', 10)} invoke(s)\n")

    wanted_graphs = set(args.graph or [])
    for rec in recipes:
        graphs = [g for g in rec.enabled_graphs if not wanted_graphs or g.id in wanted_graphs]
        for graph in graphs:
            def _progress(reading, _rec=rec, _g=graph) -> None:
                if reading.ok:
                    spread = (
                        f" ({reading.min_ms}/{reading.max_ms}/{reading.std_ms})"
                        if reading.std_ms is not None else ""
                    )
                    print(f"        load {reading.index}: {reading.latency_ms} ms{spread}")
                else:
                    print(f"        load {reading.index}: failed — {reading.error[:100]}")

            # The canary's own row needs no canary: reading it would measure
            # the same graph twice and compare it against itself.
            row_canary = None if rec.id == cfg.get("canary_model") else canary
            events = fmod.measure_graph(
                rec, graph, ctx, canary=row_canary, board=not args.no_board,
                on_reading=_progress,
            )
            last = events[-1] if events else None
            ok = last is not None and last.status == Status.PASS
            print(f"  {'✓' if ok else '✗'} {rec.id}/{graph.id:<22} reached {last.stage if last else '—'}")
            for ev in events:
                m = ev.metrics
                if ev.status == Status.PASS and m.get("latency_ms_median"):
                    gate = m.get("determinism_gate", "?")
                    mark = "✓" if gate == "trusted" else "!"
                    cv = m.get("latency_ms_cv")
                    print(f"        {mark} {m['latency_ms_median']} ms median · "
                          f"cos {m.get('ontarget_cos')} · {m.get('profile_used')} · "
                          f"{m.get('loads_ok')}x{m.get('invokes_per_load')} loads"
                          + (f" · cv {cv * 100:.2f}%" if cv is not None else "")
                          + (f" · predicted {m['predicted_ms']:.4f} ms "
                             f"(x{m['predicted_vs_measured']:.1f})" if m.get("predicted_ms") else ""))
                    print(f"          gate: {gate} — {m.get('determinism_reason', '')}")
                elif ev.status != Status.PASS and ev.error:
                    tag = "INFRA " if ev.is_infra else ""
                    print(f"        [{ev.stage}] {tag}{ev.failure_class} {ev.error[:120]}")
            if not ok:
                failed += 1
    print(f"\n  run `zoo report` to fold {len(ctx.log.read_all())} events into RESULTS.md\n")
    return 1 if failed else 0


def cmd_report(args: argparse.Namespace) -> int:
    from zoo.config import ROOT
    from zoo.store import report as rmod
    from zoo.store import snapshot as smod

    snap = smod.load_and_fold()

    if args.new_signatures:
        unknown = [s for s in snap.signatures.values() if not s["known_issue"]]
        if not unknown:
            print("\n  no unmatched failure signatures — the catalogue explains everything seen\n")
            return 0
        print(f"\n  {len(unknown)} unmatched signature(s) — the discovery queue\n")
        for sig in sorted(unknown, key=lambda s: -s["count"]):
            print(f"  ×{sig['count']:<3} {sig['failure_class'] or 'UNKNOWN'}  "
                  f"[{', '.join(sig['stages'])}]  {', '.join(sig['models'][:5])}")
            print(f"        {sig['example_error'][:160]}")
        print()
        return 0

    recipes = {}
    try:
        from zoo import recipe as recmod

        recipes = {r.id: r for r in recmod.discover(ROOT / "models")}
    except Exception:  # noqa: BLE001 - a broken recipe must not block reporting
        pass

    snap_path = smod.write(snap)
    md_path = rmod.write(snap, recipes=recipes)
    print(f"\n  {snap.total_events} events → {len(snap.graphs)} graph(s)")
    print(f"  wrote {snap_path.relative_to(ROOT)} and {md_path.relative_to(ROOT)}\n")
    return 0


def _print_issue(issue, *, full: bool) -> None:  # noqa: ANN001
    import textwrap

    flags = [issue.zoo_action, issue.stage] + (["SILENT"] if issue.silent else [])
    print(f"\n  {issue.id}  [{' · '.join(f for f in flags if f)}]")
    wrap = textwrap.TextWrapper(width=96, initial_indent="      ", subsequent_indent="      ")
    print(wrap.fill(issue.title))
    fields = (("symptom", issue.symptom), ("cause", issue.cause)) if full else ()
    for label, text in (*fields, ("fix", issue.workaround)):
        if text:
            print(wrap.fill(f"{label}: {text}"))
    if full:
        if issue.error_signature:
            print(f"      signature: {issue.error_signature!r}")
        for src in issue.sources:
            print(f"      source: {src}")


def cmd_atlas(args: argparse.Namespace) -> int:
    from collections import Counter

    from zoo.faults import signatures as sigs

    issues = sigs.known_issues()
    if args.classify:
        text = sys.stdin.read() if args.classify == "-" else Path(args.classify).read_text(
            errors="replace"
        )
        result = sigs.classify(text)
        match = next((i for i in issues if i.id == result.known_issue), None)
        print(f"\n  class: {result.failure_class}   infra: {result.is_infra}")
        if match is None:
            print("  no catalogue entry matches — a candidate for the atlas\n")
            return 1
        _print_issue(match, full=True)
        print()
        return 0

    if args.id:
        match = next((i for i in issues if i.id == args.id), None)
        if match is None:
            print(f"\n  no entry {args.id!r}\n", file=sys.stderr)
            return 1
        _print_issue(match, full=True)
        print()
        return 0

    if not args.terms and not args.silent and not args.action:
        by_action = Counter(i.zoo_action for i in issues)
        print(f"\n  failure atlas — {len(issues)} constraints, "
              f"{sum(i.silent for i in issues)} silent, "
              f"{sum(bool(i.error_signature) for i in issues)} with a verbatim signature")
        for action, n in by_action.items():
            print(f"    {action:<22} {n:>3}")
        print("\n  zoo atlas <words>   search · --id <id> · --classify <log|->\n")
        return 0

    found = sigs.search_catalogue(args.terms, silent_only=args.silent, action=args.action)
    for issue in found:
        _print_issue(issue, full=args.full)
    print(f"\n  {len(found)} of {len(issues)} entries\n")
    return 0 if found else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="zoo",
        description="Push models through the ST Edge AI toolchain onto an STM32N6570-DK.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_doctor = sub.add_parser("doctor", help="validate the toolchain and board")
    p_doctor.add_argument(
        "--level",
        choices=("tools", "board", "full"),
        default="tools",
        help="tools = no hardware needed (default)",
    )
    p_doctor.add_argument("--no-colour", action="store_true")
    p_doctor.set_defaults(func=cmd_doctor)

    p_prof = sub.add_parser("profiles", help="materialise and list compilation profiles")
    p_prof.add_argument("--force", action="store_true", help="rewrite even if unchanged")
    p_prof.set_defaults(func=cmd_profiles)

    p_ops = sub.add_parser(
        "ops", help="query the operator oracle (which ops reach the NPU)"
    )
    p_ops.add_argument("query", nargs="*", help="ONNX op names to look up; omit for a summary")
    p_ops.add_argument("--refresh", action="store_true", help="rebuild the cached table")
    p_ops.set_defaults(func=cmd_ops)

    p_init = sub.add_parser("init", help="draft a recipe from a Hugging Face model id")
    p_init.add_argument("repo", help='e.g. "onnx-community/mobilenet_v2_1.0_224"')
    p_init.add_argument("--revision", default="main")
    p_init.add_argument("--out", help="output path (default: models/<domain>/<slug>.toml)")
    p_init.add_argument("--max-mb", type=float, default=400.0,
                        help="skip downloading graphs larger than this; head-scan them instead")
    p_init.add_argument("--stdout", action="store_true", help="print instead of writing")
    p_init.add_argument("--force", action="store_true", help="overwrite an existing recipe")
    p_init.set_defaults(func=cmd_init)

    p_lint = sub.add_parser("lint", help="static screen: reject what cannot work, no ST tool")
    p_lint.add_argument("target", nargs="+", help="recipe id, recipe path, or .onnx path")
    p_lint.add_argument("--unlocked", action="store_true",
                        help="assume the transformer profile's recognition passes are on")
    p_lint.add_argument("-v", "--verbose", action="store_true", help="show info-level notes")
    p_lint.set_defaults(func=cmd_lint)

    p_screen = sub.add_parser(
        "screen", help="run every recipe through the board-free funnel"
    )
    p_screen.add_argument("--only", nargs="*", help="restrict to these recipe ids")
    p_screen.add_argument("--unlocked", action="store_true",
                          help="assume the transformer profile's recognition passes")
    p_screen.set_defaults(func=cmd_screen)

    p_measure = sub.add_parser(
        "measure", help="quantise, compile and measure on the board (needs hardware)"
    )
    p_measure.add_argument("--only", nargs="*", help="restrict to these recipe ids")
    p_measure.add_argument(
        "--graph", nargs="*",
        help="restrict to these graph ids. A recipe with several graphs otherwise "
             "costs a full pass to re-measure one of them, which on a bench that "
             "wedges every few loads is most of a session",
    )
    p_measure.add_argument(
        "--loads", type=int,
        help="full reloads per row (default: policy min_loads). Below the policy "
             "figure the row is recorded as 'insufficient', never as trusted",
    )
    p_measure.add_argument(
        "--invokes", type=int,
        help="invokes per load (default: policy invokes_per_load)",
    )
    p_measure.add_argument(
        "--no-canary", action="store_true",
        help="skip the bench canary — faster, and the rows say so",
    )
    p_measure.add_argument(
        "--no-board", action="store_true",
        help="stop after the compile; quantise and generate need no hardware",
    )
    p_measure.set_defaults(func=cmd_measure)

    p_report = sub.add_parser("report", help="fold the event log into RESULTS.md")
    p_report.add_argument(
        "--new-signatures",
        action="store_true",
        help="list failure signatures with no catalogue entry — the discovery queue",
    )
    p_report.set_defaults(func=cmd_report)

    p_atlas = sub.add_parser(
        "atlas", help="search the failure atlas, or classify a log against it (no toolchain)"
    )
    p_atlas.add_argument("terms", nargs="*", help="words that must all appear in an entry")
    p_atlas.add_argument("--id", help="print one entry in full")
    p_atlas.add_argument("--classify", metavar="LOG",
                         help="match a log file (or - for stdin) against the signatures")
    p_atlas.add_argument("--silent", action="store_true", help="only silent failures")
    p_atlas.add_argument("--action", help="only one section, e.g. board-invariant")
    p_atlas.add_argument("--full", action="store_true", help="include symptom, cause, sources")
    p_atlas.set_defaults(func=cmd_atlas)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except ConfigError as exc:
        print(f"\nconfiguration error:\n  {exc}\n", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130
    except BrokenPipeError:
        # `zoo atlas stall | head`: the reader left; that is not an error.
        import os

        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        return 0


if __name__ == "__main__":
    sys.exit(main())
