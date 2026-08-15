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
                 omod.FRONTEND_ONLY, omod.UNSUPPORTED):
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


if __name__ == "__main__":
    sys.exit(main())
