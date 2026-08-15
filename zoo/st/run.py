"""Subprocess wrapper: explicit cwd, explicit env, captured logs, no surprises.

Three habits are enforced here because their absence caused real, expensive
bugs in the two prior projects on this machine:

1. **cwd is always explicit.** `stedgeai` drops `st_ai_output/` and `st_ai_ws/`
   scratch directories into whatever directory it is invoked from. Pointing
   cwd at the per-run output directory keeps that litter where it belongs.

2. **The full argv is recorded, always.** Both prior projects' only surviving
   record of how a model was compiled was the `Parameters:` line inside a
   generated text report. Every command the zoo runs is written next to its
   output as `cmd.txt` before it starts, so a killed run still says what it
   was doing.

3. **GCC_PATH is exported.** ST's own `NPU_Validation/armgcc/Makefile`
   hardcodes a Windows path on line 1, so any build launched without an
   exported GCC_PATH fails with a bare `Error 127`.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class RunResult:
    argv: list[str]
    cwd: Path
    returncode: int
    stdout: str
    stderr: str
    duration_s: float
    log_path: Path | None = None

    @property
    def ok(self) -> bool:
        return self.returncode == 0

    @property
    def combined(self) -> str:
        """stdout and stderr together, for signature fingerprinting.

        ST's tools are inconsistent about which stream carries an error --
        `TOOL ERROR` lines show up on stdout while Python tracebacks from the
        loader scripts go to stderr -- so fault matching must see both.
        """
        return self.stdout + ("\n" + self.stderr if self.stderr else "")

    def pretty_cmd(self) -> str:
        return " ".join(shlex.quote(a) for a in self.argv)


def run(
    argv: list[str | Path],
    *,
    cwd: Path,
    env_extra: dict[str, str] | None = None,
    timeout_s: float = 1800.0,
    log_dir: Path | None = None,
    log_name: str = "run",
    check: bool = False,
) -> RunResult:
    """Run a command with the zoo's conventions.

    `cwd` is created if absent. If `log_dir` is given, `<log_name>.cmd` and
    `<log_name>.log` are written there -- the former *before* the process
    starts, so a hung or killed run still leaves evidence of what it was.
    """
    argv_s = [str(a) for a in argv]
    cwd.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)

    log_path: Path | None = None
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / f"{log_name}.cmd").write_text(
            " ".join(shlex.quote(a) for a in argv_s) + f"\n# cwd: {cwd}\n"
        )
        log_path = log_dir / f"{log_name}.log"

    started = time.monotonic()
    try:
        proc = subprocess.run(
            argv_s,
            cwd=str(cwd),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        rc, out, err = proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as exc:
        rc = 124  # conventional timeout exit code
        out = exc.stdout.decode("utf-8", "replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        err = exc.stderr.decode("utf-8", "replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        err += f"\n[zoo] timed out after {timeout_s:.0f}s"
    except FileNotFoundError as exc:
        rc, out, err = 127, "", f"[zoo] executable not found: {exc}"

    duration = time.monotonic() - started

    result = RunResult(
        argv=argv_s,
        cwd=cwd,
        returncode=rc,
        stdout=out,
        stderr=err,
        duration_s=duration,
        log_path=log_path,
    )

    if log_path is not None:
        log_path.write_text(
            f"$ {result.pretty_cmd()}\n# cwd: {cwd}\n"
            f"# exit: {rc}  duration: {duration:.2f}s\n"
            f"{'-' * 72}\n{out}\n"
            + (f"{'-' * 72}\nSTDERR\n{err}\n" if err else "")
        )

    if check and not result.ok:
        raise RuntimeError(
            f"command failed (exit {rc}): {result.pretty_cmd()}\n"
            f"{result.combined[-4000:]}"
        )
    return result


def gcc_env(gcc_path: Path) -> dict[str, str]:
    """Environment for anything that shells out to ST's armgcc Makefiles.

    `GCC_PATH` must be *exported*, not merely set in the Makefile: ST's
    `NPU_Validation/armgcc/Makefile` opens with a hardcoded Windows path
    (`GCC_PATH ?= "/c/Users/foobar/TOOLS/..."`), so a link step without this
    dies with a bare `Error 127` that names nothing.
    """
    return {
        "GCC_PATH": str(gcc_path),
        "PATH": f"{gcc_path}{os.pathsep}{os.environ.get('PATH', '')}",
    }
