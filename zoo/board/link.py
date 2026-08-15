"""USB/serial link hygiene: get the probe reachable, and prove it is.

Under WSL2 the ST-LINK reaches Linux only through usbipd, and its bus id
changes on every physical replug — so the busid in config is a hint, not a
guarantee, and rediscovery by VID:PID is the reliable path.
"""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from zoo.config import Toolchain

#: STMicroelectronics ST-LINK/V3.
STLINK_VID_PID = "0483:3754"


class BoardError(RuntimeError):
    """Always an infrastructure problem, never a model verdict."""


@dataclass
class LinkState:
    port: Path | None
    attached: bool
    busid: str = ""
    detail: str = ""


def _usbipd(tc: Toolchain, *args: str, timeout: float = 60.0) -> str:
    if not tc.usbipd or not Path(tc.usbipd).exists():
        raise BoardError("usbipd-win not found; set [board] usbipd in toolchain.toml")
    proc = subprocess.run(
        [tc.usbipd, *args], capture_output=True, text=True, timeout=timeout, check=False
    )
    return proc.stdout + proc.stderr


def discover_busid(tc: Toolchain) -> str | None:
    """Find the ST-LINK's current bus id by VID:PID, not by memory."""
    for line in _usbipd(tc, "list").splitlines():
        if STLINK_VID_PID in line:
            return line.split()[0]
    return None


def attach(tc: Toolchain, *, wait_s: float = 8.0) -> LinkState:
    """Make the probe visible in WSL, rediscovering its bus id if needed."""
    port = Path(tc.serial_port)
    if port.exists():
        return LinkState(port=port, attached=True, detail="already attached")

    busid = discover_busid(tc) or tc.usbipd_busid
    if not busid:
        raise BoardError(
            f"no ST-LINK ({STLINK_VID_PID}) visible to Windows — is the board plugged in?"
        )

    out = _usbipd(tc, "attach", "--wsl", "--busid", busid)
    deadline = time.monotonic() + wait_s
    while time.monotonic() < deadline:
        if port.exists():
            return LinkState(port=port, attached=True, busid=busid, detail=out.strip()[-200:])
        time.sleep(0.5)
    raise BoardError(f"attached busid {busid} but {port} never appeared: {out.strip()[-300:]}")


def kill_stale_gdbserver() -> int:
    """A leftover ST-LINK_gdbserver holds the probe and breaks the next run.

    Both prior projects made this the first line of every measurement script.
    """
    proc = subprocess.run(
        ["pkill", "-x", "ST-LINK_gdbserver"], capture_output=True, text=True, check=False
    )
    if proc.returncode == 0:
        time.sleep(1.0)
        return 1
    return 0


def holders(port: Path) -> list[str]:
    """Processes holding the serial port. Anything here will break a measure."""
    proc = subprocess.run(
        ["bash", "-lc", f"command -v fuser >/dev/null && fuser -v {port} 2>&1 || true"],
        capture_output=True, text=True, check=False,
    )
    return [ln for ln in proc.stdout.splitlines() if ln.strip()]


#: A wedged probe. Confirmed on this bench: neither an SWD reset nor a usbipd
#: detach/attach clears it — the device re-enumerates and still refuses. Only
#: unplugging the USB cable does. So retrying is not merely useless, it burns
#: minutes per attempt while looking like progress.
WEDGED = "DEV_USB_COMM_ERR"


def probe_responds(tc: Toolchain, *, timeout: float = 60.0) -> tuple[bool, str]:
    """Can the programmer reach the target over SWD at all?"""
    proc = subprocess.run(
        [str(tc.programmer_cli), "-c", "port=SWD", "mode=HOTPLUG"],
        capture_output=True, text=True, timeout=timeout, check=False,
    )
    out = proc.stdout + proc.stderr
    return (WEDGED not in out and proc.returncode == 0), out


def assert_probe_healthy(tc: Toolchain) -> None:
    """Fail fast, and tell the human the one thing that actually works.

    Called before a load rather than after three failed attempts: a wedged
    ST-LINK cannot be recovered in software, so the honest response is to stop
    and say so.
    """
    ok, out = probe_responds(tc)
    if ok:
        return
    if WEDGED in out:
        raise BoardError(
            f"ST-LINK is wedged ({WEDGED}). Neither an SWD reset nor a usbipd "
            "detach/attach clears this — physically unplug and replug the USB "
            "cable, then re-run. (Killing a runner mid-inference is the usual "
            "cause.)"
        )
    raise BoardError(f"probe did not respond over SWD: {out.strip()[-300:]}")


def preflight(tc: Toolchain) -> LinkState:
    """Everything that must be true before a measurement is attempted."""
    killed = kill_stale_gdbserver()
    state = attach(tc)
    if killed:
        state.detail = (state.detail + "; killed a stale gdbserver").strip("; ")
    if not state.port or not state.port.exists():
        raise BoardError(f"{tc.serial_port} not present after attach")
    assert_probe_healthy(tc)
    return state
