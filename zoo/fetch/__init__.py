"""Getting an ONNX file, which is three different problems wearing one hat.

`onnx-community/*` and friends publish plain Hugging Face blobs. Easy — with
one rule that matters more than it looks: **always take the fp32 file**, never
the `*_int8.onnx` / `*_quantized.onnx` sitting next to it. Those are ONNX
Runtime *dynamic* or QOperator quantisations, and ST's documentation is
explicit that dynamic/weight-only quantisation is unsupported and will be
silently converted back to float at import. The same goes for `qualcomm/*`
`w8a8` assets (AIMET/QAIRT uint8-asymmetric, `zero_point: 144`) and
`opencv/*_int8bq.onnx` (block-quantised). All three are excellent accuracy
references and wrong as deployment artifacts. The zoo fetches fp32 and runs
its own static QDQ pass.

`qualcomm/*` no longer host ONNX on Hugging Face at all. Each repo is a README
plus a `release_assets.json` pointing at zips on
`qaihub-public-assets.s3.us-west-2.amazonaws.com`, keyed by precision.

`STMicroelectronics/*` on Hugging Face are README-only pointer repos; the
artifacts live in the `stm32ai-modelzoo` GitHub repository behind git-LFS.
Their real value is the measured on-board benchmark tables in the READMEs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from zoo.config import RUNS_DIR

CACHE_DIR = RUNS_DIR / "cache"

#: Filenames that are vendor pre-quantisations, not deployment inputs.
_PREQUANTISED = re.compile(
    r"(_int8|_uint8|_quantized|_qdq|_q4|_q4f16|_bnb4|_fp16|_int8bq|_w8a8|_w8a16)\.onnx$",
    re.I,
)


class FetchError(RuntimeError):
    pass


@dataclass
class RemoteFile:
    repo: str
    filename: str
    size: int | None
    url: str

    @property
    def is_prequantised(self) -> bool:
        return bool(_PREQUANTISED.search(self.filename))

    @property
    def is_external_data(self) -> bool:
        return self.filename.endswith(".onnx_data")

    @property
    def stem(self) -> str:
        return Path(self.filename).stem


def hf_url(repo: str, filename: str, revision: str = "main") -> str:
    return f"https://huggingface.co/{repo}/resolve/{revision}/{filename}"


def list_hf_onnx(repo: str, revision: str = "main") -> list[RemoteFile]:
    """Every `.onnx` in a Hugging Face repo, with sizes, fp32 files first."""
    import requests

    resp = requests.get(
        f"https://huggingface.co/api/models/{repo}",
        params={"blobs": "true", "revision": revision},
        timeout=60,
    )
    if resp.status_code == 404:
        raise FetchError(f"no such Hugging Face model: {repo}")
    resp.raise_for_status()
    doc = resp.json()

    files: list[RemoteFile] = []
    for sib in doc.get("siblings", []):
        name = sib.get("rfilename", "")
        if not name.endswith((".onnx", ".onnx_data")):
            continue
        files.append(
            RemoteFile(
                repo=repo,
                filename=name,
                size=sib.get("size"),
                url=hf_url(repo, name, revision),
            )
        )
    if not files:
        raise FetchError(
            f"{repo} publishes no .onnx files. "
            "qualcomm/* repos indirect through release_assets.json to S3; "
            "STMicroelectronics/* repos are README pointers to the "
            "stm32ai-modelzoo GitHub repository."
        )
    # fp32 first, then by size: the fp32 file is what the zoo actually wants.
    files.sort(key=lambda f: (f.is_prequantised, f.size or 0))
    return files


def cache_path(repo: str, filename: str, revision: str = "main") -> Path:
    safe = repo.replace("/", "__")
    return CACHE_DIR / safe / revision / filename


def download(remote: RemoteFile, revision: str = "main", *, force: bool = False) -> Path:
    """Fetch to the content cache, skipping if already present and complete."""
    import requests

    dest = cache_path(remote.repo, remote.filename, revision)
    if dest.is_file() and not force:
        if remote.size is None or dest.stat().st_size == remote.size:
            return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(remote.url, stream=True, timeout=300) as resp:
        resp.raise_for_status()
        with tmp.open("wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
    tmp.replace(dest)
    return dest
