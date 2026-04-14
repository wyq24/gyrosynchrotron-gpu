"""CLI helpers for JAX tools on machines where direct script execution is brittle."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


_REEXEC_ENV = "MW_JAX_TOOL_REEXEC"


def maybe_reexec_tool_main(
    module_name: str,
    *,
    script_path: str | Path,
    repo_root: str | Path,
    extra_paths=(),
) -> None:
    if os.environ.get(_REEXEC_ENV) == "1":
        return
    if __name__ != "jax_ffi.cli_reentry":
        return
    if not sys.argv or not str(sys.argv[0]).endswith(".py"):
        return
    if Path(sys.argv[0]).resolve() != Path(script_path).resolve():
        return

    repo = Path(repo_root).resolve()
    search_paths = [str(repo)]
    for path in extra_paths:
        resolved = str(Path(path).resolve())
        if resolved not in search_paths:
            search_paths.append(resolved)

    code = (
        "import sys\n"
        f"sys.path[:0] = {search_paths!r}\n"
        f"import {module_name} as _tool\n"
        f"sys.argv = [{Path(sys.argv[0]).name!r}, *sys.argv[1:]]\n"
        "raise SystemExit(_tool.main())\n"
    )

    env = dict(os.environ)
    env[_REEXEC_ENV] = "1"
    proc = subprocess.run([sys.executable, "-c", code, *sys.argv[1:]], env=env, check=False)
    raise SystemExit(proc.returncode)
