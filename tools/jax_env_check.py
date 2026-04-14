import argparse
import platform
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jax_ffi.cli_reentry import maybe_reexec_tool_main

maybe_reexec_tool_main(
    "jax_env_check",
    script_path=__file__,
    repo_root=REPO_ROOT,
    extra_paths=(REPO_ROOT / "tools",),
)


def _print_line(label, value):
    print(f"{label}: {value}")


def main():
    parser = argparse.ArgumentParser(description="Inspect the local Python/JAX environment for the staged JAX FFI work.")
    parser.add_argument("--require-jax", action="store_true", help="Exit nonzero if jax is unavailable.")
    parser.add_argument("--require-ffi", action="store_true", help="Exit nonzero if jax.ffi is unavailable.")
    parser.add_argument("--show-devices", action="store_true", help="Print visible JAX devices when available.")
    args = parser.parse_args()

    _print_line("python_executable", sys.executable)
    _print_line("python_version", sys.version.replace("\n", " "))
    _print_line("platform", platform.platform())

    try:
        import numpy as np

        _print_line("numpy_version", np.__version__)
    except Exception as exc:
        _print_line("numpy_error", f"{type(exc).__name__}: {exc}")

    try:
        import jax
        import jaxlib
    except Exception as exc:
        _print_line("jax_status", f"unavailable ({type(exc).__name__}: {exc})")
        return 1 if args.require_jax or args.require_ffi else 0

    _print_line("jax_version", jax.__version__)
    _print_line("jaxlib_version", jaxlib.__version__)
    _print_line("jax_enable_x64", getattr(jax.config, "jax_enable_x64", "unknown"))
    _print_line("default_backend", jax.default_backend())

    ffi = getattr(jax, "ffi", None)
    has_ffi = ffi is not None
    _print_line("has_jax_ffi", has_ffi)
    if has_ffi:
        for name in ("ffi_call", "register_ffi_target", "pycapsule"):
            _print_line(f"jax.ffi.{name}", hasattr(ffi, name))

    include_root = Path(jaxlib.__file__).resolve().parent / "include"
    c_api_header = include_root / "xla" / "ffi" / "api" / "c_api.h"
    api_header = include_root / "xla" / "ffi" / "api" / "api.h"
    _print_line("jaxlib_include_root", include_root)
    _print_line("xla_ffi_c_api_header", c_api_header.exists())
    _print_line("xla_ffi_api_header", api_header.exists())

    if args.show_devices:
        try:
            devices = jax.devices()
            _print_line("devices", ", ".join(f"{device.platform}:{device}" for device in devices))
        except Exception as exc:
            _print_line("devices_error", f"{type(exc).__name__}: {exc}")

    if args.require_ffi and not has_ffi:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
