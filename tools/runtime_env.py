from pathlib import Path
import shutil
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = REPO_ROOT / "examples"
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

import GScodes


def _read_build_info():
    path = REPO_ROOT / "source" / "MWTransferArr.buildinfo"
    info = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            info[key.strip()] = value.strip()
    return path, info


def collect_runtime_summary(libname=None):
    library = GScodes.resolve_library_path(libname, prefer_source=True, require_local_source=True)
    library_info = GScodes.inspect_shared_library(library)
    build_info_path, build_info = _read_build_info()

    summary = {
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "library": library,
        "library_binary_format": library_info["binary_format"],
        "library_expected_format": library_info["expected_binary_format"] or "unknown",
        "nvcc_path": shutil.which("nvcc") or "not found",
        "build_info_path": str(build_info_path),
    }
    for key, value in build_info.items():
        summary[f"build_{key}"] = value

    try:
        summary["cuda_available_in_library"] = "yes" if GScodes.cuda_available(library) else "no"
    except Exception as exc:
        summary["cuda_available_in_library"] = f"error: {exc}"

    return summary


def print_runtime_summary(libname=None):
    summary = collect_runtime_summary(libname)
    print("Runtime environment")
    print(f"python_executable: {summary['python_executable']}")
    print(f"python_version: {summary['python_version']}")
    print(f"library: {summary['library']}")
    print(
        f"library_binary_format: {summary['library_binary_format']} "
        f"(expected {summary['library_expected_format']})"
    )
    print(f"build_info: {summary['build_info_path']}")
    if "build_host_os" in summary:
        print(f"build_host_os: {summary['build_host_os']}")
    if "build_cxx" in summary:
        print(f"build_cxx: {summary['build_cxx']}")
    if "build_cxx_path" in summary:
        print(f"build_cxx_path: {summary['build_cxx_path']}")
    if "build_cxx_version" in summary:
        print(f"build_cxx_version: {summary['build_cxx_version']}")
    if "build_cppflags" in summary:
        print(f"build_cppflags: {summary['build_cppflags']}")
    if "build_cxxflags" in summary:
        print(f"build_cxxflags: {summary['build_cxxflags']}")
    if "build_ldflags" in summary:
        print(f"build_ldflags: {summary['build_ldflags']}")
    if "build_ldlibs" in summary:
        print(f"build_ldlibs: {summary['build_ldlibs']}")
    if "build_cuda" in summary:
        print(f"build_cuda: {summary['build_cuda']}")
    if "build_nvcc_path" in summary:
        print(f"build_nvcc_path: {summary['build_nvcc_path']}")
    print(f"nvcc_path: {summary['nvcc_path']}")
    print(f"cuda_available_in_library: {summary['cuda_available_in_library']}")
