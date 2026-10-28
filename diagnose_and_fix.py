"""
Diagnose where `toffee` is being imported from and check if the patched
cli.py is loaded. If not, copy the patched files from the unzipped
toffee-full.zip into the actual install location.

Usage:
    python diagnose_and_fix.py /path/to/toffee-full.zip
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


def find_toffee_quant_cli() -> Path | None:
    """Find where toffee.quant.cli is actually being loaded from."""
    try:
        spec = importlib.util.find_spec("toffee.quant.cli")
        if spec is None or spec.origin is None:
            return None
        return Path(spec.origin)
    except (ImportError, ModuleNotFoundError):
        return None


def has_patched_flags(cli_path: Path) -> bool:
    """Check whether the cli.py at the given path has --parallel-workers."""
    try:
        src = cli_path.read_text()
        return "--parallel-workers" in src
    except OSError:
        return False


def extract_zip(zip_path: Path, dest: Path) -> Path:
    """Extract the toffee-full.zip and return the path to the extracted toffee/quant dir."""
    extract_dir = dest / "_toffee_full_extracted"
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)
    return extract_dir / "toffee" / "quant"


def copy_patched_files(src_quant: Path, dst_quant: Path) -> None:
    """Copy the four refined files from src_quant to dst_quant."""
    files = ["cli.py", "rco.py", "gsq.py", "refine.py"]
    for name in files:
        src = src_quant / name
        dst = dst_quant / name
        if not src.exists():
            print(f"  ERROR: source file not found: {src}")
            continue
        # Backup the existing file.
        if dst.exists():
            backup = dst.with_suffix(dst.suffix + ".bak")
            shutil.copy2(dst, backup)
            print(f"  backed up {dst.name} -> {backup.name}")
        shutil.copy2(src, dst)
        print(f"  copied {name}  ({src.stat().st_size} -> {dst.stat().st_size} bytes)")


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    zip_path = Path(sys.argv[1]).expanduser().resolve()
    if not zip_path.exists():
        print(f"ERROR: zip not found: {zip_path}")
        return 1

    print("=" * 70)
    print("Diagnosing toffee install")
    print("=" * 70)

    cli_path = find_toffee_quant_cli()
    if cli_path is None:
        print("\nERROR: cannot find toffee.quant.cli in the current Python.")
        print("\nLikely causes:")
        print("  - You're not in the right venv (run `source .venv/bin/activate`)")
        print("  - toffee isn't installed (run `pip install -e .` from the project root)")
        print("  - PYTHONPATH isn't set to the project root")
        print("\nTry:")
        print("  cd /home/solis/Documentos/Code/OPAL")
        print("  source .venv/bin/activate.fish   # or .venv/bin/activate")
        print("  PYTHONPATH=$PWD python diagnose_and_fix.py /path/to/toffee-full.zip")
        return 1

    print(f"\ntoffee.quant.cli is loaded from:")
    print(f"  {cli_path}")
    print(f"\nHas --parallel-workers flag: {has_patched_flags(cli_path)}")

    if has_patched_flags(cli_path):
        print("\nNo fix needed. The patched cli.py is being loaded.")
        print("\nIf `toffee quant gsq --parallel-workers 4` still fails, check:")
        print("  - Are you running the right `toffee` command? (try `which toffee`)")
        print("  - Are you in the right venv? (try `which python`)")
        return 0

    print("\n" + "=" * 70)
    print("Fixing: copying patched files into the install location")
    print("=" * 70)

    src_quant = extract_zip(zip_path, Path("/tmp"))
    if not src_quant.exists():
        print(f"ERROR: extracted quant/ dir not found at {src_quant}")
        return 1
    print(f"\nSource: {src_quant}")
    print(f"Dest:   {cli_path.parent}")

    copy_patched_files(src_quant, cli_path.parent)

    # Re-check.
    print("\n" + "=" * 70)
    print("Verifying the fix")
    print("=" * 70)
    # Force re-import.
    for mod_name in list(sys.modules.keys()):
        if mod_name.startswith("toffee"):
            del sys.modules[mod_name]
    cli_path = find_toffee_quant_cli()
    if cli_path is None:
        print("ERROR: cannot find toffee.quant.cli after the fix.")
        return 1
    print(f"\ntoffee.quant.cli is now loaded from: {cli_path}")
    print(f"Has --parallel-workers flag: {has_patched_flags(cli_path)}")
    if has_patched_flags(cli_path):
        print("\nFix verified. Re-run your command:")
        print("  toffee quant gsq --gguf ... --f16-gguf ... --parallel-workers 4 ...")
        return 0
    print("\nFix did NOT take. You probably have a non-editable install.")
    print("Re-install editable:")
    print("  pip install -e .")
    return 1


if __name__ == "__main__":
    sys.exit(main())
