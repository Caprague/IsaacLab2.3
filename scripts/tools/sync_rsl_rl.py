# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sync local ``rsl_rl/`` directory to the path specified by ``RSL_RL_PATH``.

This script copies the project-managed RSL RL library code to the Isaac Sim
Python environment on the training host, ensuring the training algorithm
library stays in sync with the project repository.

Behavior:
    - If ``RSL_RL_PATH`` does not exist, the directory is created before syncing.
    - If ``RSL_RL_PATH`` already exists, all contents are removed and replaced
      with a fresh copy, eliminating any drift.

Usage:

.. code-block:: bash

    # Ensure RSL_RL_PATH is set, e.g.:
    export RSL_RL_PATH=/home/ls_gms/Isaac/IsaacSim5.1/kit/python/lib/python3.11/site-packages

    # Run the sync script
    python scripts/tools/sync_rsl_rl.py
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Sync the local rsl_rl/ directory to the RSL_RL_PATH target.",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help=(
            "Path to the local rsl_rl/ source directory. "
            "Defaults to the 'rsl_rl/' folder at the project root (auto-detected relative to this script)."
        ),
    )
    parser.add_argument(
        "--target",
        type=str,
        default=None,
        help=(
            "Target path to sync to. Overrides the RSL_RL_PATH environment variable. "
            "If neither is set, the script will exit with an error."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview the sync operation without making any changes.",
    )
    return parser.parse_args()


def resolve_source_dir(script_path: Path) -> Path:
    """Resolve the rsl_rl source directory.

    The source directory is expected to be at the project root, i.e.,
    ``<project_root>/rsl_rl/``.

    Args:
        script_path: Absolute path to this script file.

    Returns:
        Resolved source directory path.

    Raises:
        FileNotFoundError: If the source directory does not exist.
    """
    # scripts/tools/sync_rsl_rl.py -> project root is two levels up
    project_root = script_path.parent.parent.parent
    source_dir = project_root / "rsl_rl"
    if not source_dir.is_dir():
        raise FileNotFoundError(
            f"Source directory not found: {source_dir}\n"
            "Please ensure the 'rsl_rl/' folder exists at the project root."
        )
    return source_dir


def resolve_target_dir(target_override: str | None) -> Path:
    """Resolve the target directory from CLI argument or environment variable.

    Args:
        target_override: Explicit target path from ``--target`` argument.

    Returns:
        Resolved target directory path.

    Raises:
        ValueError: If neither ``--target`` nor ``RSL_RL_PATH`` is set.
    """
    if target_override:
        return Path(target_override)

    rsl_rl_path = os.environ.get("RSL_RL_PATH")
    if not rsl_rl_path:
        raise ValueError(
            "Target path is not specified.\n"
            "Set the RSL_RL_PATH environment variable, or use --target <path>."
        )
    return Path(rsl_rl_path)


def _ignore_patterns(directory: str, filenames: list[str]) -> list[str]:
    """Return filenames to ignore during copy.

    Args:
        directory: Current directory being processed.
        filenames: List of filenames in the current directory.

    Returns:
        List of filenames to skip.
    """
    ignored = []
    for name in filenames:
        if name == "__pycache__":
            ignored.append(name)
        elif name.endswith(".pyc"):
            ignored.append(name)
        elif name.endswith(".pyo"):
            ignored.append(name)
    return ignored


def clear_directory(target_dir: Path) -> None:
    """Remove all contents inside a directory without deleting the directory itself.

    Args:
        target_dir: Directory to clear.
    """
    for item in target_dir.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()


def sync_rsl_rl(source_dir: Path, target_dir: Path, dry_run: bool = False) -> None:
    """Sync the rsl_rl source directory to the target path.

    If the target does not exist, it is created.
    If the target exists, all its contents are removed and replaced with a
    fresh copy from the source.

    Args:
        source_dir: Path to the local ``rsl_rl/`` source directory.
        target_dir: Path to sync to (value of ``RSL_RL_PATH``).
        dry_run: If True, only preview the operation.
    """
    print(f"Source : {source_dir}")
    print(f"Target : {target_dir}")

    if dry_run:
        print("\n[DRY RUN] No changes will be made.\n")

    if not target_dir.exists():
        print(f"\nTarget does not exist. Creating: {target_dir}")
        if not dry_run:
            target_dir.mkdir(parents=True, exist_ok=True)
            print("Directory created.")
    else:
        if not target_dir.is_dir():
            raise RuntimeError(
                f"Target path exists but is not a directory: {target_dir}"
            )
        print(f"\nTarget exists. Clearing all contents in: {target_dir}")
        if not dry_run:
            clear_directory(target_dir)
            print("Contents cleared.")

    print(f"Copying files from {source_dir} -> {target_dir} ...")
    if not dry_run:
        shutil.copytree(
            str(source_dir),
            str(target_dir),
            ignore=_ignore_patterns,
            dirs_exist_ok=True,
        )
    print("Sync complete.")


def main() -> None:
    """Entry point for the RSL RL sync script."""
    args = parse_args()

    try:
        script_path = Path(__file__).resolve()
        source_dir = Path(args.source).resolve() if args.source else resolve_source_dir(script_path)
        target_dir = resolve_target_dir(args.target)

        print("=" * 60)
        print("  RSL RL Sync Tool")
        print("=" * 60)

        sync_rsl_rl(source_dir, target_dir, dry_run=args.dry_run)

    except (ValueError, FileNotFoundError, RuntimeError) as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
