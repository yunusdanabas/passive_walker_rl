#!/usr/bin/env python3
"""
Experiments Migration Tool

Move legacy experiment outputs into the unified structure.
Supports dry-run and optional deletion of empty legacy dirs.
"""

from __future__ import annotations
import argparse
import shutil
from pathlib import Path
from passive_walker.config.paths import (
    EXPERIMENTS_ROOT, PLOTS_DIR, METRICS_DIR, REPORTS_DIR, PPO_PLOTS_DIR
)


MAPPINGS = [
    (EXPERIMENTS_ROOT / "evaluation", METRICS_DIR),
    (EXPERIMENTS_ROOT / "evaluation_results", METRICS_DIR),
    (EXPERIMENTS_ROOT / "outputs" / "plots", PLOTS_DIR),
    (EXPERIMENTS_ROOT / "outputs" / "reports", REPORTS_DIR),
    (EXPERIMENTS_ROOT / "outputs" / "metrics", METRICS_DIR),
    (EXPERIMENTS_ROOT / "results" / "figures", PLOTS_DIR),
    (EXPERIMENTS_ROOT / "results" / "data", METRICS_DIR),
    (EXPERIMENTS_ROOT / "ppo_plots", PPO_PLOTS_DIR),
]


def migrate_path(src: Path, dst: Path, dry_run: bool = True, move: bool = False):
    if not src.exists():
        return []
    moved = []
    for item in src.rglob('*'):
        if item.is_file():
            rel = item.relative_to(src)
            target = dst / rel
            if dry_run:
                moved.append((item, target))
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                if move:
                    shutil.move(str(item), str(target))
                else:
                    shutil.copy2(item, target)
                moved.append((item, target))
    return moved


def main():
    parser = argparse.ArgumentParser(description="Migrate experiments outputs to unified structure")
    parser.add_argument("--dry-run", action="store_true", help="Show actions without copying/moving")
    parser.add_argument("--move", action="store_true", help="Move files instead of copying (cleans legacy files)")
    parser.add_argument("--delete-empty-dirs", action="store_true", help="Delete empty legacy dirs after migration")
    args = parser.parse_args()

    summary = []
    for src, dst in MAPPINGS:
        moved = migrate_path(src, dst, dry_run=args.dry_run, move=args.move)
        if moved:
            mode = 'moving' if args.move and not args.dry_run else 'copying'
            if args.dry_run:
                mode = f"DRY {mode}"
            print(f"{mode.capitalize()} {len(moved)} files: {src} -> {dst}")
            summary.extend(moved)
        else:
            print(f"No files to migrate in: {src}")

    print(f"\nTotal files {'to migrate' if args.dry_run else 'migrated'}: {len(summary)}")
    if args.delete_empty_dirs and not args.dry_run:
        for src, _ in MAPPINGS:
            if src.exists():
                # Remove empty subdirectories
                for p in sorted([p for p in src.rglob('*') if p.is_dir()], reverse=True):
                    try:
                        p.rmdir()
                    except OSError:
                        pass
                try:
                    src.rmdir()
                except OSError:
                    pass
        print("Attempted to remove empty legacy directories.")


if __name__ == "__main__":
    main()


