"""Allocate whole episodes; source Zarr is opened read-only and never copied."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import zarr
from modular_policy.common.episode_split import (
    DEFAULT_COUNTS, MT4_TASKS, SPLIT_NAMES, make_manifest,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zarr", default="data/rlbench/mt4_expert_200.zarr")
    parser.add_argument("--output", default="splits/mt4_seed42.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tasks", nargs="+", default=MT4_TASKS,
                        help="Task names in source block order")
    parser.add_argument("--episodes-per-task", type=int, default=50)
    parser.add_argument("--counts", type=int, nargs=5, default=DEFAULT_COUNTS,
                        metavar=("TRAIN", "VAL", "RISK", "CAL", "TEST"))
    args = parser.parse_args()
    root = zarr.open_group(args.zarr, mode="r")
    manifest = make_manifest(root, args.zarr, args.tasks, args.episodes_per_task,
                             args.counts, args.seed)
    output = Path(args.output)
    serialized = json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    if output.exists():
        if json.loads(output.read_text()) != manifest:
            raise FileExistsError(f"Refusing to replace a different split: {output}")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x") as f:
            f.write(serialized)
    print("task\t" + "\t".join(SPLIT_NAMES))
    for task in args.tasks:
        print(task + "\t" + "\t".join(str(len(manifest["by_task"][task][s])) for s in SPLIT_NAMES))
    print("TOTAL\t" + "\t".join(str(len(manifest["splits"][s])) for s in SPLIT_NAMES))
    print(f"Manifest: {output.resolve()}")


if __name__ == "__main__":
    main()
