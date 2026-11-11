#!/usr/bin/env python3
"""CLI entry point for independent one-step-deviation oracle collection."""
import pathlib
import sys

ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from modular_policy.workspace.collect_one_step_oracle import collect_one_step_oracle


if __name__ == '__main__':
    collect_one_step_oracle()
