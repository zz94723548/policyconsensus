"""Command-line entry point for RLBench branch-state validation."""
if __name__ == '__main__':
    import os
    import pathlib
    import sys

    root = pathlib.Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root))
    os.chdir(root)

from modular_policy.workspace.validate_branch_state import validate_branch_state


if __name__ == '__main__':
    validate_branch_state()
