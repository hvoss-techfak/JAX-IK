"""Runs both new model comparisons (evaluation 2: SMPL-X arm+finger chain;
evaluation 3: UR5) end-to-end in one invocation, using .venv_compare's
python. Evaluation 1 (4-bone SMPL-X arm, paper_evaluation/compare_results.csv
/ compare_table.png) is already computed and frozen -- this script does not
re-run it.

Usage:
    paper_evaluation/.venv_compare/bin/python -m paper_evaluation.compare.run_all_models

Outputs:
    paper_evaluation/compare_results_smplx_fingers.csv
    paper_evaluation/compare_table_smplx_fingers.png
    paper_evaluation/compare_results_ur5.csv
    paper_evaluation/compare_table_ur5.png
"""

from . import run_compare_fingers, run_compare_ur5


def main():
    print("=== Evaluation 2: SMPL-X arm+finger chain ===")
    run_compare_fingers.main()
    print("\n=== Evaluation 3: UR5 ===")
    run_compare_ur5.main()


if __name__ == "__main__":
    main()
