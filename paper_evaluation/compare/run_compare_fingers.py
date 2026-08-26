"""Evaluation 2: SMPL-X 7-bone arm+finger chain (left_collar -> left_index3).

Same 6 algorithms, same 3 tiers, same methodology as evaluation 1
(paper_evaluation/compare_results.csv / compare_table.png, frozen/
untouched) -- just a longer 21-DOF chain and a different EE bone. This
script is the direct product of generalizing common.py/targets.py/
algos.py/legacy.py into a common.ModelSpec-driven pipeline (see those
modules' docstrings); evaluation 1's own run_compare.py is left as-is.

Output: paper_evaluation/compare_table_smplx_fingers.png/.csv
"""

import os
import time

import numpy as np

from . import algos, common, legacy, report
from . import targets as targets_mod

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_DIR = os.path.dirname(_HERE)

SPEC = common.FINGER_SPEC
TIERS = ["easy", "medium", "hard"]
CUSTOM_OBJECTIVE_WEIGHT = 0.25  # historical value, see evaluation 1's run_compare.py

CSV_PATH = os.path.join(_PKG_DIR, "compare_results_smplx_fingers.csv")
PNG_PATH = os.path.join(_PKG_DIR, "compare_table_smplx_fingers.png")


def run_native_algos():
    results = {}
    for tier in TIERS:
        bounds = targets_mod.tier_bounds_deg(tier, SPEC)
        threshold = targets_mod.TIER_THRESHOLD[tier]
        jax_solver = common.build_jax_ik_solver(
            bounds, threshold=threshold, model_file=SPEC.model_file, controlled_bones=SPEC.controlled_bones
        )
        fk_solver = jax_solver.fk_solver
        offsets, lengths = common.bone_offsets_and_lengths(fk_solver, SPEC.controlled_bones)
        ikpy_chain = common.build_ikpy_chain(bounds, offsets, lengths, SPEC.controlled_bones)
        rtb_robot = common.build_rtb_robot(bounds, offsets, lengths, SPEC.controlled_bones)

        tgs = targets_mod.generate_or_load(tier, verbose=True, spec=SPEC)

        for algo_name, solve_fn, obj in [
            ("jax_ik", algos.solve_jax_ik, jax_solver),
            ("ikpy", algos.solve_ikpy, ikpy_chain),
            ("roboticstoolbox", algos.solve_rtb, rtb_robot),
        ]:
            tg0 = tgs[0]
            pos0 = np.asarray(tg0["pos"], dtype=np.float32)
            R0 = np.asarray(tg0["R"], dtype=np.float32) if tier == "hard" else None
            for _ in range(2 if algo_name == "jax_ik" else 0):
                solve_fn(obj, tier, pos0, R0, spec=SPEC)

            times_ms, iters, succ = [], [], []
            t0 = time.time()
            for tg in tgs:
                pos = np.asarray(tg["pos"], dtype=np.float32)
                R = np.asarray(tg["R"], dtype=np.float32) if tier == "hard" else None
                ms, it, ang = solve_fn(obj, tier, pos, R, spec=SPEC)
                ok = algos.reverify_success(fk_solver, tier, ang, pos, R, spec=SPEC)
                times_ms.append(ms)
                iters.append(it)
                succ.append(ok)
            print(
                f"[{algo_name}/{tier}] {len(tgs)} solves in {time.time()-t0:.1f}s, "
                f"success={100*np.mean(succ):.1f}%"
            )
            results[(algo_name, tier)] = {"time_ms": times_ms, "iters": iters, "succ": succ}
    return results


def run_legacy_algos():
    results = {}
    for tier in TIERS:
        bounds = targets_mod.tier_bounds_deg(tier, SPEC)
        tgs = targets_mod.generate_or_load(tier, verbose=False, spec=SPEC)
        pos_list = [t["pos"] for t in tgs]
        custom = tier == "hard"

        for algo_name, solver_type in [("CCD", "ccd"), ("FABRIK", "fabrik")]:
            t0 = time.time()
            rows = legacy.run_ccd_or_fabrik(
                solver_type, tier, bounds, pos_list, custom_objective=custom,
                model_file=SPEC.model_file, controlled_bones=SPEC.controlled_bones,
            )
            times_ms = [r[1] * 1000.0 for r in rows]
            iters = [r[2] for r in rows]
            succ = [r[3] for r in rows]
            print(
                f"[{algo_name}/{tier}] {len(rows)} parsed rows in {time.time()-t0:.1f}s, "
                f"success={100*np.mean(succ) if succ else float('nan'):.1f}%"
            )
            results[(algo_name, tier)] = {"time_ms": times_ms, "iters": iters, "succ": succ}

        t0 = time.time()
        weight = CUSTOM_OBJECTIVE_WEIGHT if custom else 0.0
        rows = legacy.run_tensorflow(
            tier, bounds, pos_list, additional_objective_weight=weight,
            model_file=SPEC.model_file, controlled_bones=SPEC.controlled_bones,
        )
        times_ms = [r[1] * 1000.0 for r in rows]
        iters = [r[2] for r in rows]
        succ = [r[3] for r in rows]
        print(
            f"[TensorFlow/{tier}] {len(rows)} parsed rows in {time.time()-t0:.1f}s, "
            f"success={100*np.mean(succ) if succ else float('nan'):.1f}%"
        )
        results[("TensorFlow", tier)] = {"time_ms": times_ms, "iters": iters, "succ": succ}
    return results


def main():
    native = run_native_algos()
    legacy_res = run_legacy_algos()
    all_results = {}
    all_results.update(native)
    all_results.update(legacy_res)
    rows = report.build_rows(all_results)
    report.write_csv(rows, CSV_PATH)
    report.render_table_png(
        rows,
        PNG_PATH,
        "JAX-IK vs. ikpy / roboticstoolbox / CCD / FABRIK / TensorFlow -- SMPL-X arm+finger chain (7 bones, 21 DOF)\n"
        "Easy / Medium / Hard -- mean ± std over 100 verified-reachable targets/tier (CPU only)",
    )


if __name__ == "__main__":
    main()
