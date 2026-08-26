"""Evaluation 3: UR5 (models/UR5.urdf) -- jax_ik vs. ikpy vs. roboticstoolbox.

CCD/FABRIK/TensorFlow (ablation.py/ik_tensorflow.py) were attempted via a
helper.py shim (dispatching their hardcoded load_skeleton_from_gltf import
to jax_ik's own load_skeleton_from_urdf) and that part worked -- but both
scripts' *own* FKSolver/objective code independently hardcodes the SMPL-X
finger bone "{hand}_index3_look" as their effector target
(ablation.py's FKSolver.__init__ raises ValueError immediately if that bone
doesn't exist, unconditionally, regardless of controlled_bones; ik_tensorflow.py
builds its objective directly against that bone name too). UR5 has no such
bone and never will, and fixing this needs editing ablation.py/
ik_tensorflow.py themselves, which is off-limits (frozen). So this
evaluation is jax_ik + ikpy + roboticstoolbox only -- a real architectural
blocker in the frozen scripts, not first-try friction.

Output: paper_evaluation/compare_table_ur5.png/.csv
"""

import os
import time

import numpy as np

from . import algos, report
from . import algos_ur5
from . import ur5_common as ur5
from . import targets_ur5 as targets_mod

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_DIR = os.path.dirname(_HERE)

TIERS = ["easy", "medium", "hard"]
CSV_PATH = os.path.join(_PKG_DIR, "compare_results_ur5.csv")
PNG_PATH = os.path.join(_PKG_DIR, "compare_table_ur5.png")


def run_all():
    results = {}
    for tier in TIERS:
        bounds6 = None if tier == "easy" else targets_mod.tier_bounds_deg(tier)
        # ikpy/roboticstoolbox always need an explicit numeric bound (no
        # "auto-derive" concept for them) -- easy tier's "None" for jax_ik
        # (auto-derived from the URDF, clamped to +-180) is numerically
        # +-180 per joint, so use that explicitly for the other two.
        bounds6_explicit = [(-180.0, 180.0)] * 6 if tier == "easy" else bounds6
        threshold = targets_mod.TIER_THRESHOLD[tier]

        jax_solver = ur5.build_jax_ik_solver(bounds6, threshold=threshold)
        fk_solver = jax_solver.fk_solver
        ikpy_chain = ur5.build_ikpy_chain(bounds6_explicit)
        rtb_robot = ur5.build_rtb_robot(bounds6_explicit)

        tgs = targets_mod.generate_or_load(tier, verbose=True)

        # jax_ik: reuse algos.solve_jax_ik/reverify_success via UR5_SPEC.
        tg0 = tgs[0]
        pos0 = np.asarray(tg0["pos"], dtype=np.float32)
        R0 = np.asarray(tg0["R"], dtype=np.float32) if tier == "hard" else None
        for _ in range(2):
            algos.solve_jax_ik(jax_solver, tier, pos0, R0, spec=ur5.UR5_SPEC)

        times_ms, iters, succ = [], [], []
        t0 = time.time()
        for tg in tgs:
            pos = np.asarray(tg["pos"], dtype=np.float32)
            R = np.asarray(tg["R"], dtype=np.float32) if tier == "hard" else None
            ms, it, ang18 = algos.solve_jax_ik(jax_solver, tier, pos, R, spec=ur5.UR5_SPEC)
            ok = algos.reverify_success(fk_solver, tier, ang18, pos, R, spec=ur5.UR5_SPEC)
            times_ms.append(ms)
            iters.append(it)
            succ.append(ok)
        print(f"[jax_ik/{tier}] {len(tgs)} solves in {time.time()-t0:.1f}s, success={100*np.mean(succ):.1f}%")
        results[("jax_ik", tier)] = {"time_ms": times_ms, "iters": iters, "succ": succ}

        for algo_name, solve_fn, obj in [
            ("ikpy", algos_ur5.solve_ikpy, ikpy_chain),
            ("roboticstoolbox", algos_ur5.solve_rtb, rtb_robot),
        ]:
            times_ms, iters, succ = [], [], []
            t0 = time.time()
            for tg in tgs:
                pos = np.asarray(tg["pos"], dtype=np.float32)
                R = np.asarray(tg["R"], dtype=np.float32) if tier == "hard" else None
                ms, it, q6 = solve_fn(obj, tier, pos, R)
                angles18 = ur5.angle18_from_q6(q6)
                ok = algos.reverify_success(fk_solver, tier, angles18, pos, R, spec=ur5.UR5_SPEC)
                times_ms.append(ms)
                iters.append(it)
                succ.append(ok)
            print(f"[{algo_name}/{tier}] {len(tgs)} solves in {time.time()-t0:.1f}s, success={100*np.mean(succ):.1f}%")
            results[(algo_name, tier)] = {"time_ms": times_ms, "iters": iters, "succ": succ}
    return results


def main():
    results = run_all()
    rows = report.build_rows(results)
    report.write_csv(rows, CSV_PATH)
    report.render_table_png(
        rows,
        PNG_PATH,
        "JAX-IK vs. ikpy / roboticstoolbox -- UR5 (models/UR5.urdf, 6-DOF)\n"
        "Easy (+-180) / Medium / Hard (+-90, multi-objective) -- mean ± std, 100 verified-reachable targets/tier (CPU only)\n"
        "CCD/FABRIK/TensorFlow dropped: hardcode SMPL-X's \"index3_look\" bone, incompatible with UR5 -- see module docstring",
    )


if __name__ == "__main__":
    main()
