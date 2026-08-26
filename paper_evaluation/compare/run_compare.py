"""Orchestrates the full jax-ik vs. external-package IK comparison.

Algorithms: jax_ik (current, v0.4), ikpy, roboticstoolbox, CCD, FABRIK,
TensorFlow (the old jax-ik variant reimplemented in TF, from
ik_tensorflow.py) -- IPOPT was attempted but dropped, see README note in
this directory / the final report: cyipopt has no prebuilt wheel for this
platform and building from source needs a system IPOPT + pkg-config
install that isn't available here.

Tiers: easy (reach only, loose bounds), medium (reach only, strict
anatomical bounds), hard (position+orientation, strict bounds). 100
verified-reachable targets per tier (see targets.py).

Outputs: paper_evaluation/compare_results.csv and
paper_evaluation/compare_table.png.
"""

import csv
import os
import time

import numpy as np

from . import algos, common, legacy, targets as targets_mod

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_DIR = os.path.dirname(_HERE)

TIERS = ["easy", "medium", "hard"]
CUSTOM_OBJECTIVE_WEIGHT = 0.25  # historical value from timing.py's "with custom objective" runs

CSV_PATH = os.path.join(_PKG_DIR, "compare_results.csv")
PNG_PATH = os.path.join(_PKG_DIR, "compare_table.png")


def _stats(values):
    arr = np.asarray(values, dtype=np.float64)
    return float(np.mean(arr)), float(np.std(arr))


def run_native_algos():
    """jax_ik, ikpy, roboticstoolbox -- per-target, uniform re-verified success."""
    results = {}  # (algo, tier) -> dict of lists
    for tier in TIERS:
        bounds = targets_mod.TIER_BOUNDS[tier]()
        threshold = targets_mod.TIER_THRESHOLD[tier]
        jax_solver = common.build_jax_ik_solver(bounds, threshold=threshold)
        fk_solver = jax_solver.fk_solver
        offsets, lengths = common.bone_offsets_and_lengths(fk_solver)
        ikpy_chain = common.build_ikpy_chain(bounds, offsets, lengths)
        rtb_robot = common.build_rtb_robot(bounds, offsets, lengths)

        tgs = targets_mod.generate_or_load(tier, verbose=False)

        for algo_name, solve_fn, obj in [
            ("jax_ik", algos.solve_jax_ik, jax_solver),
            ("ikpy", algos.solve_ikpy, ikpy_chain),
            ("roboticstoolbox", algos.solve_rtb, rtb_robot),
        ]:
            # Warm up (JIT compile for jax_ik; negligible for the others).
            tg0 = tgs[0]
            pos0 = np.asarray(tg0["pos"], dtype=np.float32)
            R0 = np.asarray(tg0["R"], dtype=np.float32) if tier == "hard" else None
            for _ in range(2 if algo_name == "jax_ik" else 0):
                solve_fn(obj, tier, pos0, R0)

            times_ms, iters, succ = [], [], []
            t0 = time.time()
            for tg in tgs:
                pos = np.asarray(tg["pos"], dtype=np.float32)
                R = np.asarray(tg["R"], dtype=np.float32) if tier == "hard" else None
                ms, it, ang = solve_fn(obj, tier, pos, R)
                ok = algos.reverify_success(fk_solver, tier, ang, pos, R)
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
        bounds = targets_mod.TIER_BOUNDS[tier]()
        tgs = targets_mod.generate_or_load(tier, verbose=False)
        pos_list = [t["pos"] for t in tgs]
        custom = tier == "hard"

        for algo_name, solver_type in [("CCD", "ccd"), ("FABRIK", "fabrik")]:
            t0 = time.time()
            rows = legacy.run_ccd_or_fabrik(solver_type, tier, bounds, pos_list, custom_objective=custom)
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
        rows = legacy.run_tensorflow(tier, bounds, pos_list, additional_objective_weight=weight)
        times_ms = [r[1] * 1000.0 for r in rows]
        iters = [r[2] for r in rows]
        succ = [r[3] for r in rows]
        print(
            f"[TensorFlow/{tier}] {len(rows)} parsed rows in {time.time()-t0:.1f}s, "
            f"success={100*np.mean(succ) if succ else float('nan'):.1f}%"
        )
        results[("TensorFlow", tier)] = {"time_ms": times_ms, "iters": iters, "succ": succ}
    return results


def write_csv(all_results):
    rows = []
    for (algo, tier), d in all_results.items():
        t_mean, t_std = _stats(d["time_ms"])
        it_mean, it_std = _stats(d["iters"])
        tpi = [t / i for t, i in zip(d["time_ms"], d["iters"]) if i > 0]
        tpi_mean, tpi_std = _stats(tpi) if tpi else (float("nan"), float("nan"))
        succ_rate = 100.0 * float(np.mean(d["succ"])) if d["succ"] else float("nan")
        rows.append(
            {
                "algorithm": algo,
                "tier": tier,
                "solving_time_ms_mean": t_mean,
                "solving_time_ms_std": t_std,
                "iterations_mean": it_mean,
                "iterations_std": it_std,
                "time_per_iter_ms_mean": tpi_mean,
                "time_per_iter_ms_std": tpi_std,
                "success_rate_pct": succ_rate,
            }
        )

    tier_order = {"easy": 0, "medium": 1, "hard": 2}
    rows.sort(key=lambda r: (tier_order[r["tier"]], r["algorithm"] != "jax_ik", r["algorithm"]))

    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {CSV_PATH}")
    return rows


def render_table_png(rows):
    import matplotlib.pyplot as plt
    from matplotlib import font_manager  # noqa: F401

    tier_labels = {
        "easy": "Easy\n(reach only)",
        "medium": "Medium\n(reach, strict limits)",
        "hard": "Hard\n(multi-objective, strict limits)",
    }

    col_labels = ["Algorithm", "Tier", "Solving Time (ms)", "Iterations", "Time / Iteration (ms)", "Success Rate (%)"]
    cell_text = []
    row_is_jaxik = []
    row_is_tier_start = []
    prev_tier = None
    for r in rows:
        is_start = r["tier"] != prev_tier
        prev_tier = r["tier"]
        row_is_tier_start.append(is_start)
        row_is_jaxik.append(r["algorithm"] == "jax_ik")
        cell_text.append(
            [
                r["algorithm"],
                tier_labels[r["tier"]] if is_start else "",
                f"{r['solving_time_ms_mean']:.3f} ± {r['solving_time_ms_std']:.3f}",
                f"{r['iterations_mean']:.1f} ± {r['iterations_std']:.1f}",
                f"{r['time_per_iter_ms_mean']:.4f} ± {r['time_per_iter_ms_std']:.4f}",
                f"{r['success_rate_pct']:.1f}",
            ]
        )

    n_rows = len(cell_text)
    fig_h = 0.55 + 0.5 * (n_rows + 1)
    fig, ax = plt.subplots(figsize=(12.5, fig_h), dpi=220)
    ax.axis("off")

    col_widths = [0.14, 0.22, 0.19, 0.15, 0.19, 0.14]
    title_frac = 0.62 / fig_h  # reserve just enough for the (2-line) title
    tbl = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellLoc="center",
        colWidths=col_widths,
        bbox=[0.0, 0.0, 1.0, 1.0 - title_frac],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)

    header_color = "#2b2b40"
    jaxik_color = "#e7f0ff"
    tier_band_colors = ["#ffffff", "#f4f6fa"]
    tier_idx = -1

    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor("#c9ccd6")
        if row == 0:
            cell.set_text_props(weight="bold", color="white")
            cell.set_facecolor(header_color)
            continue
        data_row = row - 1
        if row_is_tier_start[data_row]:
            tier_idx += 1
        bg = tier_band_colors[tier_idx % 2]
        if row_is_jaxik[data_row]:
            bg = jaxik_color
            cell.set_text_props(weight="bold")
        cell.set_facecolor(bg)

    ax.set_title(
        "JAX-IK vs. ikpy / roboticstoolbox / CCD / FABRIK / TensorFlow"
        " (IPOPT attempted, dropped -- see report)\n"
        "Easy / Medium / Hard -- mean ± std over 100 verified-reachable targets/tier (CPU only)",
        fontsize=11.5,
        fontweight="bold",
        pad=8,
    )
    fig.tight_layout()
    fig.savefig(PNG_PATH, dpi=220, bbox_inches="tight")
    print(f"Wrote {PNG_PATH}")


def main():
    native = run_native_algos()
    legacy_res = run_legacy_algos()
    all_results = {}
    all_results.update(native)
    all_results.update(legacy_res)
    rows = write_csv(all_results)
    render_table_png(rows)


if __name__ == "__main__":
    main()
