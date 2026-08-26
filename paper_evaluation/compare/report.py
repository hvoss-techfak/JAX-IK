"""Shared CSV + PNG table writer for the model-comparison evaluations.

Factored out of the (frozen, untouched) evaluation-1 run_compare.py so
evaluation 2 (finger chain) and evaluation 3 (UR5) can reuse the exact same
output format without duplicating it. Table layout mirrors run_compare.py's
render_table_png, including the tier-column width/short-label fix (avoid
the earlier column-overflow bug: short two-line tier labels + explicit
column widths, not auto-sized).
"""

import csv

import numpy as np


def stats(values):
    arr = np.asarray(values, dtype=np.float64)
    return float(np.mean(arr)), float(np.std(arr))


def build_rows(all_results, jaxik_name="jax_ik", tier_order=("easy", "medium", "hard")):
    order = {t: i for i, t in enumerate(tier_order)}
    rows = []
    for (algo, tier), d in all_results.items():
        t_mean, t_std = stats(d["time_ms"])
        it_mean, it_std = stats(d["iters"])
        tpi = [t / i for t, i in zip(d["time_ms"], d["iters"]) if i > 0]
        tpi_mean, tpi_std = stats(tpi) if tpi else (float("nan"), float("nan"))
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
    rows.sort(key=lambda r: (order[r["tier"]], r["algorithm"] != jaxik_name, r["algorithm"]))
    return rows


def write_csv(rows, csv_path):
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {csv_path}")


TIER_LABELS = {
    "easy": "Easy\n(reach only)",
    "medium": "Medium\n(reach, strict limits)",
    "hard": "Hard\n(multi-objective, strict limits)",
}


def render_table_png(rows, png_path, title, jaxik_name="jax_ik", tier_labels=None):
    import matplotlib.pyplot as plt

    tier_labels = tier_labels or TIER_LABELS
    col_labels = ["Algorithm", "Tier", "Solving Time (ms)", "Iterations", "Time / Iteration (ms)", "Success Rate (%)"]
    cell_text = []
    row_is_jaxik = []
    row_is_tier_start = []
    prev_tier = None
    for r in rows:
        is_start = r["tier"] != prev_tier
        prev_tier = r["tier"]
        row_is_tier_start.append(is_start)
        row_is_jaxik.append(r["algorithm"] == jaxik_name)
        cell_text.append(
            [
                r["algorithm"],
                tier_labels.get(r["tier"], r["tier"]) if is_start else "",
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
    title_frac = min(0.9, (0.28 * (title.count("\n") + 1) + 0.34) / fig_h)
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

    ax.set_title(title, fontsize=11.5, fontweight="bold", pad=8)
    fig.tight_layout()
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    print(f"Wrote {png_path}")
