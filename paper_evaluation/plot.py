import numpy as np

import plotly.graph_objects as go
from scipy import stats


def plot_success_rate(all_results, output_filename):
    """
    Create a bar plot for the success rate across all implementations.
    Bars show the mean success rate with standard deviation error bars.
    Sorted in ascending order of mean success rate.
    """
    stats = {}
    for name, runs in all_results.items():
        sr_list = []
        for run in runs:
            for measurement in run:
                # Use index 4 for success (based on tuple: (iteration, time, time_per_iteration, steps, success))
                if measurement[4] is not None:
                    sr_list.append(measurement[4])
        if sr_list:
            mean_sr = np.mean(sr_list)
            std_sr = np.std(sr_list)
            stats[name] = (mean_sr, std_sr)

    sorted_stats = sorted(stats.items(), key=lambda x: x[1][0])
    alg_names = [x[0] for x in sorted_stats]
    mean_values = [x[1][0] for x in sorted_stats]
    std_values = [x[1][1] for x in sorted_stats]
    x_pos = list(range(len(alg_names)))

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=x_pos,
        y=mean_values,
        error_y=dict(
            type='data',
            array=std_values,
            visible=True
        ),
        marker_color='lightskyblue'
    ))

    fig.update_layout(
        title="Success Rate (Combined Conditions)",
        xaxis=dict(
            title="Algorithm",
            tickmode='array',
            tickvals=x_pos,
            ticktext=alg_names
        ),
        yaxis=dict(
            title="Mean Success Rate"
        ),
        margin=dict(l=20, r=20, t=40, b=20)
    )

    fig.write_image(output_filename)
    print(f"Plot saved as {output_filename}")


def plot_results(all_results, condition, output_filename):
    """
    Create a bar plot (with error bars for standard deviation) for the algorithms matching
    a given condition (e.g. "only target" or "custom objective"). The bars are sorted in ascending order of mean time.
    Uses the per-iteration time (measurement index 2) for the performance metric.
    """
    # Filter out results by checking if the condition string is in the algorithm name.
    filtered = {name: runs for name, runs in all_results.items() if condition in name}
    stats = {}
    for name, runs in filtered.items():
        times = []
        for run in runs:
            for measurements in run:
                for measurement in measurements:
                    if measurement[1] is not None:
                        times.append(measurement[1])
        if times:
            mean_time = np.mean(times)
            std_time = np.std(times)
            stats[name] = (mean_time, std_time)

    sorted_stats = sorted(stats.items(), key=lambda x: x[1][0])
    alg_names = [x[0].replace(" only target","").replace(" with custom objective","") for x in sorted_stats]
    mean_times = [x[1][0] for x in sorted_stats]
    std_times = [x[1][1] for x in sorted_stats]
    x_pos = list(range(len(alg_names)))

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=x_pos,
        y=mean_times,
        error_y=dict(
            type='data',
            array=std_times,
            visible=True
        ),
        marker_color='lightskyblue'
    ))
    fig.add_hline(y=0.033333333, line_width=1, line_dash="dash", line_color="black")

    fig.update_layout(
        xaxis=dict(
            tickmode='array',
            tickvals=x_pos,
            ticktext=alg_names
        ),
        yaxis=dict(
            title="Mean Time (seconds)"
        ),
        margin=dict(l=20, r=20, t=40, b=20),
    )

    fig.write_image(output_filename)
    print(f"Plot saved as {output_filename}")



def plot_cpu_gpu_comparison(all_results, output_filename):
    """
    Create a bar plot comparing CPU vs GPU performance for TensorFlow and JAX,
    based on average time per iteration (measurement[2]).
    """
    implementations = ['tensorflow', 'jax']
    devices = ['cpu', 'gpu']

    stats = {}
    for impl in implementations:
        for device in devices:
            times = []
            for name, runs in all_results.items():
                if impl in name and device in name:
                    for run in runs:
                        for measurements in run:
                            for measurement in measurements:
                                if measurement[2] is not None:
                                    times.append(measurement[2])
            if times:
                mean_time = np.mean(times)
                std_time = np.std(times)
                stats[f"{impl.upper()}-{device.upper()}"] = (mean_time, std_time)

    sorted_stats = sorted(stats.items(), key=lambda x: x[1][0])
    labels = [x[0] for x in sorted_stats]
    mean_times = [x[1][0] for x in sorted_stats]
    std_times = [x[1][1] for x in sorted_stats]
    x_pos = list(range(len(labels)))

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=x_pos,
        y=mean_times,
        error_y=dict(
            type='data',
            array=std_times,
            visible=True
        ),
    ))

    fig.update_layout(
        title="CPU vs GPU Performance (Avg. Time per Iteration)",
        xaxis=dict(
            tickmode='array',
            tickvals=x_pos,
            ticktext=labels
        ),
        yaxis=dict(
            title="Mean Time per Iteration (seconds)"
        ),
        margin=dict(l=20, r=20, t=40, b=20),
        yaxis_range=[0, 0.01]
    )

    fig.write_image(output_filename)
    print(f"Comparison plot saved as {output_filename}")

def create_gpu_cpu_latex_table(all_results):
    """
    Create a LaTeX formatted table summarizing CPU and GPU performance comparisons
    for TensorFlow and JAX implementations based on average time per iteration.
    """
    implementations = ['tensorflow', 'jax']
    devices = ['cpu', 'gpu']

    table = "\\begin{tabular}{lcc}\n"
    table += "\\hline\\hline\n"
    table += "Implementation & Device & Avg. Time per Iteration (s) \\ \\hline\n"

    for impl in implementations:
        for device in devices:
            times = []
            for name, runs in all_results.items():
                if impl in name and device in name:
                    for run in runs:
                        for measurements in run:
                            for measurement in measurements:
                                if measurement[2] is not None:
                                    times.append(measurement[2])
            if times:
                mean_time = np.mean(times)
                std_time = np.std(times)
                table += f"{impl.capitalize()} & {device.upper()} & {mean_time:.4f} $\\pm$ {std_time:.4f} \\ \\n"

    table += "\\hline\\hline\n"
    table += "\\end{tabular}"

    print(table)

def plot_time_per_iteration(all_results, output_filename, significance_threshold=0.05):
    """
    Create an interactive Plotly bar chart of mean time per iteration with error bars,
    and add significance annotations (lines and stars) comparing the best-performing algorithm
    to each of the others.
    """
    # Compute per-iteration times and store raw values for significance testing.
    raw_data = {}
    stats_dict = {}
    for name, runs in all_results.items():
        # Normalize algorithm names.
        if "jax" in name.lower():
            name = "Jax"
        elif "tensorflow" in name.lower():
            name = "TensorFlow"
        elif "fabrik" in name.lower():
            name = "FABRIK"
        elif "ccd" in name.lower():
            name = "CCD"

        tpi_list = []
        for run in runs:
            for measurements in run:
                for measurement in measurements:
                    iteration_num, time_taken,time_per_iteration, steps, success = measurement
                    tpi_list.append(time_per_iteration)
        if tpi_list:
            # If the algorithm already exists (e.g. from multiple runs), extend the list.
            raw_data.setdefault(name, []).extend(tpi_list)

    # Compute mean and standard deviation for each algorithm.
    for name, values in raw_data.items():
        stats_dict[name] = (np.mean(values), np.std(values))

    # Sort algorithms in ascending order of mean time.
    sorted_stats = sorted(stats_dict.items(), key=lambda x: x[1][0])
    alg_names = [x[0] for x in sorted_stats]
    mean_values = [x[1][0] for x in sorted_stats]
    std_values = [x[1][1] for x in sorted_stats]

    # Use numeric x positions so that we can easily place shapes and annotations.
    x_pos = np.arange(len(alg_names))

    # Create Plotly bar chart with error bars.
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=x_pos,
        y=mean_values,
        error_y=dict(
            type='data',
            array=std_values,
            visible=True
        ),
        marker_color='lightskyblue'
    ))

    # Compare the best-performing algorithm (first bar) with each other algorithm.
    reference = alg_names[0]
    ref_values = raw_data[reference]
    shapes = []
    annotations = []

    # Choose an offset for the significance lines based on the range of mean values.
    y_range = max(mean_values) - min(mean_values)
    offset = 0.05 * y_range if y_range != 0 else 0.1

    for i in range(1, len(alg_names)):
        current_alg = alg_names[i]
        cur_values = raw_data[current_alg]

        # Perform a two-sample t-test (Welch's t-test).
        t_stat, p_val = stats.ttest_ind(ref_values, cur_values, equal_var=False)

        # Determine significance stars.
        if p_val < 0.001:
            stars = '***'
        elif p_val < 0.01:
            stars = '**'
        elif p_val < significance_threshold:
            stars = '*'
        else:
            stars = 'ns'

        # Determine the vertical position for the significance line.
        y_ref = mean_values[0] + std_values[0]
        y_cur = mean_values[i] + std_values[i]
        # Add a stacking offset to avoid overlap if you have several comparisons.
        y_line = max(y_ref, y_cur) + offset + (i - 1) * offset

        # x positions: reference bar is at x=0, current bar is at x=i.
        x0 = 0
        x1 = i

        # Draw horizontal line.
        shapes.append(dict(
            type="line",
            xref="x",
            yref="y",
            x0=x0,
            x1=x1,
            y0=y_line,
            y1=y_line,
            line=dict(color="black", width=1)
        ))
        # Draw vertical ticks at each end.
        shapes.append(dict(
            type="line",
            xref="x",
            yref="y",
            x0=x0,
            x1=x0,
            y0=y_line,
            y1=y_line - offset / 2,
            line=dict(color="black", width=1)
        ))
        shapes.append(dict(
            type="line",
            xref="x",
            yref="y",
            x0=x1,
            x1=x1,
            y0=y_line,
            y1=y_line - offset / 2,
            line=dict(color="black", width=1)
        ))
        # Place the significance stars in the middle.
        annotations.append(dict(
            x=(x0 + x1) / 2,
            y=y_line + offset / 2,
            text=stars,
            showarrow=False,
            font=dict(color="black", size=12)
        ))

    # Update layout with axis labels, custom tick labels, shapes, and annotations.
    fig.update_layout(
        xaxis_title="Algorithm",
        yaxis_title="Mean Time per Iteration (seconds)",
        xaxis=dict(
            tickmode='array',
            tickvals=x_pos,
            ticktext=alg_names
        ),
        shapes=shapes,
        annotations=annotations,
        margin=dict(l=20, r=20, t=20, b=20), # Adjust these values as needed
        yaxis_range = [0, 0.0065]
    )

    # Save the figure as an image.
    fig.write_image(output_filename)
    print(f"Plot saved as {output_filename}")


def print_latex_table(all_results):
    """
    Print a LaTeX table spanning two columns with columns:
    Algorithm name, Custom Objective (Yes/No), Solving Time (mean ± std),
    Iterations (mean ± std), Time per Iteration (mean ± std), and Success Rate (mean ± std)
    """
    header = r"""\begin{table*}[ht]
\centering
\begin{tabular}{l c c c c c}
\hline
Algorithm & Custom Objective & Solving Time (ms) & Iterations & Time per Iteration (ms) & Success Rate (%) \\
\hline"""
    print(header)

    for alg, runs in all_results.items():
        # Flatten the list of runs (each run is a list of tuples)
        flat_data = [item for run in runs for item in run]
        flat_data = [item for run in flat_data for item in run]
        if not flat_data:
            continue

        # Extract data from the flattened list
        # Tuple format: (iteration_num, time_taken, time_per_iteration, steps, success)
        times = np.array([x[1] for x in flat_data])

        iterations = np.array([x[3] for x in flat_data])
        tpis = times/iterations
        successes = np.array([x[4] for x in flat_data])

        mean_time = np.mean(times)
        std_time = np.std(times)
        mean_iter = np.mean(iterations)
        std_iter = np.std(iterations)
        mean_tpi = np.mean(tpis)
        std_tpi = np.std(tpis)
        mean_success = np.mean(successes)
        std_success = np.std(successes)

        # Determine if custom objective is used based on the algorithm name.
        custom_obj = "Yes" if "custom objective" in alg.lower() else "No"
        row = f"{alg} & {custom_obj} & {mean_time*1000:.2f} $\\pm$ {std_time*1000:.2f} & {mean_iter:.2f} $\\pm$ {std_iter:.2f} & {mean_tpi*1000:.2f} $\\pm$ {std_tpi*1000:.2f} & {mean_success*100:.2f} $\\pm$ {std_success*100:.2f} \\\\"
        print(row)

    footer = r"""\hline
\end{tabular}
\caption{Performance metrics across algorithms.}
\label{tab:performance_metrics}
\end{table*}"""
    print(footer)
