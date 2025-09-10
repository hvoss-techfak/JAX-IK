import json
import os
import re
import subprocess
from concurrent.futures.process import ProcessPoolExecutor

import numpy as np
from plot import (
    create_gpu_cpu_latex_table,
    plot_cpu_gpu_comparison,
    plot_results,
    plot_success_rate,
    plot_time_per_iteration,
    print_latex_table,
)
from sklearn.cluster import KMeans
from tqdm import tqdm

SUCCESSFUL_TARGETS_FILE = "successful_targets.json"
NUM_PROCESSES = 24  # Adjust based on system capabilities
BATCH_SIZE = 400  # Number of targets


def run_script_full(script_name, args, num_runs=5):
    """
    Run the given script (a Python file) num_runs times with the specified arguments.
    Returns a list where each element is a list of tuples containing
    (iteration_number, time_taken, steps, success) for each iteration found in the output.
    """
    all_runs_data = []

    env = os.environ.copy()
    # env["JAX_PLATFORMS"] = "cpu"
    for run in tqdm(range(1, num_runs + 1)):
        cmd = ["python", script_name] + args
        result = subprocess.run(cmd, capture_output=True, text=True, env=env)
        output = result.stdout + "\n" + result.stderr
        print(output)

        run_data = []
        # Process each line looking for iteration data.
        for line in output.splitlines():
            # Expected line format:
            # "Time for iteration 22: 0.5242 seconds. Steps: 1000. Success: False"
            m = re.search(
                r"Time for iteration\s+(\d+):\s+([\d\.]+)\s+seconds\. Steps:\s+(\d+)\. Success:\s+(True|False)",
                line,
            )
            if m:
                iteration_num = int(m.group(1))
                time_taken = float(m.group(2))
                time_per_iteration = time_taken / iteration_num

                steps = int(m.group(3))
                success = 1 if m.group(4) == "True" else 0
                print(
                    f"Iteration {iteration_num}: {time_taken:.8f} seconds, {steps} steps, success={success}"
                )
                run_data.append(
                    (iteration_num, time_taken, time_per_iteration, steps, success)
                )
            # print average data
        time_taken = np.median([o[1] for o in run_data])
        avg_time_per_iteration = np.median([o[2] for o in run_data])
        avg_iterations = np.median([o[3] for o in run_data])
        success_rate = np.median([o[4] for o in run_data])
        print(f"Average time taken: {time_taken:.8f} seconds")
        print(f"Average time per iteration: {avg_time_per_iteration:.8f} seconds")
        print(f"Average total iterations: {avg_iterations:.8f}")
        print(f"Average success rate: {success_rate:.8f}")

        if run_data:
            all_runs_data.append(run_data)

    return all_runs_data


def run_script_first(script_name, target_list):
    """
    Runs the script with a batch of targets and captures which ones were solved.
    """
    env = os.environ.copy()
    # env["JAX_PLATFORMS"] = "cpu"

    # Convert the target list to a JSON string
    targets_str = json.dumps(target_list)
    args = ["--target_points", targets_str]

    cmd = ["python", script_name] + args
    # print(cmd)
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)

    output = result.stdout + "\n" + result.stderr
    # print(output)

    # Extract solved targets from the output
    solved_targets = []
    pattern = re.compile(r"Found solution for target\s+\[([^\]]+)\]")

    for match in pattern.finditer(output):
        target_str = match.group(1)
        target = list(map(float, target_str.split()))
        solved_targets.append(target)

    return solved_targets


def process_targets(script_name, num_batches):
    """
    Generate batches of target points, run the script, and collect successful ones.
    """
    successful_targets = []
    targets = [
        (np.asarray([0.15, 0, 0.35]) + np.random.uniform(-0.5, 0.5, (3))).tolist()
        for _ in range(BATCH_SIZE)
    ]
    successful_targets.extend(run_script_first(script_name, targets))

    return successful_targets


def generate_successful_targets(script_name, num_samples=100):
    tlist = []
    exec = ProcessPoolExecutor(max_workers=NUM_PROCESSES)
    args = [(script_name, BATCH_SIZE) for _ in range(num_samples)]

    successful_targets = []
    for a in args:
        tlist.append(exec.submit(process_targets, *a))

    tbar = tqdm(tlist, desc="Processing Targets")
    for t in tbar:
        successful_targets.extend(t.result())
        tbar.set_description(f"Found {len(successful_targets)} successful targets")

    # Save to JSON file
    with open(SUCCESSFUL_TARGETS_FILE, "w") as f:
        json.dump(successful_targets, f, indent=4)

    print(
        f"\nSaved {len(successful_targets)} successful targets to {SUCCESSFUL_TARGETS_FILE}"
    )


def load_or_generate_targets():
    """
    Load successful targets if available, otherwise generate new ones.
    """
    if os.path.exists(SUCCESSFUL_TARGETS_FILE):
        with open(SUCCESSFUL_TARGETS_FILE, "r") as f:
            return json.load(f)
    else:
        generate_successful_targets(
            "blender_rainbow_jax_single.py"
        )  # Adjust script if needed
        with open(SUCCESSFUL_TARGETS_FILE, "r") as f:
            return json.load(f)


def reduce_targets(targets, num_targets=1000):
    """
    Reduce a large list of 3D target points to `num_targets` points that are as far apart as possible.
    Uses k-means clustering and selects cluster centroids.
    """
    targets_np = np.array(targets)  # Convert list to NumPy array

    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=num_targets, n_init=10, random_state=42)
    kmeans.fit(targets_np)

    # Get the cluster centroids as the reduced target set
    reduced_targets = kmeans.cluster_centers_.tolist()

    return reduced_targets


# Example usage:
# all_results = {
#     "jax_run": [...],
#     "tensorflow_run": [...],
#     "fabrik_run": [...],
#     "ccd_run": [...],
# }
# plot_time_per_iteration_plotly(all_results, "time_per_iteration.png")


def main():
    # Load or generate successful targets
    targets = load_or_generate_targets()
    targets = reduce_targets(targets, num_targets=20)
    targets_str = json.dumps(targets)

    implementations = [
        [
            "tensorflow with custom objective cpu",
            "ik_tensorflow.py",
            ["--target_points", targets_str, "--subpoints", "0", "--cpu_only"],
        ],
        [
            "tensorflow non c cpu",
            "ik_tensorflow_non_c.py",
            ["--target_points", targets_str, "--subpoints", "0", "--cpu_only"],
        ],
        [
            "tensorflow non function cpu",
            "ik_tensorflow_non_function.py",
            ["--target_points", targets_str, "--subpoints", "0", "--cpu_only"],
        ],
        [
            "tensorflow non c non function cpu",
            "ik_tensorflow_non_c_function.py",
            ["--target_points", targets_str, "--subpoints", "0", "--cpu_only"],
        ],
    ]

    num_runs = 1  # Number of times to run each implementation
    all_results = {}

    for name, script, args in implementations:
        print(f"\n=== Testing {name} ===")
        ret = []
        for _ in range(num_runs):
            out = run_script_full(script, args)
            # print average statistics
            time_taken = np.mean([o[1] for o in out])
            avg_time_per_iteration = np.mean([o[2] for o in out])
            avg_iterations = np.mean([o[3] for o in out])
            success_rate = np.mean([o[4] for o in out])
            print("----------------------------")
            print(f"Average time taken: {time_taken:.8f} seconds")
            print(f"Average time per iteration: {avg_time_per_iteration:.8f} seconds")
            print(f"Average total iterations: {avg_iterations:.8f}")
            print(f"Average success rate: {success_rate:.8f}")

            ret.append(out)
        all_results[name] = ret

        # Save performance results
        output_file = "performance_results_ablation.json"
        with open(output_file, "w") as f:
            json.dump(all_results, f, indent=4)

    # print(f"\nPerformance data saved to {output_file}")

    # Load performance results
    with open("performance_results_ablation.json", "r") as f:
        all_results = json.load(f)

    plot_cpu_gpu_comparison(all_results, "results_cpu_gpu.png")
    create_gpu_cpu_latex_table(all_results)
    # Plot results for "only target" and "custom objective" conditions.
    plot_results(all_results, "only target", "results_only_target.png")
    plot_results(all_results, "custom objective", "results_custom_objective.png")
    plot_time_per_iteration(all_results, "results_time_per_iteration.png")
    plot_success_rate(all_results, "results_success_rate.png")

    print_latex_table(all_results)


if __name__ == "__main__":
    main()
