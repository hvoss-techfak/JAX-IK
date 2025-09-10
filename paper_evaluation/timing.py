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
BATCH_SIZE = 400  # Number of targets to process in one script call


def run_script_full(script_name, args, num_runs=5):
    """
    Run the given script multiple times with specified arguments and collect performance data.

    Args:
        script_name (str): Name of the Python script to execute.
        args (list): List of arguments to pass to the script.
        num_runs (int): Number of times to run the script. Default is 5.

    Returns:
        list: A list of lists containing tuples with iteration data:
              (iteration_number, time_taken, time_per_iteration, steps, success).
    """
    all_runs_data = []

    print("Running script:", script_name)

    env = os.environ.copy()
    # env["JAX_PLATFORMS"] = "cpu"
    for run in tqdm(range(1, num_runs + 1)):
        cmd = ["python", script_name] + args
        result = subprocess.run(cmd, capture_output=True, text=True, env=env)
        output = result.stdout + "\n" + result.stderr
        # print(output)

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
                # print(f"Iteration {iteration_num}: {time_taken:.8f} seconds, {steps} steps, success={success}")
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
    Run the script with a batch of targets and extract solved targets.

    Args:
        script_name (str): Name of the Python script to execute.
        target_list (list): List of target points to process.

    Returns:
        list: A list of solved target points.
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
    Generate batches of target points, run the script, and collect successful targets.

    Args:
        script_name (str): Name of the Python script to execute.
        num_batches (int): Number of batches to process.

    Returns:
        list: A list of successful target points.
    """
    successful_targets = []
    targets = [
        (np.asarray([0.15, 0, 0.35]) + np.random.uniform(-0.5, 0.5, (3))).tolist()
        for _ in range(BATCH_SIZE)
    ]
    successful_targets.extend(run_script_first(script_name, targets))

    return successful_targets


def generate_successful_targets(script_name, num_samples=100):
    """
    Generate successful targets by running the script in parallel.

    Args:
        script_name (str): Name of the Python script to execute.
        num_samples (int): Number of samples to generate. Default is 100.

    Returns:
        None
    """
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
    Load successful targets from a file if available, otherwise generate new ones.

    Returns:
        list: A list of successful target points.
    """
    if os.path.exists(SUCCESSFUL_TARGETS_FILE):
        with open(SUCCESSFUL_TARGETS_FILE, "r") as f:
            return json.load(f)
    # else:
    #     generate_successful_targets("blender_rainbow_jax_single.py")  # Adjust script if needed
    #     with open(SUCCESSFUL_TARGETS_FILE, "r") as f:
    #         return json.load(f)


def reduce_targets(targets, num_targets=1000):
    """
    Reduce a large list of 3D target points to a specified number of points using k-means clustering.

    Args:
        targets (list): List of 3D target points.
        num_targets (int): Number of target points to reduce to. Default is 1000.

    Returns:
        list: A list of reduced target points (cluster centroids).
    """
    targets_np = np.array(targets)  # Convert list to NumPy array

    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=num_targets, n_init=10, random_state=42)
    kmeans.fit(targets_np)

    # Get the cluster centroids as the reduced target set
    reduced_targets = kmeans.cluster_centers_.tolist()

    return reduced_targets


def main():
    """
    Main function to load or generate targets, reduce them, and test various implementations.
    """
    # Load successful targets
    targets = load_or_generate_targets()
    targets = reduce_targets(targets, num_targets=1010)
    targets_str = json.dumps(targets)

    implementations = [
        [
            "ipopt only target",
            "ik_ipopt.py",
            ["--target_points", targets_str, "--additional_objective_weight", "0.0"],
        ],
        [
            "ipopt with custom objective",
            "ik_ipopt.py",
            [
                "--target_points",
                targets_str,
                "--additional_objective_weight",
                "0.25",
            ],
        ],
        [
            "ccd only target",
            "ablation.py",
            ["--target_points", targets_str, "--solver_type", "ccd"],
        ],
        [
            "fabrik only target",
            "ablation.py",
            ["--target_points", targets_str, "--solver_type", "fabrik"],
        ],
        [
            "ccd with custom objective",
            "ablation.py",
            [
                "--target_points",
                targets_str,
                "--additional_objective_weight",
                "0.25",
                "--solver_type",
                "ccd",
                "--custom_objective",
                "True",
            ],
        ],
        [
            "fabrik with custom objective",
            "ablation.py",
            [
                "--target_points",
                targets_str,
                "--additional_objective_weight",
                "0.25",
                "--solver_type",
                "fabrik",
                "--custom_objective",
                "True",
            ],
        ],
        [
            "tensorflow only target gpu",
            "ik_tensorflow.py",
            [
                "--target_points",
                targets_str,
                "--additional_objective_weight",
                "0.0",
                "--subpoints",
                "0",
            ],
        ],
        [
            "jax only target gpu",
            "ik_jax.py",
            [
                "--target_points",
                targets_str,
                "--additional_objective_weight",
                "0.0",
                "--subpoints",
                "0",
            ],
        ],
        [
            "jax with custom objective gpu",
            "ik_jax.py",
            [
                "--target_points",
                targets_str,
                "--additional_objective_weight",
                "0.25",
                "--subpoints",
                "0",
            ],
        ],
        [
            "tensorflow with custom objective gpu",
            "ik_tensorflow.py",
            [
                "--target_points",
                targets_str,
                "--additional_objective_weight",
                "0.25",
                "--subpoints",
                "0",
            ],
        ],
        [
            "tensorflow only target cpu",
            "ik_tensorflow.py",
            [
                "--target_points",
                targets_str,
                "--additional_objective_weight",
                "0.0",
                "--subpoints",
                "0",
                "--cpu_only",
            ],
        ],
        [
            "jax only target cpu",
            "ik_jax.py",
            [
                "--target_points",
                targets_str,
                "--additional_objective_weight",
                "0.0",
                "--subpoints",
                "0",
                "--cpu_only",
            ],
        ],
        [
            "jax with custom objective cpu",
            "ik_jax.py",
            [
                "--target_points",
                targets_str,
                "--subpoints",
                "0",
                "--additional_objective_weight",
                "0.25",
                "--cpu_only",
            ],
        ],
        [
            "tensorflow with custom objective cpu",
            "ik_tensorflow.py",
            [
                "--target_points",
                targets_str,
                "--subpoints",
                "0",
                "--additional_objective_weight",
                "0.25",
                "--cpu_only",
            ],
        ],
        [
            "tensorflow only target subpoints 5 cpu",
            "ik_tensorflow.py",
            [
                "--target_points",
                targets_str,
                "--additional_objective_weight",
                "0.0",
                "--subpoints",
                "5",
                "--cpu_only",
            ],
        ],
        [
            "jax only target subpoints 5 cpu",
            "ik_jax.py",
            [
                "--target_points",
                targets_str,
                "--additional_objective_weight",
                "0.0",
                "--subpoints",
                "5",
                "--cpu_only",
            ],
        ],
        [
            "jax with custom objective subpoints 5 cpu",
            "ik_jax.py",
            [
                "--target_points",
                targets_str,
                "--subpoints",
                "5",
                "--additional_objective_weight",
                "0.25",
                "--cpu_only",
            ],
        ],
        [
            "tensorflow with custom objective subpoints 5 cpu",
            "ik_tensorflow.py",
            [
                "--target_points",
                targets_str,
                "--subpoints",
                "5",
                "--additional_objective_weight",
                "0.25",
                "--cpu_only",
            ],
        ],
        [
            "tensorflow only target subpoints 10 cpu",
            "ik_tensorflow.py",
            [
                "--target_points",
                targets_str,
                "--additional_objective_weight",
                "0.0",
                "--subpoints",
                "10",
                "--cpu_only",
            ],
        ],
        [
            "jax only target subpoints 10 cpu",
            "ik_jax.py",
            [
                "--target_points",
                targets_str,
                "--additional_objective_weight",
                "0.0",
                "--subpoints",
                "10",
                "--cpu_only",
            ],
        ],
        [
            "jax with custom objective subpoints 10 cpu",
            "ik_jax.py",
            [
                "--target_points",
                targets_str,
                "--subpoints",
                "10",
                "--additional_objective_weight",
                "0.25",
                "--cpu_only",
            ],
        ],
        [
            "tensorflow with custom objective subpoints 10 cpu",
            "ik_tensorflow.py",
            [
                "--target_points",
                targets_str,
                "--subpoints",
                "10",
                "--additional_objective_weight",
                "0.25",
                "--cpu_only",
            ],
        ],
        [
            "tensorflow only target subpoints 5 gpu",
            "ik_tensorflow.py",
            [
                "--target_points",
                targets_str,
                "--additional_objective_weight",
                "0.0",
                "--subpoints",
                "5",
            ],
        ],
        [
            "jax only target subpoints 5 gpu",
            "ik_jax.py",
            [
                "--target_points",
                targets_str,
                "--additional_objective_weight",
                "0.0",
                "--subpoints",
                "5",
            ],
        ],
        [
            "jax with custom objective subpoints 5 gpu",
            "ik_jax.py",
            [
                "--target_points",
                targets_str,
                "--additional_objective_weight",
                "0.25",
                "--subpoints",
                "5",
            ],
        ],
        [
            "tensorflow with custom objective subpoints 5 gpu",
            "ik_tensorflow.py",
            [
                "--target_points",
                targets_str,
                "--additional_objective_weight",
                "0.25",
                "--subpoints",
                "5",
            ],
        ],
        [
            "tensorflow only target subpoints 10 gpu",
            "ik_tensorflow.py",
            [
                "--target_points",
                targets_str,
                "--additional_objective_weight",
                "0.0",
                "--subpoints",
                "10",
            ],
        ],
        [
            "jax only target subpoints 10 gpu",
            "ik_jax.py",
            [
                "--target_points",
                targets_str,
                "--additional_objective_weight",
                "0.0",
                "--subpoints",
                "10",
            ],
        ],
        [
            "jax with custom objective subpoints 10 gpu",
            "ik_jax.py",
            [
                "--target_points",
                targets_str,
                "--additional_objective_weight",
                "0.25",
                "--subpoints",
                "10",
            ],
        ],
        [
            "tensorflow with custom objective subpoints 10 gpu",
            "ik_tensorflow.py",
            [
                "--target_points",
                targets_str,
                "--additional_objective_weight",
                "0.25",
                "--subpoints",
                "10",
            ],
        ],
    ]

    all_results = {}

    # for name, script, args in implementations:
    #     print(f"\n=== Testing {name} ===")
    #     ret = []
    #     ret.append(run_script_full(script, args))
    #     all_results[name] = ret
    #
    #     # Save performance results
    #     output_file = "performance_results_subpoints.json"
    #     with open(output_file, "w") as f:
    #         json.dump(all_results, f, indent=4)

    # print(f"\nPerformance data saved to {output_file}")

    # Load performance results
    with open("performance_results_subpoints.json", "r") as f:
        all_results = json.load(f)

    plot_cpu_gpu_comparison(all_results, "results_cpu_gpu.png")
    create_gpu_cpu_latex_table(all_results)
    plot_results(all_results, "only target", "results_only_target.png")
    plot_results(all_results, "custom objective", "results_custom_objective.png")
    plot_time_per_iteration(all_results, "results_time_per_iteration.png")
    plot_success_rate(all_results, "results_success_rate.png")

    print_latex_table(all_results)


if __name__ == "__main__":
    main()
