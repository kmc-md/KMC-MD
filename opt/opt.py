import os
import shutil
import subprocess
import time
import numpy as np
from bayes_opt import BayesianOptimization
from scipy.interpolate import interp1d
import re
import pandas as pd
import sys
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events

# Open the main.py script and modify the rate coefficients
def modify_kmc_script(new_rate_coefficients, script_path):
    with open(script_path, 'r') as file:
        lines = file.readlines()

    for i, line in enumerate(lines):
        for reaction_id, new_k in new_rate_coefficients.items():
            if line.strip().startswith(f"{reaction_id}:"):
                new_k_str = f"'k': {new_k}" + "}," if i < len(lines) - 1 else "}"
                lines[i] = re.sub(r"\'k\':\s*[\d.E+-]+.*", new_k_str, line)

    with open(script_path, 'w') as file:
        file.writelines(lines)

# Update the base directory in the main.py script for each runX folder
def update_base_dir_in_script(script_path, new_base_dir):
    with open(script_path, 'r') as file:
        lines = file.readlines()

    updated_lines = []
    for line in lines:
        if "base_dir=" in line:
            indentation = re.match(r"\s*", line).group()
            updated_line = f"{indentation}base_dir = '{new_base_dir}'\n"
            updated_lines.append(updated_line)
        else:
            updated_lines.append(line)

    with open(script_path, 'w') as file:
        file.writelines(updated_lines)

# Prepare the runX folders to copy the files and run them
def prepare_and_submit_runs(base_dir, rate_coefficients, num_runs=3):
    run_dirs = []
    job_ids = []

    file_extensions = ['.itp', '.mdp', '.gro', '.top']

    for i in range(1, num_runs + 1):
        run_dir = os.path.join(base_dir, f"run_{i}")
        os.makedirs(run_dir, exist_ok=True)
        run_dirs.append(run_dir)

        for item in os.listdir(base_dir):
            if any(item.endswith(ext) for ext in file_extensions) or item == 'main.py' or item == 'sub.sh': # Name of bash file to submit the main.py script
                src_path = os.path.join(base_dir, item)
                dest_path = os.path.join(run_dir, item)
                shutil.copy2(src_path, dest_path)

        kmc_script_path = os.path.join(run_dir, 'main.py')

        modify_kmc_script(rate_coefficients, kmc_script_path)
        update_base_dir_in_script(kmc_script_path, run_dir)

        result = subprocess.run(['sbatch', 'sub.sh'], cwd=run_dir, capture_output=True, text=True)
        job_id = result.stdout.strip().split()[-1]
        job_ids.append(job_id)

    return run_dirs, job_ids

# Monitor the jobs in the three runX folders after every 60 s to check if they are completed
def monitor_jobs(job_ids, check_interval=60):
    """
    Monitors the status of submitted jobs until all have completed.
    """
    while True:
        jobs_running = False
        for job_id in job_ids:
            result = subprocess.run(['squeue', '-j', job_id], capture_output=True, text=True)
            if job_id in result.stdout:
                jobs_running = True
                break
        
        if not jobs_running:
            print("All jobs have completed.")
            break
        else:
            print("Jobs still running. Checking again in 60 seconds.")
            time.sleep(check_interval)
    return  

# Compile output data from the runX folders to calculate average % DHC and time (min)
def process_simulation_outputs(run_dirs):
    all_data = []
    for run_dir in run_dirs:
        output_path = os.path.join(run_dir, 'output.txt')
        if os.path.exists(output_path):
            data = pd.read_csv(output_path, sep="\s+", header=None, names=['Time', 'Dehydrochlorination'])
            data.set_index('Time', inplace=True)
            all_data.append(data)
        else:
            print(f"Output file not found in directory: {run_dir}")

    combined_data = pd.concat(all_data, axis=1)
    mean_dehydrochlorination = combined_data.mean(axis=1, skipna=True)

    return mean_dehydrochlorination

# Clean up the runX folders before next iteration
def cleanup_run_directories(base_dir):
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path) and item.startswith("run_"):
            shutil.rmtree(item_path)
            
# Load experimental data from exp.xvg file (First column is time(min) and second column is % DHC)
def load_experimental_data(experimental_data_path):
    exp_data = pd.read_csv(experimental_data_path, sep="\s+", header=None, names=['Time', 'Dehydrochlorination'])
    exp_data.set_index('Time', inplace=True)
    return exp_data

# Calculate the RMSE between simulated and experimental data
def compare_results_to_experimental(sim_results, exp_data):
    sim_results = sim_results[sim_results.index <= 4]
    exp_data = exp_data[exp_data.index <= 4]
    
    sim_dehydro = sim_results.values.flatten().astype(float)
    sim_times_index = sim_results.index.astype(float)
    
    f_interp = interp1d(sim_times_index, sim_dehydro, kind='cubic', bounds_error=False, fill_value=np.nan)
    
    exp_dehydro_levels = exp_data.index.astype(float)
    interp_sim_times = f_interp(exp_dehydro_levels)
    
    valid_indices = ~np.isnan(interp_sim_times)
    interp_sim_times = interp_sim_times[valid_indices]
    valid_exp_times = exp_data.values.flatten()[valid_indices]
    error = np.sqrt(np.mean((interp_sim_times - valid_exp_times) ** 2))
    
    return error

# Main iteration loop
def main():
    base_dir = '/path/to/your/optimization/directory'
    experimental_data_path = '/path/to/your/experimental/data'
    pbounds = {
        'rate1': (1e7, 1e9),
        'rate2': (1e7, 1e9),
        'rate3': (1e7, 1e9),
    }

    output_file_path = os.path.join(base_dir, 'opt.txt')
    convergence_threshold = 0.01  # Minimum improvement in RMSE to consider
    stable_iters = 5  # Number of iterations to check for stability in RMSE improvement
    rmse_history = []  # Keep track of RMSE values to check for convergence

    def optimized_objective(rate1, rate2, rate3):
        log_msg = f"New Parameters: rate1={rate1}, rate2={rate2}, rate3={rate3}\n"
        
        rate_coefficients = {1: rate1, 2: rate2, 3: rate3}
        run_dirs, job_ids = prepare_and_submit_runs(base_dir, rate_coefficients)
        monitor_jobs(job_ids)
        
        sim_results = process_simulation_outputs(run_dirs)
        exp_data = load_experimental_data(experimental_data_path)
        rmse = compare_results_to_experimental(sim_results, exp_data)
        cleanup_run_directories(base_dir)

        log_msg += f"RMSE: {rmse}\n"
        with open(output_file_path, 'a') as file:
            file.write(log_msg)

        return -rmse

    optimizer = BayesianOptimization(
        f=optimized_objective,
        pbounds=pbounds,
        random_state=1,
    )

    logger = JSONLogger(path="./logs.json")
    optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

    for i in range(100):  # Maximum number of iterations
        optimizer.maximize(
            init_points=10 if i == 0 else 0,
            n_iter=1,  # Optimize one step at a time
        )
        latest_rmse = -optimizer.max['target']
        rmse_history.append(latest_rmse)

        # Check for convergence
        if i >= stable_iters and all(abs(rmse_history[-j-1] - rmse_history[-j]) < convergence_threshold for j in range(1, stable_iters)):
            print(f"Convergence achieved after {i+1} iterations.")
            break

    best_params = optimizer.max['params']
    best_rmse = -optimizer.max['target']

    with open(output_file_path, 'a') as file:
        for i, res in enumerate(optimizer.res):
            file.write(f"Iteration: {i+1}, Target: {res['target']}, Params: {res['params']}\n")

        file.write(f"\nBest Parameters: {best_params}\n")
        file.write(f"Best RMSE: {best_rmse}\n")
        file.write("Convergence achieved.\n")


if __name__ == "__main__":
    main()

