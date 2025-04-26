import os
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.stats import norm
import sys
import glob
import argparse

def bootstrapp(t, rounds=50000):
    alpha = 0.8
    sub_set = int(alpha * len(t))
    tau_bootstr = []
    for _ in range(rounds):
        np.random.shuffle(t)
        t_b = t[:sub_set]
        t_b_sorted_50 = (np.sort(t_b)[int(len(t_b)/2.0 - 0.5)] + np.sort(t_b)[int(len(t_b)/2)]) / 2.0
        tau_bootstr.append(t_b_sorted_50)
    return tau_bootstr

def extract_exit_times_from_folders(timestep_fs):
    folder_pattern = "ramd_force-*"
    folders = sorted(glob.glob(folder_pattern))
    extracted_times = []
    folder_names = []

    for folder in folders:
        log_path = os.path.join(folder, "namd_output.log")
        if not os.path.isfile(log_path):
            print(f"Missing log in: {folder}")
            continue

        with open(log_path, 'r') as f:
            lines = f.readlines()

        exit_step = None
        for line in lines:
            if "EXIT:" in line and "LIGAND EXIT EVENT DETECTED" in line:
                match = re.search(r'EXIT:\s+(\d+)', line)
                if match:
                    exit_step = int(match.group(1))
                    break  # Only first relevant line
        if exit_step:
            time_ns = exit_step * timestep_fs * 1e-6
            extracted_times.append(time_ns)
            folder_names.append(folder)
        else:
            print(f"No EXIT time found in {folder}")

    # Save to detachment_times.dat
    with open("detachment_times.dat", "w") as out_file:
        out_file.write("# Folder_Name    Ligand_Exit_Time_ns\n")
        for name, time_ns in zip(folder_names, extracted_times):
            out_file.write(f"{name}\t{time_ns:.6f}\n")

    print("\nSaved extracted exit times to 'detachment_times.dat'.\n")

    return folder_names, extracted_times

def main():
    # Print developer info at start of execution
    print("Developed By \n GOURAV CHAKRABORTY \n Indian Institute of Technology (ISM) Dhanbad\n")

    note = """

************************************************************
*                                                          *
*  NOTE: THE PADDING TIME (-p/--padding) MUST BE LARGER    *
*        THAN THE HIGHEST DETACHMENT TIME IN THE DATA      *
*        (IN NANOSECONDS) TO AVOID ZERO VARIANCE ISSUES.   *
*                                                          *
************************************************************
"""

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=f"""
Analyze ligand residence (exit) times from RAMD simulations using NAMD.

This script extracts ligand exit steps from folders, computes residence times,
performs bootstrapping for tau estimation, and generates plots.

{note}
"""
    )

    parser.add_argument("-r", "--rounds", type=int, default=50000,
                        help="Number of bootstrap rounds (default: 50000)")
    parser.add_argument("-t", "--timestep", type=float, default=1.0,
                        help="Timestep in femtoseconds (default: 1.0 fs)")
    parser.add_argument("-p", "--padding", type=float, default=5.0,
                        help="Padding value in nanoseconds for bootstrapping (default: 5.0 ns)")

    args = parser.parse_args()

    folder_names, all_times = extract_exit_times_from_folders(args.timestep)

    if not all_times:
        print("No valid data found. Exiting.")
        sys.exit(1)

    times_set = [np.array(all_times)]
    fig = plt.figure(figsize=(5, 7))
    gs = gridspec.GridSpec(nrows=3, ncols=1, wspace=0.2, hspace=0.6)

    mue_set = []
    print("\n ============== Bootstrapping and tau computation ==================\n")

    times = times_set[0]
    if len(times) > 2:
        # Add dummy points if less than 10 to stabilize bootstrap
        while len(times) < 10:
            times = np.concatenate((times, [args.padding]))

        # Raw CDF plot
        ax0 = fig.add_subplot(gs[0, 0])
        ax0.hist(times, bins=int(len(times)/2), cumulative=True, histtype="step", color='k', lw=1)
        ax0.set_title("Raw CDF", fontsize=12)
        ax0.set_xlabel('Dissociation time [ns]', fontsize=10)
        ax0.plot([min(times), max(times)], [len(times)/2.0]*2, color='red', alpha=0.5)

        # Bootstrap tau estimation
        bt2 = bootstrapp(times, rounds=args.rounds)
        ax1 = fig.add_subplot(gs[1, 0])
        ax1.hist(x=bt2, bins=6, alpha=0.8, density=True, histtype="step")
        mu, std = norm.fit(bt2)
        mue_set.append(np.round(mu, 1))
        x = np.linspace(0.8 * min(bt2), max(bt2), 100)
        p = norm.pdf(x, mu, std)
        ax1.plot(x, p, 'k', linewidth=2)
        ax1.plot([mu, mu], [0, max(p)], color='red', alpha=0.5)
        ax1.plot([min(x), max(x)], [max(p)/2.0]*2, color='red', alpha=0.5)
        ax1.set_title("Tau distribution", fontsize=12)
        ax1.set_xlabel('Residence time [ns]', fontsize=10)
        ax1.set_yticks([])

        # KS test plot with fixed poisson evaluation points
        ax2 = fig.add_subplot(gs[2, 0])
        hist, bin_edges = np.histogram(times, bins=len(times))
        hist_center = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        CD = np.cumsum(hist) / np.max(np.cumsum(hist))

        poisson = 1 - np.exp(-hist_center / mu)  # Evaluate at histogram centers

        ax2.scatter(np.log10(hist_center), CD, marker='o')
        ax2.plot(np.log10(hist_center), poisson, color='k')
        ax2.set_ylim(0, 1)
        ax2.set_xlim(-1.5, 1.5)
        ax2.set_xlabel('log(Res. time [ns])', fontsize=10)
        ax2.set_title("KS test: {:.2f}".format(np.max(np.abs(poisson - CD))), fontsize=12)
        ax2.plot([np.log10(mu)]*2, [0, 1], color='red', alpha=0.5)
        ax2.grid(linestyle='--', linewidth=0.5)

        print("Relative residence time and SD: ", round(mu, 2), round(std, 2))

    plt.savefig("res_times.png", bbox_inches='tight', dpi=300)

    # Boxplot summary
    fig = plt.figure(figsize=(5, 2))
    plt.boxplot(times_set, showmeans=True,
                meanline=True,
                meanprops=dict(linestyle='--', linewidth=1.5, color='firebrick'),
                medianprops=dict(linestyle='-', linewidth=2.0, color='orange'))

    plt.ylabel("Residence time [ns]", fontsize=10)
    plt.title(f"Residence times, mean: {np.mean(mue_set):.2f}, std: {np.std(mue_set):.2f}", fontsize=10)
    plt.grid(linestyle='--', linewidth=0.5)
    plt.savefig("res_times_summary.png", bbox_inches='tight', dpi=300)

if __name__ == "__main__":
    main()

