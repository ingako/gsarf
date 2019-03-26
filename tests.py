#!/usr/bin/env python3

import os
import subprocess
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon
from scipy.stats import friedmanchisquare

base_dir = os.path.expanduser('~')
gpu_path = base_dir + "/random-forest-gpu/"
moa_path = base_dir + "/moa-release-2018.6.0/"

data_sets = ["stagger_gradual_drift"]

for data_set in data_sets:
    data_path = f"./data/{data_set}/"
    gpu_output = f"{gpu_path}/{data_path}/result_gpu.csv"
    moa_output = f"{moa_path}/bin/result_moa.csv"

    final_statistics_path = f"{gpu_path}/statistics/{data_set}_stats.csv"

    with open(final_statistics_path, "w") as out:
        out.write("#seed,gpu_accuracy_mean,gpu_accuracy_stdev,gpu_kappa_mean,gpu_kappa_stdev,"
                  "moa_accuracy_mean,moa_accuracy_stdev,moa_kappa_mean,moa_kappa_stdev\n")

    with open(final_statistics_path, "a") as out:
        for seed in range(0, 100):
            data_file_name = "{:02d}.csv".format(seed)
            print(f"==========================seed {seed}==========================")

            # run gpu random forest
            print("Running gpu random forest...")
            command = ["time", "./main.o", "-t", "100", "-p", data_path, "-n", data_file_name]
            print(" ".join(command))
            gpu_process = subprocess.Popen(command)
            gpu_process.wait()

            # run moa random forest
            print("Running moa random forest...")
            with open(moa_output, "w") as moa_out:
                command = ["bash",
                           f"{moa_path}/bin/{data_set}.sh",
                           f"{data_path}/{data_file_name}"]
                print(" ".join(command))
                moa_process = subprocess.call(command, stdout=moa_out)

            df = pd.read_csv(gpu_output)
            gpu_accuracy = df["accuracy"].values.tolist()
            gpu_kappa = df["kappa"].values.tolist()

            df = pd.read_csv(moa_output)
            moa_accuracy = df["classifications correct (percent)"]#.values.tolist()
            moa_kappa = df["Kappa Statistic (percent)"]#.values.tolist()

            gpu_accuracy_mean = np.mean(gpu_accuracy)
            gpu_accuracy_stdev = np.std(gpu_accuracy)
            gpu_kappa_mean = np.mean(gpu_kappa);
            gpu_kappa_stdev = np.std(gpu_kappa);

            moa_accuracy_mean = np.mean(moa_accuracy)
            moa_accuracy_stdev = np.std(moa_accuracy)
            moa_kappa_mean = np.mean(moa_kappa)
            moa_kappa_stdev = np.std(moa_kappa)

            out.write(f"{seed},"
                      f"{gpu_accuracy_mean},{gpu_accuracy_stdev},"
                      f"{gpu_kappa_mean},{gpu_kappa_stdev},"
                      f"{moa_accuracy_mean},{moa_accuracy_stdev},"
                      f"{moa_kappa_mean},{moa_kappa_stdev}\n")

    df = pd.read_csv(final_statistics_path)
    gpu_accuracy_mean = df["gpu_accuracy_mean"]
    moa_accuracy_mean = df["moa_accuracy_mean"]

    # stat, p = friedmanchisquare(gpu_accuracy_mean, moa_accuracy_mean)
    stat, p = wilcoxon(gpu_accuracy_mean, moa_accuracy_mean)

    with open(final_statistics_path, "w") as out:
        alpha = 0.05
        if p > alpha:
            out.write("Same distirbutions")
        else:
            out.write("Different distributions")
