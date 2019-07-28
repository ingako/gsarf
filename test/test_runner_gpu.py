#!/usr/bin/env python3

import os
from os.path import dirname, abspath
import time
import subprocess
import math
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon
from scipy.stats import friedmanchisquare

base_dir = os.path.expanduser('~')
gpu_path = f"{base_dir}/random-forest-gpu/"
moa_path = f"{base_dir}/moa-release-2018.6.0/"
stats_output_path = f"{gpu_path}/statistics/"

if not os.path.exists(stats_output_path):
    os.makedirs(stats_output_path)

data_sets = ["led_abrupt"]

for data_set in data_sets:

    data_path = f"./data/{data_set}/"
    gpu_output = f"{gpu_path}/{data_path}/result_gpu.csv"
    moa_output = f"{gpu_path}/{data_path}/result_moa.csv"

    cur_stats_output_path = f"{stats_output_path}/{data_set}_stats.csv"

    with open(cur_stats_output_path, "w") as out:
        out.write("data_file_name,"
                  "gpu_accuracy_mean,gpu_accuracy_stdev,"
                  "gpu_kappa_mean,gpu_kappa_stdev\n")

        for data_file_name in sorted(os.listdir(data_path)):
            if not data_file_name.endswith(".csv"):
                continue

            print(f"======================={data_file_name}=======================")
            print("Running gpu random forest...")

            command = ["time", "./main.o",
                       "-t", "100",
                       "-i", "200",
                       "-g", "200"
                       "-p", data_path,
                       "-n", data_file_name]
            print(" ".join(command))

            gpu_process = subprocess.Popen(command)
            gpu_process.wait()

            df = pd.read_csv(gpu_output)
            gpu_accuracy = df["accuracy"].values.tolist()
            gpu_kappa = df["kappa"].values.tolist()

            gpu_accuracy_mean = np.mean(gpu_accuracy)
            gpu_accuracy_stdev = np.std(gpu_accuracy)
            gpu_kappa_mean = np.mean(gpu_kappa)
            gpu_kappa_stdev = np.std(gpu_kappa)

            out.write(f"{data_file_name},"
                      f"{gpu_accuracy_mean},{gpu_accuracy_stdev},"
                      f"{gpu_kappa_mean},{gpu_kappa_stdev}\n")
            out.flush()

    with open(cur_stats_output_path, "r") as gpu_out:
        df_moa = pd.read_csv(moa_output)
        df_gpu = pd.read_csv(gpu_out)

        df_merged = df_moa.merge(df_gpu, on="data_file_name")
        df_merged.to_csv(cur_stats_output_path, index=False)

        # moa_accuracy_mean = df_merged["moa_accuracy_mean"]
        # gpu_accuracy_mean = df_merged["gpu_accuracy_mean"]

        # stat, p = friedmanchisquare(gpu_accuracy_mean, moa_accuracy_mean)
        # stat, p = wilcoxon(gpu_accuracy_mean, moa_accuracy_mean)

        # with open(distribution_status_output_path, "a") as out:
        #     alpha = 0.05
        #     if p > alpha:
        #         out.write("Same distirbutions")
        #     else:
        #         out.write("Different distributions")
