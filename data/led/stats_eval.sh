#!/usr/bin/env python3

import os
import subprocess
import math
import pandas as pd
import numpy as np

moa_output = "stats.csv"

with open(moa_output, "w") as out:
    out.write("data_file_name,"
              "accuracy_mean,accuracy_stdev\n")

    accuracies = []

    for seed in range(0, 10):
        data_file_name = f"result_moa_{seed}.csv"

        with open(data_file_name, "r") as moa_out_raw:

            df = pd.read_csv(moa_out_raw)
            moa_accuracy = df["classifications correct (percent)"]
            moa_kappa = df["Kappa Statistic (percent)"]

            moa_accuracy_mean = np.mean(moa_accuracy)
            moa_accuracy_stdev = np.std(moa_accuracy)
            moa_kappa_mean = np.mean(moa_kappa)
            moa_kappa_stdev = np.std(moa_kappa)

            accuracies.append(moa_accuracy_mean)

    acc_mean = sum(accuracies) / float(len(accuracies))
    acc_std = np.std(accuracies)

    out.write(f"moa,"
              f"{acc_mean},{acc_std}\n")
    out.flush()



stats_output = "stats.csv"


data_file_names = ["result_gpu_bg", "result_gpu"]

with open(stats_output, "a") as out:
    for data_file_prefix in data_file_names:

        accuracies = []
        for seed in range(0, 10):

            data_file_name = f"{data_file_prefix}_{seed}.csv"

            with open(data_file_name, "r") as result_file:
                df = pd.read_csv(result_file)
                accuracy = df["accuracy"]
                kappa = df["kappa"]

                accuracy_mean = np.mean(accuracy)
                accuracy_stdev = np.std(accuracy)
                kappa_mean = np.mean(kappa)
                kappa_stdev = np.std(kappa)

                accuracies.append(accuracy_mean)

        acc_mean = sum(accuracies) / float(len(accuracies))
        acc_std = np.std(accuracies)

        out.write(f"{data_file_prefix},"
                  f"{acc_mean},{acc_std}\n")
        out.flush()
