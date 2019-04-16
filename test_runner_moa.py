#!/usr/bin/env python3

import os
import subprocess
import math
import pandas as pd
import numpy as np

base_dir = os.path.expanduser('~')
gpu_path = f"{base_dir}/random-forest-gpu/"
moa_path = f"{base_dir}/moa/"

data_sets = ["hyperplane_abrupt", "agrawal_abrupt", "hyperplane", "agrawal"]

for data_set in data_sets:
    data_path = f"./data/{data_set}/"
    moa_output = f"{data_path}/result_moa.csv"

    num_of_attributes = 0
    with open(f"{data_path}/attributes.txt", "r") as f:
        num_of_attributes = len(f.readline().split(',')) - 1
        num_of_attributes = math.floor(math.log2(num_of_attributes))

    with open(moa_output, "w") as out:
        out.write("data_file_name,"
                  "moa_accuracy_mean,moa_accuracy_stdev,"
                  "moa_kappa_mean,moa_kappa_stdev\n")

        for data_file_name in sorted(os.listdir(data_path)):
            if not data_file_name.endswith(".csv"):
                continue

            print(f"======================={data_file_name}=======================")

            print("Running moa random forest...")
            with open("moa_out_raw.csv", "w") as moa_out_raw:
                command = ["bash",
                           f"{moa_path}/bin/moa_runner.sh",
                           f"{data_path}/{data_file_name}",
                           str(num_of_attributes)]
                print(" ".join(command))
                moa_process = subprocess.call(command, stdout=moa_out_raw)

            with open("moa_out_raw.csv", "r") as moa_out_raw:

                df = pd.read_csv(moa_out_raw)
                moa_accuracy = df["classifications correct (percent)"]
                moa_kappa = df["Kappa Statistic (percent)"]

                moa_accuracy_mean = np.mean(moa_accuracy)
                moa_accuracy_stdev = np.std(moa_accuracy)
                moa_kappa_mean = np.mean(moa_kappa)
                moa_kappa_stdev = np.std(moa_kappa)

                out.write(f"{data_file_name},"
                          f"{moa_accuracy_mean},{moa_accuracy_stdev},"
                          f"{moa_kappa_mean},{moa_kappa_stdev}\n")
                out.flush()
