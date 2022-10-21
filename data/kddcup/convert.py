#!/usr/bin/env python3

def bin(arr, attr_val, new_line):
    if attr_val.isdigit():
        attr_val = int(attr_val)
    attr_idx = arr.index(attr_val)
    if attr_idx == -1:
        print("index must be found")

    for i in range(1, len(arr)):
        new_line.append('1' if i == attr_idx else '0')

def bin_continous(num_bins, bin_width, attr_val, new_line):
    attr_idx = int(attr_val) / bin_width
    for i in range(1, num_bins):
        new_line.append('1' if i == attr_idx else '0')

protocol_type = ["icmp", "tcp", "udp"]
flag = ["OTH", "REJ", "RSTO", "RSTOS0", "RSTR", "S0", "S1", "S2", "S3", "SF", "SH"]
land = [0, 1]
wrong_grament = [0, 1, 3]
urgent = [0, 1, 14, 2, 3, 5]
num_failed_logins = [0, 1, 2, 3, 4, 5]
logged_in = [0,1]
root_shell = [0,1]
su_attempted = [0,1,2]
num_shells = [0,1,2]
num_access_files = [0,1,2,3,4,5,6,7,8,9]
is_host_login = [0,1]
is_guest_login = [0,1]

keys = [protocol_type, flag, land, wrong_grament, urgent, num_failed_logins, logged_in, root_shell, su_attempted, num_shells, num_access_files, is_host_login, is_guest_login]

labels = ["back.", "buffer_overflow.", "ftp_write.", "guess_passwd.", "imap.", "ipsweep.", "land.", "loadmodule.", "multihop.", "neptune.", "nmap.", "normal.", "perl.", "phf.", "pod.", "portsweep.", "rootkit.", "satan.", "smurf.", "spy.", "teardrop.", "warezclient.", "warezmaster." ]

filename = "kddcup.raw.csv"
new_filename = "kddcup.csv"

with open(filename, "r") as f:
    lines = f.readlines()
lines = [x.strip() for x in lines]

with open(new_filename, "w") as out:
    for line in lines:
        new_line = []
        attributes = [x.strip() for x in line.split(',')]

        for i in range(0, len(keys)):
            bin(keys[i], attributes[i], new_line)

        for i in range(len(keys), len(attributes) - 1):
            if i == len(keys) or i == len(keys) + 1:
                bin_continous(6, 100, attributes[i], new_line)
            elif i == len(keys) + 9 or i == len(keys) + 10:
                bin_continous(6, 50, attributes[i], new_line)
            else:
                bin_continous(6, 20, float(attributes[i]) * 100, new_line)

        new_line.append(str(labels.index(attributes[-1])))

        new_line.append("\n")
        out.write(','.join(new_line))
        out.flush()
