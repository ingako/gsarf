#!/usr/bin/env bash
./main.o -t 60 -i 100 -d 11 -e 201 -k 0.0 -x 0.05 -y 0.005 -p data/kddcup/ -n kddcup.csv
./main.o -c -t 60 -i 100 -d 11 -e 201 -k 0.0 -x 0.05 -y 0.005 -p data/kddcup/ -n kddcup.csv
