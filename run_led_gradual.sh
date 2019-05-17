#!/usr/bin/env bash

./main.o -c -t 60 -i 50 -d 10 -e 80 -k 0.25 -p data/led/ -n led_gradual_test.csv

# -c turns on the state-adaptive algorithm
# -t number of trees
# -i number of instances to be trained at the same time on GPU
# -d depth of the trees
# -e maximum edit distance
# -k kappa threshold
# -p path that contains the data file
# -n data file name
