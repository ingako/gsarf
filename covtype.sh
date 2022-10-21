#!/usr/bin/env bash
./main.o -t 100 -i 50 -d 11 -e 300 -k 0 -p data/covtype/ -n covtype_binary_attributes.csv
./main.o -c -t 100 -i 50 -d 11 -e 300 -k 0 -p data/covtype/ -n covtype_binary_attributes.csv
