#!/usr/bin/env bash

for seed in {0..9}
do
    /usr/bin/time -f "%S %U" ./main.o -t 100 -i 50 -d 11 -e 200 -k 0.25 -b 0.1 -p data/agrawal_2_w1/ -n ${seed}.csv 2>> data/agrawal_2_w1/log.garf.time
done

for seed in {0..9}
do
    /usr/bin/time -f "%S %U" ./main.o -c -t 100 -i 50 -d 11 -e 200 -k 0.25 -b 0.1 -p data/agrawal_2_w1/ -n ${seed}.csv 2>> data/agrawal_2_w1/log.gsarf.time
done

for seed in {0..9}
do
    /usr/bin/time -f "%S %U" ./main.o -t 100 -i 50 -d 11 -e 90 -k 0.33 -p data/agrawal_2_w25k/ -n ${seed}.csv 2>> data/agrawal_2_w25k/log.garf.time
done

for seed in {0..9}
do
    /usr/bin/time -f "%S %U" ./main.o  -c -t 100 -i 50 -d 11 -e 90 -k 0.33 -b 0.1 -p data/agrawal_2_w25k/ -n ${seed}.csv 2>> data/agrawal_2_w25k/log.gsarf.time
done

for seed in {0..9}
do
	/usr/bin/time -f "%S %U" ./main.o -t 100 -i 50 -d 11 -e 90 -k 0.33 -b 0.1 -p data/agrawal_2_w50k/ -n ${seed}.csv 2>> log.garf.time
done

for seed in {0..9}
do
	/usr/bin/time -f "%S %U" ./main.o -c -t 100 -i 50 -d 11 -e 90 -k 0.33 -b 0.1 -p data/agrawal_2_w50k/ -n ${seed}.csv 2>> log.gsarf.time
done

for seed in {0..9}
do
	/usr/bin/time -f "%S %U" ./main.o -t 100 -i 50 -d 11 -e 180 -k 0.43 -p data/agrawal_3_w1/ -n ${seed}.csv 2>> data/agrawal_3_w1/log.garf.time
done

for seed in {0..9}
do
    /usr/bin/time -f "%S %U" ./main.o -c -t 100 -i 50 -d 11 -e 180 -k 0.43 -p data/agrawal_3_w1/ -n ${seed}.csv 2>> data/agrawal_3_w1/log.gsarf.time
done

for seed in {0..9}
do
    /usr/bin/time -f "%S %U" ./main.o -t 100 -i 50 -d 11 -e 120 -k 0.35 -p data/agrawal_3_w25k/ -n ${seed}.csv 2>> data/agrawal_3_w25k/log.garf.time
done

for seed in {0..9}
do
    /usr/bin/time -f "%S %U" ./main.o -c -t 100 -i 50 -d 11 -e 120 -k 0.35 -p data/agrawal_3_w25k/ -n ${seed}.csv 2>> data/agrawal_3_w25k/log.gsarf.time
done

for seed in {0..9}
do
    /usr/bin/time -f "%S %U" ./main.o -t 100 -i 50 -d 11 -e 120 -k 0.35 -p data/agrawal_3_mixed/ -n ${seed}.csv 2>> data/agrawal_3_mixed/log.garf.time
done

for seed in {0..9}
do
    /usr/bin/time -f "%S %U" ./main.o -c -t 100 -i 50 -d 11 -e 120 -k 0.35 -p data/agrawal_3_mixed/ -n ${seed}.csv 2>> data/agrawal_3_mixed/log.gsarf.time
done
