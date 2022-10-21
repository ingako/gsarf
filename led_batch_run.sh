#!/usr/bin/env bash

for seed in {0..9}
do
	/usr/bin/time -f "%S %U" ./main.o -t 60 -i 100 -d 11 -e 300 -k 0.05 -b 0.1 -p data/led_2_w1/ -n ${seed}.csv 2>> data/led_2_w1/log.garf.time
done

for seed in {0..9}
do
	/usr/bin/time -f "%S %U" ./main.o -c -t 60 -i 100 -d 11 -e 300 -k 0.05 -b 0.1 -p data/led_2_w1/ -n ${seed}.csv 2>> data/led_2_w1/log.gsarf.time
done

for seed in {0..9}
do
	/usr/bin/time -f "%S %U" ./main.o -t 60 -i 100 -d 11 -e 300 -k 0.05 -b 0.1 -p data/led_2_w25k/ -n ${seed}.csv 2>> data/led_2_w25k/log.garf.time
done

for seed in {0..9}
do
	/usr/bin/time -f "%S %U" ./main.o -c -t 60 -i 100 -d 11 -e 300 -k 0.05 -b 0.1 -p data/led_2_w25k/ -n ${seed}.csv 2>> data/led_2_w25k/log.gsarf.time
done

for seed in {0..9}
do
	/usr/bin/time -f "%S %U" ./main.o -t 60 -i 100 -d 11 -e 110 -k 0.15 -b 0.1 -p data/led_2_w50k/ -n ${seed}.csv 2>> data/led_2_w50k/log.garf.time
done

for seed in {0..9} 
do
	/usr/bin/time -f "%S %U" ./main.o -c -t 60 -i 100 -d 11 -e 110 -k 0.15 -b 0.1 -p data/led_2_w50k/ -n ${seed}.csv 2>> data/led_2_w50k/log.gsarf.time
done

for seed in {0..9}
do
	/usr/bin/time -f "%S %U" ./main.o -t 60 -i 100 -d 11 -e 130 -k 0.85 -b 0.1 -p data/led_3_w1/ -n ${seed}.csv 2>> data/led_3_w1/log.garf.time
done

for seed in {0..9}
do
	/usr/bin/time -f "%S %U" ./main.o -c -t 60 -i 100 -d 11 -e 130 -k 0.85 -b 0.1 -p data/led_3_w1/ -n ${seed}.csv 2>> data/led_3_w1/log.gsarf.time
done

for seed in {0..9}
do
	/usr/bin/time -f "%S %U" ./main.o -t 60 -i 100 -d 11 -e 130 -k 0.15 -b 0.1 -p data/led_3_w25k/ -n ${seed}.csv 2>> data/led_3_w25k/log.garf.time
done

for seed in {0..9}
do
	/usr/bin/time -f "%S %U" ./main.o -c -t 60 -i 100 -d 11 -e 130 -k 0.15 -b 0.1 -p data/led_3_w25k/ -n ${seed}.csv 2>> data/led_3_w25k/log.gsarf.time
done

for seed in {0..9}
do
	/usr/bin/time -f "%S %U" ./main.o -t 60 -i 100 -d 11 -e 130 -k 0.15 -b 0.1 -p data/led_3_mixed/ -n ${seed}.csv 2>> data/led_3_mixed/log.garf.time
done

for seed in {0..9}
do
	/usr/bin/time -f "%S %U" ./main.o -c -t 60 -i 100 -d 11 -e 130 -k 0.15 -b 0.1 -p data/led_3_mixed/ -n ${seed}.csv 2>> data/led_3_mixed/log.gsarf.time
done
