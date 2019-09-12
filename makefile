all:
	nvcc -O3 -o main.o src/ADWIN.cpp src/main.cpp src/*.cu --std=c++14

clean:
	rm *.o
