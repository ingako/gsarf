all:
	nvcc -O3 -o main.o src/ADWIN.cpp src/main.cpp src/*.cu

clean:
	rm *.o
