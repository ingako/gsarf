all:
	nvcc -O3 -o main.o src/main.cpp src/*.cu

clean:
	rm *.o
