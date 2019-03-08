all:
	nvcc -o main.o main.cu --std=c++11

clean:
	rm *.o
