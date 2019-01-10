all:
	nvcc -o main.o main.cu

clean:
	rm *.o
