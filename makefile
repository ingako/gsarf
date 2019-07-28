all:
	nvcc -o main.o src/main.cu --std=c++14 

debug:
	nvcc -o main.o src/main.cu --std=c++14 -DDEBUG=1 -g -lineinfo

clean:
	rm *.o
