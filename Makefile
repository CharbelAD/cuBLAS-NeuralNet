main: main.cu ann.cu matrix.cu mnist.cu
	nvcc -o nn.out matrix.cu ann.cu mnist.cu main.cu -lm -lcublas

clean:
	rm -f nn.out