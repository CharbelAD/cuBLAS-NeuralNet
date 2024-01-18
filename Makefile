CC=nvcc
CFLAGS=-lm -lcublas --extended-lambda
OBJ=matrix.o ann.o mnist.o main.o
EXE=nn.out

# Pattern rule for object files
%.o: %.cu
	$(CC) -c -o $@ $< $(CFLAGS)

# Main rule for executable
$(EXE): $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS)

# Clean rule
.PHONY: clean
clean:
	rm -f $(EXE) $(OBJ)
