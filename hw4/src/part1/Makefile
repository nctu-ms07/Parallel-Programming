TARGET = hello pi_block_linear pi_block_tree pi_gather pi_nonblock_linear pi_reduce

CXX := mpicxx
CXXFLAGS := -std=c++17 -Wall -Wextra -O3

.PHONY: all
all: $(TARGET)
	mv hello mpi_hello

%: %.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

.PHONY: clean
clean:
	rm -f *.o $(TARGET) mpi_hello