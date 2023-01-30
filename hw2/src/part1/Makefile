TARGET := pi.out

CXX := clang++
ifeq (/usr/bin/clang++-11,$(wildcard /usr/bin/clang++-11*))
    CXX=clang++-11
endif

CXXFLAGS := -std=c++17 -O3 -mavx2 -Wall

.PHONY: all
all: $(TARGET)

%.out: %.cpp
	$(CXX) $(CXXFLAGS) -I./Xoshiro256PlusSIMD/include $< -o $@ -lpthread

.PHONY: clean
clean:
	rm -f *.o $(TARGET)
