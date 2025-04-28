CXX = nvcc
CXXFLAGS = -O3

TARGET = nbody_cuda
SRC = nbody.cu

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRC)

solar.out: $(TARGET)
	date
	./$(TARGET) planet 200 5000000 10000 > solar.out
	date

solar.pdf: solar.out
	python3 plot.py solar.out solar.pdf 1000

random.out: $(TARGET)
	date
	./$(TARGET) 1000 1 10000 100 > random.out
	date

clean:
	rm -f $(TARGET) solar.out random.out solar.pdf

