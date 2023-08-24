pretty:
	clang-format --verbose -i include/*.hpp src/*.cpp src/*.cu

all: pretty

.PHONY : pretty
