pretty:
	clang-format --verbose -i include/kmm/*.hpp include/kmm/*/*.hpp src/*.cpp src/*/*.cpp
	clang-format --verbose -i include/kmm/*.cuh include/kmm/*/*.cuh src/*.cu src/*/*.cu
	clang-format --verbose -i test/*.cpp
	clang-format --verbose -i examples/*.cu

all: pretty

.PHONY : pretty
