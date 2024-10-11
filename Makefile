pretty:
	clang-format-16 --verbose -i include/kmm/*.hpp include/kmm/*/*.hpp src/*.cpp src/*/*.cpp src/*/*.cu src/*/*.cuh
	clang-format-16 --verbose -i test/*.cpp
	clang-format-16 --verbose -i examples/*.cu

all: pretty

.PHONY : pretty
