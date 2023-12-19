pretty:
	clang-format --verbose -i include/kmm/*.hpp include/kmm/*/*.hpp src/*.cpp src/*/*.cpp
	clang-format --verbose -i test/*.cpp
	clang-format --verbose -i examples/*.cu

all: pretty

.PHONY : pretty
