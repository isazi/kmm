pretty:
	clang-format-13 --verbose -i include/kmm/*.hpp include/kmm/*/*.hpp src/*.cpp src/*/*.cpp
	clang-format-13 --verbose -i test/*.cpp
	clang-format-13 --verbose -i examples/*.cu

all: pretty

.PHONY : pretty
