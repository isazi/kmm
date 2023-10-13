pretty:
	clang-format --verbose -i include/kmm/*.hpp include/kmm/*.cuh src/*.cu src/*.cpp
	clang-format --verbose -i test/*.cu test/*.cpp
	clang-format --verbose -i examples/*.cu

all: pretty

.PHONY : pretty
