pretty:
	clang-format --verbose -i include/*.hpp src/*.cu
	clang-format --verbose -i test/*.cu

all: pretty

.PHONY : pretty
