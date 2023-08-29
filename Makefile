pretty:
	clang-format --verbose -i include/*.hpp src/*.cu
	clang-format --verbose -i test/*.cu
	clang-format --verbose -i examples/*.cu

all: pretty

.PHONY : pretty
