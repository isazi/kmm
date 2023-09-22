pretty:
	clang-format --verbose -i include/*.hpp src/*.cu src/*.cpp
	clang-format --verbose -i test/*.cu test/*.cpp
	clang-format --verbose -i examples/*.cu

all: pretty

.PHONY : pretty
