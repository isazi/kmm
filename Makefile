pretty:
	clang-format --verbose -i include/*.hpp src/*.cu

all: pretty

.PHONY : pretty
