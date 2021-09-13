# the following two options are used to control all C and C++ compilations
CFLAGS   ?= -march=native -O3
CXXFLAGS ?= -march=native -O3
export CFLAGS
export CXXFLAGS

# these options are only used for the programs directly inside "./c/"
IIOLIBS     = -lz -ltiff -lpng -ljpeg -lm


# default rule builds only the programs necessary for the test
default: sift libraries

# the "all" rule builds three further correlators
all: default

# test for the default configuration
test: default
	env PYTHONPATH=. pytest tests

sift:
	$(MAKE) -j -C 3rdparty/sift/simd libsift4ctypes.so
	cp 3rdparty/sift/simd/libsift4ctypes.so lib


# generic rule for building binary objects from C sources
c/%.o : c/%.c
	$(CC) -fpic $(CFLAGS) -c $< -o $@

# generic rule for building binary objects from C++ sources
c/%.o: c/%.cpp
	$(CXX) -fpic $(CXXFLAGS) -c $^ -o $@


#
# rules to build the dynamic objects that are used via ctypes
#

libraries: lib/disp_to_h.so

lib/disp_to_h.so: c/disp_to_h.o c/rpc.o
	$(CC) -shared $^ $(IIOLIBS) -o $@




# automatic dependency generation
-include .deps.mk
.PHONY:
depend:
	$(CC) -MM `ls c/*.c c/*.cpp` | sed '/^[^ ]/s/^/c\//' > .deps.mk


# rules for cleaning, nothing interesting below this point
clean: clean_sift

distclean: clean ; $(RM) .deps.mk


# clean targets that use recursive makefiles
clean_sift:       ; $(MAKE) clean -C 3rdparty/sift/simd


.PHONY: default all sift clean clean_sift\
	test distclean


# The following conditional statement appends "-std=gnu99" to CFLAGS when the
# compiler does not define __STDC_VERSION__.  The idea is that many older
# compilers are able to compile standard C when given that option.
# This hack seems to work for all versions of gcc, clang and icc.
CVERSION = $(shell $(CC) -dM -E - < /dev/null | grep __STDC_VERSION__)
ifeq ($(CVERSION),)
CFLAGS := $(CFLAGS) -std=gnu99
endif
