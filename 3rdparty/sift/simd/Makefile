# always needs -march flag for SIMD support
CXXFLAGS ?= -march=native -O3

# objects
OBJ = LibImages/LibImages.o \
      LibSSE/LibSSE.o \
      LibSift/KeyPoint.o \
      LibSift/LibSift.o \
      LibSift/Octave.o \
      LibSift/ScaleSpace.o \
      Utilities/Memory.o \
      Utilities/Parameters.o \
      Utilities/Time.o \
      Utilities/Utilities.o

# shared library with ctypes interface (to use from python)
LIB = libsift4ctypes.so

# command line executable
BIN = sift_roi


# default target builds everything
all: $(LIB) $(BIN)

# build objects for the shared object with the option '-fpic'
$(LIB): override CXXFLAGS := $(CXXFLAGS) -fpic
$(LIB): sift4ctypes.cpp $(OBJ)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -shared -o $@ $^ $(LDLIBS)

# build command line interface with gdal flags
$(BIN): CXXFLAGS := $(CXXFLAGS) `gdal-config --cflags` -std=c++11
$(BIN): LDLIBS   := $(LDLIBS) `gdal-config --libs`
$(BIN): sift_roi.cpp $(OBJ)


# bureaucracy
clean: ; $(RM) $(BIN) $(LIB) $(OBJ) *.o
