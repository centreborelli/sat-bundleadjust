C99 = $(CC) -std=c99
CFLAGS = -g -O3 -DNDEBUG -DDONT_USE_TEST_MAIN
CPPFLAGS = -g -O3 -fpermissive -DNDEBUG -DDONT_USE_TEST_MAIN
LDLIBS = -lstdc++ -lz -lm -ltiff

default: srtm4 srtm4_which_tile

Geoid.o: Geoid.cpp
	$(CXX) $(CPPFLAGS) -c $^ -o $@

geoid_height_wrapper.o: geoid_height_wrapper.cpp
	$(CXX) $(CPPFLAGS) -c $^ -o $@ -DGEOID_DATA_FILE_PATH="\"$(CURDIR)/data\""

srtm4: srtm4.c Geoid.o geoid_height_wrapper.o
	$(C99) $(CFLAGS) -DMAIN_SRTM4 $^ $(LDLIBS) -o $@

srtm4_which_tile: srtm4.c Geoid.o geoid_height_wrapper.o
	$(C99) $(CFLAGS) -DMAIN_SRTM4_WHICH_TILE $^ $(LDLIBS) -o $@

clean:
	-rm -f srtm4 srtm4_which_tile
	-rm -f  Geoid.o geoid_height_wrapper.o
	-rm -f -r srtm4.dSYM srtm4_which_tile.dSYM
