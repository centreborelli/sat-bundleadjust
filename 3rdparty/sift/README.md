# sift_anatomy

Note: This repository contains two programs on separate folders

        original/   The original ``sift_anatomy'' code by Ives Rey Otero
        simd/       An optimized reimplementation by Marc Lebrun using AVX


## Overview

This C ANSI source code is related to the article

    [1] "Anatomy of the SIFT Method."
        I. Rey Otero  and  M. Delbracio
        Image Processing Online, 2013.
        http://dx.doi.org/10.5201/ipol.2014.82

An online demo facility can be found at http://dx.doi.org/10.5201/ipol.2014.82

## Patent Warning and License

The SIFT method is patented

    [2] "Method and apparatus for identifying scale invariant features in an image."
        David G. Lowe
        Patent number: 6711293
        Filing date: Mar 6, 2000
        Issue date: Mar 23, 2004
        Application number: 09/519,89

 These source codes are made available for the exclusive aim of serving as
 scientific tool to verify the soundness and completeness of the algorithm
 description. Compilation, execution and redistribution of this file may
 violate patents rights in certain countries. The situation being different
 for every country and changing over time, it is your responsibility to
 determine which patent rights restrictions apply to you before you compile,
 use, modify, or redistribute this file. A patent lawyer is qualified to make
 this determination. If and only if they don't conflict with any patent terms,
 you can benefit from the following license terms attached to this file.

This program is free software: you can use, modify and/or
redistribute it under the terms of the simplified BSD
License. You should have received a copy of this license along
this program. If not, see
<http://www.opensource.org/licenses/bsd-license.html>.


## Compiling (Linux) 

Type `make` in the directory where the Makefile is located. The compilation of the source code provides three executables:

1. `sift_cli`  applies the SIFT method to a PNG image. Its uses either standard
               parameters (as documented in [1]) user selected parameters.

2. `match_cli`   matches the SIFT keypoints extracted from two image.

3. `sift_cli_default`  applies the SIFT method to a PNG image. Only uses standard
                       parameters


## SIFT executable - Usage 


`./sift_cli image [options...]  [> keys]`

#### Options for scale-space computation and detection thresholds:
    -ss_noct        (8)  number of octaves
    -ss_nspo        (3)  number of scales per octaves
    -ss_dmin      (0.5)  the sampling distance in the first octave
    -ss_smin      (0.8)  blur level on the seed image
    -ss_sin       (0.5)  assumed level of blur in the input image

    -thresh_dog (0.0133) threshold over the DoG response
    -thresh_edge   (10)  threshold over the ratio of principal curvature

#### Options for orientation attribution:
    -ori_nbins    (36)   number of bins in the orientation histogram
    -ori_thresh  (0.8)   threhsold for considering local maxima in
                         the orientation histogram
    -ori_lambda  (1.5)   sets how local is the analysis of the gradient
                         distribution

#### Options for descriptor computation:
    -descr_nhist   (4)   number of histograms per dimension
    -descr_nori    (8)   number of bins in each histogram
    -descr_lambda  (6)   sets how local the descriptor is

#### Options for output verbosity 
    -verb_keys   label   flag to output the intermediary sets of keypoints
    -verb_ss     label   flag to output the scalespaces (Gaussian and DoG)



## MATCHING executable - Usage

`match_cli keys1 keys2 [options...]`

```
    -ori_nbins    (36)  number of bins in the orientation histogram (used only for keypoints input/output)
    -descr_nhist   (4)  number of histograms per dimension
    -descr_nori    (8)  number of bins in each histogram

    -absolute thresh (250) threshold applied on the euclidean distance
    -relative thresh (0.6) threshold applied on the ratio of  distance

    -verb         label  flag for output
```


The output is a list of matches with the following formatting
```
      x1  y1  sigma1  theta1   x2  y2  sigma2  theta 2
```
#### List of files produced by `match_cli` with the `'verb'` option

1. OUTmatches.txt          The pairs matches,
2. [label]_im0.txt         The subset of matching keypoints in the first image
3. [label]_im1.txt         The subset of matching keypoints in the second image

File 1) has the following formatting:

      key1  key2a  key2b

where (key1) designates a keypoint in image1, (key2a) and (key2b) designate
respectively the nearest and the second nearest neighbors in image 2.
The data relative to each keypoint is formatted as follows

     x  y  sigma  theta  fv[1] fv[2] ... fv[d] octave scale orihist[1] ... orihist[n_bins]

where (fv) is the feature vector of dimension d=n_hist*n_hist*n_ori  and(orihist) is the orientation histogram of n_bins bins.


## lib_sift.h 

File `lib_sift.h` provides a simplified interface to the sift library.

- To extract the keypoint from the SIFT scale-space.
   ```
   struct sift_keypoint_std* sift_compute_points(double* x, int w, int h, int* n);
   ```

- To compute the feature descriptors for oriented keypoints provided by the user:
   ```
   void sift_fill_descriptors(double *x, int w, int h, struct sift_keypoint_std *k, int n);
   ```
   
- To compute orientations and  feature descriptors for keypoints provided by the user:
   ```
   void sift_fill_descriptors(double *x, int w, int h, struct sift_keypoint_std *k, int n);
   ```

- To run the standard sift algorithm:
   ```
   struct sift_keypoint_std *sift_compute_features(double *x, int w, int h, int *n);
   ```
   
- For input/output:
   ```
   struct sift_keypoint_std *sift_read_from_file(char *filename, int *n);
   void sift_write_to_file(char *filename, struct sift_keypoint_std *k, int n);
   ```


###  How to link your code to  lib_sift.h  

These are the steps to follow in order to use the library lib_sift.h in a code.

1. add  `#include "lib_sift.h"`
2. compile object files: `lib_sift.o` `lib_sift_anatomy.o` `lib_scalespace.o` `lib_keypoint.o` `lib_description.o` `lib_discrete.o`
3. link

Here a two short examples of source code with their respective compilation commands.

#### Example 1 - Code
```
#include <stdlib.h>
#include "lib_sift.h"

int main(void)
{
	// create input image
	int w = 300;
	int h = 200;
	float *x = malloc(w*h*sizeof(*x));
	for (int i = 0; i < w*h; i++)
		x[i] = rand();

	// compute sift keypoints
	int n;
	struct sift_keypoint_std *k = sift_compute_points(x, w, h, &n);

	// write to standard output
	sift_write_to_file("/dev/stdout", k, n);

	// cleanup
	free(k);
	free(x);
	return 0;
}
```

#### Code Example 1 - compilation commands

```
gcc -std=c99  -c -o lib_keypoint.o lib_keypoint.c
gcc -std=c99  -c -o lib_discrete.o lib_discrete.c
gcc -std=c99  -c -o lib_scalespace.o lib_scalespace.c
gcc -std=c99  -c -o lib_sift_anatomy.o lib_sift_anatomy.c
gcc -std=c99  -c -o lib_description.o lib_description.c
gcc -std=c99  -c -o lib_sift.o lib_sift.c
gcc -std=c99  -c -o lib_util.o lib_util.c

gcc -std=c99 -o example example.c lib_sift.o lib_sift_anatomy.o 
             lib_keypoint.o  lib_scalespace.o lib_description.o \
             lib_discrete.o lib_util.o -lm
```

####  Code Example 2 - source

```
#include <stdlib.h>
#include <stdio.h>
#include "lib_sift.h"
#include "io_png.h"

int main(int argc, char **argv)
{
    if(arg != 2){
        fprintf(stderr, "usage:\n./exemple2 image\n");
        return -1;
    }

	// Loading image
	size_t w, h;
    float* x = io_png_read_f32_gray(argv[1], &w, &h);
    for(int i=0; i < w*h; i++)
        x[i] /=256.;

	// compute sift keypoints
	int n;
	struct sift_keypoint_std *k = sift_compute_features(x, w, h, &n);

	// write to standard output
	sift_write_to_file("/dev/stdout", k, n);

	// cleanup
	free(k);
	free(x);
	return 0;
}
```

#### Code Example

```
gcc -std=c99  -c -o lib_keypoint.o lib_keypoint.c
gcc -std=c99  -c -o lib_discrete.o lib_discrete.c
gcc -std=c99  -c -o lib_scalespace.o lib_scalespace.c
gcc -std=c99  -c -o lib_sift_anatomy.o lib_sift_anatomy.c
gcc -std=c99  -c -o lib_description.o lib_description.c
gcc -std=c99  -c -o lib_sift.o lib_sift.c
gcc -std=c99  -c -o lib_util.o lib_util.c
gcc -std=c99  -c -o io_png.o io_png.c

gcc -std=c99 -o example example.c lib_sift.o lib_sift_anatomy.o \
             lib_keypoint.o  lib_scalespace.o lib_description.o \
             lib_discrete.o lib_util.o io_png.o -lm -lpng
```


## Comparison with other implementations

The executable provided by D.Lowe (http://www.cs.ubc.ca/~lowe/keypoints/, retrieved on September 11th,2014) uses a different coordinate system. This results in different orientation and different feature vectors.

In Lowe's executable, the x component increases to the right and the y component increases upward and the coordinate system adopted in the description phase.

In this code, the x component increases downward and the y component increases to the right. This is consistent with the coordinate system used during detection.

A conversion tool is provided in the source called anatomy2lowe.c to convert
To compile this tool
```
gcc -o anatomy2lowe anatomy2lowe.c -std=c99
```


## Generating the doxygen
To generate the documentation, type in the src/ directory :
```
   doxygen -g
   doxygen Doxyfile
```
doxygen documentation is in directory ./html/


## Acknowledgements

Work partially supported by
Centre National d’Etudes Spatiales (CNES, MISS Project),
European Research Council (Advanced Grant Twelve Labours),
Office of Naval Research (Grant N00014-97-1-0839),
Direction Generale de l’Armement (DGA),
Fondation Mathematique Jacques Hadamard,
Agence Nationale de la Recherche (Stereo project).
The author would like to thank Enric Meinhardt-Llopis for fruitful comments and discussions.

