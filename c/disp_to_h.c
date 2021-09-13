#include <stdbool.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>

#include "rpc.h"
#include "fail.c"


static void apply_homography(double y[2], double h[9], double x[2])
{
    //                    h[0] h[1] h[2]
    // The convention is: h[3] h[4] h[5]
    //                    h[6] h[7] h[8]
    double z = h[6]*x[0] + h[7]*x[1] + h[8];
    double tmp = x[0];  // to enable calls like 'apply_homography(x, h, x)'
    y[0] = (h[0]*x[0] + h[1]*x[1] + h[2]) / z;
    y[1] = (h[3]*tmp  + h[4]*x[1] + h[5]) / z;
}


static double invert_homography(double o[9], double i[9])
{
    double det = i[0]*i[4]*i[8] + i[2]*i[3]*i[7] + i[1]*i[5]*i[6]
               - i[2]*i[4]*i[6] - i[1]*i[3]*i[8] - i[0]*i[5]*i[7];
    o[0] = (i[4]*i[8] - i[5]*i[7]) / det;
    o[1] = (i[2]*i[7] - i[1]*i[8]) / det;
    o[2] = (i[1]*i[5] - i[2]*i[4]) / det;
    o[3] = (i[5]*i[6] - i[3]*i[8]) / det;
    o[4] = (i[0]*i[8] - i[2]*i[6]) / det;
    o[5] = (i[2]*i[3] - i[0]*i[5]) / det;
    o[6] = (i[3]*i[7] - i[4]*i[6]) / det;
    o[7] = (i[1]*i[6] - i[0]*i[7]) / det;
    o[8] = (i[0]*i[4] - i[1]*i[3]) / det;
    return det;
}


void stereo_corresp_to_lonlatalt(double *lonlatalt, float *err,  // outputs
                                 float *kp_a, float *kp_b, int n_kp,  // inputs
                                 struct rpc *rpc_a, struct rpc *rpc_b)
{

    // intermediate buffers
    double lonlat[2];
    double e, z;

    // loop over all matches
    // a 3D point is produced for each match
    for (int i = 0; i < n_kp; i++) {

        // compute (lon, lat, alt) of the 3D point
        z = rpc_height(rpc_a, rpc_b, kp_a[2 * i], kp_a[2 * i + 1],
                       kp_b[2 * i], kp_b[2 * i + 1], &e);
        eval_rpc(lonlat, rpc_a, kp_a[2 * i], kp_a[2 * i + 1], z);

        // store the output values
        lonlatalt[3 * i + 0] = lonlat[0];
        lonlatalt[3 * i + 1] = lonlat[1];
        lonlatalt[3 * i + 2] = z;
        err[i] = e;
    }
}


void disp_to_lonlatalt(double *lonlatalt, float *err,  // outputs
                 float *dispx, float *dispy, float *msk, int nx, int ny,  // inputs
                 double ha[9], double hb[9],
                 struct rpc *rpca, struct rpc *rpcb,
                 float orig_img_bounding_box[4])
{
    // invert homographies
    double ha_inv[9];
    double hb_inv[9];
    invert_homography(ha_inv, ha);
    invert_homography(hb_inv, hb);

    // read image bounding box
    float col_min = orig_img_bounding_box[0];
    float col_max = orig_img_bounding_box[1];
    float row_min = orig_img_bounding_box[2];
    float row_max = orig_img_bounding_box[3];

    // initialize output images to nan
    for (int row = 0; row < ny; row++)
    for (int col = 0; col < nx; col++) {
        int pix = col + nx*row;
        err[pix] = NAN;
        for (int k = 0; k < 3; k++)
            lonlatalt[3 * pix + k] = NAN;
    }

    // intermediate buffers
    double p[2], q[2], lonlat[2];
    double e, z;

    // loop over all the pixels of the input disp map
    // a 3D point is produced for each non-masked disparity
    for (int row = 0; row < ny; row++)
    for (int col = 0; col < nx; col++) {
        int pix = col + nx*row;
        if (!msk[pix])
            continue;

        // compute coordinates of pix in the full reference image
        double a[2] = {col, row};
        apply_homography(p, ha_inv, a);

        // check that it lies in the image domain bounding box
        if (round(p[0]) < col_min || round(p[0]) > col_max ||
            round(p[1]) < row_min || round(p[1]) > row_max)
            continue;

        // compute (lon, lat, alt) of the 3D point
        double dx = dispx[pix];
        double dy = dispy[pix];
        double b[2] = {col + dx, row + dy};
        apply_homography(q, hb_inv, b);
        z = rpc_height(rpca, rpcb, p[0], p[1], q[0], q[1], &e);
        eval_rpc(lonlat, rpca, p[0], p[1], z);

        // store the output values
        lonlatalt[3 * pix + 0] = lonlat[0];
        lonlatalt[3 * pix + 1] = lonlat[1];
        lonlatalt[3 * pix + 2] = z;
        err[pix] = e;
    }
}


float squared_distance_between_3d_points(double a[3], double b[3])
{
    float x = (a[0] - b[0]);
    float y = (a[1] - b[1]);
    float z = (a[2] - b[2]);
    return x*x + y*y + z*z;
}


void count_3d_neighbors(int *count, double *xyz, int nx, int ny, float r, int p)
{
    // count the 3d neighbors of each point
    for (int y = 0; y < ny; y++)
    for (int x = 0; x < nx; x++) {
        int pos = x + nx * y;
        double *v = xyz + pos * 3;
        int c = 0;
        int i0 = y > p ? -p : -y;
        int i1 = y < ny - p ? p : ny - y - 1;
        int j0 = x > p ? -p : -x;
        int j1 = x < nx - p ? p : nx - x - 1;
        for (int i = i0; i <= i1; i++)
        for (int j = j0; j <= j1; j++) {
            double *u = xyz + (x + j + nx * (y + i)) * 3;
            float d = squared_distance_between_3d_points(u, v);
            if (d < r*r) {
                c++;
            }
        }
        count[pos] = c;
    }
}


void remove_isolated_3d_points(
    double* xyz,  // input (and output) image, shape = (h, w, 3)
    int nx,      // width w
    int ny,      // height h
    float r,     // filtering radius, in meters
    int p,       // filtering window (square of width is 2p+1 pixels)
    int n,       // minimal number of neighbors to be an inlier
    int q)       // neighborhood for the saving step (square of width 2q+1)
{
    int *count = (int*) malloc(nx * ny * sizeof(int));
    bool *rejected = (bool*) malloc(nx * ny * sizeof(bool));

    // count the 3d neighbors of each point
    count_3d_neighbors(count, xyz, nx, ny, r, p);

    // brutally reject any point with less than n neighbors
    for (int i = 0; i < ny * nx; i++)
        rejected[i] = count[i] < n;

    // show mercy; save points with at least one close and non-rejected neighbor
    bool need_more_iterations = true;
    while (need_more_iterations) {
        need_more_iterations = false;
        // scan the grid and stop at rejected points
        for (int y = 0; y < ny; y++)
        for (int x = 0; x < nx; x++)
        if (rejected[x + y * nx])
        // explore the neighborhood (square of width 2q+1)
        for (int yy = y - q; yy < y + q + 1; yy++) {
            if (yy < 0) continue; else if (yy > ny-1) break;
            for (int xx = x - q; xx < x + q + 1; xx++) {
                if (xx < 0) continue; else if (xx > nx-1) break;
                // is the current rejected point's neighbor non-rejected?
                if (!rejected[xx + yy * nx])
                // is this connected neighbor close (in 3d)?
                if (squared_distance_between_3d_points(
                        xyz + (x + y * nx)*3, xyz + (xx + yy * nx)*3) < r*r) {
                    rejected[x + y * nx] = false; // save the point
                    yy = xx = ny + nx + 2*q + 2;  // break loops over yy and xx
                    need_more_iterations = true;  // this point may save others!
                }
            }
        }
    }

    // set to NAN the rejected pixels
    for (int i = 0; i < ny * nx; i++)
        if (rejected[i])
            for (int c = 0; c < 3; c++)
                xyz[c + i * 3] = NAN;

    free(rejected);
    free(count);
}


static void help(char *s)
{
    fprintf(stderr, "usage:\n\t"
            "%s rpc_ref.xml rpc_sec.xml disp.tif heights.tif err.tif "
            "[--mask mask.png] "
            "[-href \"h1 ... h9\"] [-hsec \"h1 ... h9\"] "
            "[--mask-orig msk.png] "
            "[--col-m x0] [--col-M xf] [--row-m y0] [--row-M yf]\n", s);
}

