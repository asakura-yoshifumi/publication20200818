/*
MIT License

Copyright (c) 2021 Yoshifumi Asakura

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include "libqhull_r/qhull_ra.h"

#define DIM 2

/*
record pairs of point ids, that constitute ridges on voronoi
only used in C
*/
void ridge_pair(qhT *qh, int *out_arr, vertexT *atvertex, vertexT *vertex, setT *center, boolT unbounded){
  /*
  put pointid ints on the out_arr of int
    (1 + num_point * num_point) x 2 in virtual shape, but 1_D array here
    rows
      0 only      count of ridges already visited
      1 or after  point id pair
    columns
      0 id of atvertex
      1 id of vertex
    out_arr[1] is never changed
  */
  int ind_left;
  int ind_right;
  int tmp;
  /* count this ridge as already visited */
  out_arr[0] += 1;
  tmp = out_arr[0];
  /* 0th pair will set on 1st row, j th pair on j+1 th row. */
  ind_left  = 2 * tmp;
  ind_right = 2 * tmp + 1;
  /* set point id */
  out_arr[ind_left]  = qh_pointid(qh, atvertex->point);
  out_arr[ind_right] = qh_pointid(qh,   vertex->point);
} /* ridge_pair */


/*
find neighboring pairs in the in_arr, coding 2 dimensions points
in 1D array, with length (number of points) x 2.
regarding point i,
    x coordinate at 2 x i
    y coordinate at 2 x i + 1

out_arr is a 1D array to record output,
coding 2 dimensions points with a hedder column.
in 1D array,
    0                   ... number of pairs
    1                   ... always 0
    and after that, with a integer n,
    2 x n and 2 x n + 1 ... indice of points that are neighbors

call this function from cython
*/
void neighbor_pairs(double *in_arr, int *out_arr, int TOTpoints) {

  boolT ismalloc;
  int curlong, totlong, exitcode;
  char options [2000];
  qhT qh_qh;
  qhT *qh= &qh_qh;
  int j, k, count_in = 0;

  QHULL_LIB_CHECK

  ismalloc= False;


  /*printf("%d points\n", TOTpoints);*/

  qh_init_A(qh, stdin, stdout, stderr, 0, NULL);
  exitcode= setjmp(qh->errexit);

  if (!exitcode) {
    coordT array[TOTpoints][DIM];

    qh->NOerrexit= False;
    sprintf(options, "qhull v Qbb Qc Qz");
    qh_initflags(qh, options);

    qh->PROJECTdelaunay = True;

    /* prepare input into array 2D */
    for(j = 0; j < TOTpoints; j++){
      for(k = 0; k < 2; k++){
        //array[j][k] = in_arr[j][k];
        array[j][k] = in_arr[count_in];
        count_in++;
        /*printf("  %d %d %d %6.2f", j, k, count_in, array[j][k]);*/
      }
      /*printf("\n");*/
    }
    /*printf("\n");*/
    qh_setdelaunay(qh, qh->hull_dim, qh->num_points, qh->first_point);

    qh_init_B(qh, array[0], TOTpoints, DIM, ismalloc);
    qh_qhull(qh); /* necessary, otherwise segmentation fault below. */

    qh_setvoronoi_all(qh);

    qh_check_output(qh);
    //print_summary(qh);

    qh_eachvoronoi_all(qh, out_arr, &ridge_pair, qh->UPPERdelaunay, qh_RIDGEall, 1);

    if (qh->VERIFYoutput && !qh->FORCEoutput && !qh->STOPadd && !qh->STOPcone && !qh->STOPpoint)
      qh_check_points(qh);
    fflush(NULL);

  }
  qh->NOerrexit= True;
#ifdef qh_NOmem
  qh_freeqhull(qh, qh_ALL);
#else
  qh_freeqhull(qh, !qh_ALL);
  qh_memfreeshort(qh, &curlong, &totlong);
  if (curlong || totlong)
    fprintf(stderr, "qhull warning (user_eg2, run 2): did not free %d bytes of long memory (%d pieces)\n",
         totlong, curlong);
#endif

} /* neighbor_pairs */
