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

#include <stdio.h>
#include <math.h>

// weno implementation for cython in c

void num_flux_k3_1d_c(
  double *vc,
  int     left,
  double *out,
  int     size
){
  double dr0     = 3.0 / 10.0;
  double dr1     = 6.0 / 10.0;
  double dr2     = 1.0 / 10.0;
  double epsilon = pow(10.0, -6.0);
  //
  double vl2[size];
  double vl1[size];
  double vr1[size];
  double vr2[size];
  //
  double alpha0[size];
  double alpha1[size];
  double alpha2[size];
  double sum_alpha[size];
  //
  double beta0[size];
  double beta1[size];
  double beta2[size];
  //
  double weight0[size];
  double weight1[size];
  double weight2[size];
  //
  double polys[3][size];
  double mat[4][3];
  //
  int    j;
  int    k;
  int    l;
  int    m;
  int    n;
  //
  double vec[3][3][size];
  //
  // stencils
  vl2[0] = vc[(size - 2)];
  vl1[0] = vc[(size - 1)];
  vr1[0] = vc[1];
  vr2[0] = vc[2];
  vl2[1] = vc[(size - 1)];
  vl1[1] = vc[0];
  vr1[1] = vc[2];
  vr2[1] = vc[3];
  for(j = 2; j < (size - 2); j++){
    vl2[j] = vc[(j - 2)];
    vl1[j] = vc[(j - 1)];
    vr1[j] = vc[(j + 1)];
    vr2[j] = vc[(j + 2)];
  }
  vl2[(size - 2)] = vc[(size - 4)];
  vl1[(size - 2)] = vc[(size - 3)];
  vr1[(size - 2)] = vc[(size - 1)];
  vr2[(size - 2)] = vc[0];
  vl2[(size - 1)] = vc[(size - 3)];
  vl1[(size - 1)] = vc[(size - 2)];
  vr1[(size - 1)] = vc[0];
  vr2[(size - 1)] = vc[1];
  // polynomials prep
  mat[0][0] = 2.0;
  mat[0][1] = -7.0;
  mat[0][2] = 11.0;
  mat[1][0] = -1.0;
  mat[1][1] = 5.0;
  mat[1][2] = 2.0;
  mat[2][0] = 2.0;
  mat[2][1] = 5.0;
  mat[2][2] = -1.0;
  mat[3][0] = 11.0;
  mat[3][1] = -7.0;
  mat[3][2] = 2.0;
  for(j = 0; j < 4; j++){
    for(k = 0; k < 3; k++){
      mat[j][k] = mat[j][k] / 6.0;
    }
  }
  for(j = 0; j < size; j++){
    vec[0][0][j] = vl2[j];
    vec[0][1][j] = vl1[j];
    vec[0][2][j] = vc[j];
    vec[1][0][j] = vl1[j];
    vec[1][1][j] = vc[j];
    vec[1][2][j] = vr1[j];
    vec[2][0][j] = vc[j];
    vec[2][1][j] = vr1[j];
    vec[2][2][j] = vr2[j];
  }
  if(left){
    n = 0;
  }else{
    n = 1;
  }
  // polys
  for(l = 0; l < 3; l++){
    for(j = 0; j < size; j++){
      polys[l][j] = 0.0;
    }
  }
  for(l = 0; l < 3; l++){
    for(m = 0; m < 3; m++){
      for(j = 0; j < size; j++){
        polys[l][j] += mat[(l + n)][m] * vec[l][m][j];
      }
    }
  }
  //
  for(j = 0; j < size; j++){
    // beta, smooth indicator
    beta0[j] = (13.0 / 12.0) * pow((vl2[j] - 2.0 * vl1[j] +  vc[j]), 2.0) + 0.25 * pow((vl2[j]      - 4.0 * vl1[j] + 3.0 *  vc[j]), 2.0);
    beta1[j] = (13.0 / 12.0) * pow((vl1[j] - 2.0 *  vc[j] + vr1[j]), 2.0) + 0.25 * pow((vl1[j]                           - vr1[j]), 2.0);
    beta2[j] = (13.0 / 12.0) * pow(( vc[j] - 2.0 * vr1[j] + vr2[j]), 2.0) + 0.25 * pow((3.0 * vc[j] - 4.0 * vr1[j] +       vr2[j]), 2.0);
    // alpha, weights
    alpha0[j] = dr0 * pow((epsilon + beta0[j]), -2.0);
    alpha1[j] = dr1 * pow((epsilon + beta1[j]), -2.0);
    alpha2[j] = dr2 * pow((epsilon + beta2[j]), -2.0);
    sum_alpha[j] = alpha0[j] + alpha1[j] + alpha2[j];
    //
    weight0[j] = alpha0[j] / sum_alpha[j];
    weight1[j] = alpha1[j] / sum_alpha[j];
    weight2[j] = alpha2[j] / sum_alpha[j];
    //
    out[j] = weight0[j] * polys[0][j] + weight1[j] * polys[1][j] + weight2[j] * polys[2][j];
  }
}


void weno_1d_c(
  double *y_left,
  double *y_right,
  double  dx,
  double *out,
  int     size
){
  double h_left[size];
  double h_right[size];
  int    j;
  // left
  num_flux_k3_1d_c(y_left,  1, h_left,  size);
  // right
  num_flux_k3_1d_c(y_right, 0, h_right, size);
  //
  // output
  j = 0;
  out[j]   = (h_left[j] - h_left[(size - 1)] + h_right[j] - h_right[(size - 1)]) / dx;
  for(j = 1; j < size; j++){
    out[j] = (h_left[j] - h_left[(j - 1)]    + h_right[j] - h_right[(j - 1)])    / dx;
  }
}
