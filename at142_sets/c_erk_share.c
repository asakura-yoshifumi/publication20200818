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

/* exp preventing overflow */
double expo(double x){
  double out;
  /* check the input size */
  if(x < -30.0){
    out = 0.0;
  } else {
    out = exp(x);
  }
  /* out can be more than about 10^-9 */
  return out;
} /* expo */

/* calculate dot product of size 1 x 2, 2 x 2, 2 x 1 */
double dot121(double * l, double * c, double * r){
  double l0, l1, out;
  l0  = l[0] * c[0] + l[1] * c[2];
  l1  = l[0] * c[1] + l[1] * c[3];
  out = l0 * r[0] + l1 * r[1];
  return out;
}/* dot121 */

/* calculate one gauss */
double gauss_one(
  double   x,
  double   y,
  double * inv_sigma_theta,
  double   center_x,
  double   center_y,
  double   gauss_max
){
  double x_mu[2] = {(x - center_x), (y - center_y)};
  double out = gauss_max * expo(-0.5 * dot121(x_mu, inv_sigma_theta, x_mu));
  /**/
  return out;
} /* gauss_one */

/* calculate multiple gauss case */
double gauss_multi(
  double   t,
  double   x,
  double   y,
  double   lenx,
  double   leny,
  double   startx,
  double   starty,
  double * inv_sigma_theta,
  double   theta,
  double   gauss_max,
  double   speed
){
  double c_x, c_y, mx, my;
  int    i,   j;
  double out = 0.0;
  /* loops to add 3 x 3 */
  for(i = -1; i < 2; i++){
    for(j = -1; j < 2; j++){
      mx  = fmod(x, lenx);
      my  = fmod(y, leny);
      c_x = fmod(speed * t * cos(theta) - startx, lenx) + i * lenx;
      c_y = fmod(speed * t * sin(theta) - starty, leny) + j * leny;
      out += gauss_one(
        mx,
        my,
        inv_sigma_theta,
        c_x,
        c_y,
        gauss_max
      );
    }
  }
  return out;
} /* gauss_multi */

/* calculate the gauss multi on a array of x, y */
void gauss_array(
  double * out,
  int      num_cell,
  double   t,
  double * xy,
  double   lenx,
  double   leny,
  double   startx,
  double   starty,
  double * inv_sigma_theta,
  double   theta,
  double   speed,
  double   speed_grow
){
  int    i;
  double gauss_max = speed_grow * t;
  /**/
  if(gauss_max < 0.0){
    gauss_max = 0.0;
  } else if(1.0 < gauss_max){
    gauss_max = 1.0;
  }
  /**/
  for(i = 0; i < num_cell; i++){
    out[i] = gauss_multi(
      t,
      xy[2 * i],
      xy[2 * i + 1],
      lenx,
      leny,
      startx,
      starty,
      inv_sigma_theta,
      theta,
      gauss_max,
      speed
    );
  }
} /* gauss_array */
