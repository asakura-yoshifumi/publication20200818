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

#include <iostream>
#include <boost/math/distributions/skew_normal.hpp>


class max_record {
  // no private
public:
  int    num_grid;
  double width_sig;
  double dmax;
  double inv_max;
  int    search_yet;
         max_record();
  double get_inv_max(
    boost::math::skew_normal_distribution<double> snd,
    double mu,
    double sigma
  );
};

max_record::max_record(){
  num_grid   = 400;
  width_sig  = 10.0;
  search_yet = 1;
  dmax       = 0.0;
  inv_max    = 1.0;
}

double max_record::get_inv_max(
    boost::math::skew_normal_distribution<double> snd,
    double mu,
    double sigma
){
  if(sigma <= 0.0){
    std::cout << ">>> error, sigma <= 0" << std::endl;
    return(0.0);
  }
  if(search_yet){
    search_yet  = 0;
    double cur;
    double x;
    double xmin = mu - sigma * width_sig;
    double xmax = mu + sigma * width_sig;
    double dx   = (xmax - xmin) / num_grid;
    for(x = xmin; x <= xmax; x += dx){
      cur = boost::math::pdf(snd, x);
      if(dmax < cur){
        dmax = cur;
      }
    }
    if(dmax == 0.0){
      inv_max = 0.0;
      std::cout << ">>> error, max of the distribusion is 0" << std::endl;
    } else {
      inv_max = 1.0 / dmax;
    }
    std::cout << ">>> " << dmax << " " << inv_max << std::endl;
  }
  return(inv_max);
}

// use this in skew_normal_1d
max_record mr;



double skew_normal_1d(
  double   mu_t,
  double   x,
  double   sigma_inner,
  double   startx,
  double   gauss_max,
  double   skew_shape
){
  // set sigma as the wave length get similar to the normal gaussian
  double sigma = sigma_inner;


  // a constant parameter
  double alpha = skew_shape * sigma;


  double mu    = mu_t + startx;

  // use below in pdf
  boost::math::skew_normal_distribution<double> snd(mu, sigma, alpha);

  double ratio = mr.get_inv_max(snd, mu, sigma);

  double out   = boost::math::pdf(snd, x) * gauss_max * ratio;
  return(out);
}
