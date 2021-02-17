# distutils:
# distutils:

################################################################################
# MIT License
#
# Copyright (c) 2021 Yoshifumi Asakura
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
################################################################################


################
### import
################
cimport cython
cimport numpy as np
import numpy as np

from numpy cimport ndarray

from cpython.mem cimport PyMem_Malloc, PyMem_Free
from libc.stdlib cimport malloc, free, abort

from cython.parallel cimport prange

from libc.math cimport pow, sin, cos, pi


################
### ERK
################

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef np.ndarray[np.float64_t, ndim = 2] erk_dt_2d(
    double            t,
    np.ndarray[np.float64_t, ndim = 3] X,
    np.ndarray[np.float64_t, ndim = 2] sigma_inner,
    double            speed1d,
    double            speed_grow,
    double            start_point
):
    cdef:
        int j
        int k
        int len0 = X.shape[0]
        int len1 = X.shape[1]
        double sigmax = sigma_inner[0, 0]
        double sigmay = sigma_inner[1, 1]
        double centx
        double centy
        double Amp_t
        double Amp_tdt = 0.0
    #
    centx = speed1d * t + start_point
    centy = np.mean(X[1, 0, :]); #print(centy)
    #
    Amp_t   = t * speed_grow
    if Amp_t <= 0.0:
        ### to discard
        Amp_t = 0.0
        #
    elif 1.0 <= Amp_t:
        ### large case
        Amp_t   = 1.0
        Amp_tdt = 0.0
        #
    else:
        ### the Amp_t is unnecessary to manipulate.
        Amp_tdt = speed_grow
        #
    ### here, Amp_t_speed is needed,
    Amp_t = Amp_t * speed1d
    #
    return(
        np.exp(
            -0.5 * (
                sigmax * (X[0, :, :] - centx)**2.0 + sigmay * (X[1, :, :] - centy)**2.0
            )
        ) * (Amp_tdt + Amp_t * sigmax * (X[0, :, :] - centx))
    )
    #
    #

cdef extern from "c_erk_share.h":
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
    ) nogil
    #

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef double erk_multi_c(
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
) nogil:
    return(gauss_multi(
        t,
        x,
        y,
        lenx,
        leny,
        startx,
        starty,
        inv_sigma_theta,
        theta,
        gauss_max,
        speed
    ))
    #

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef np.ndarray[np.float64_t, ndim = 2] erk_shared(
    double            t,
    double[:, :, ::1] X,
    double *          inv_sigma_theta,
    double            speed1d,
    double            speed_grow,
    double[:]         start_point,
    double[:]         fieldlim,
    double            theta
):
    cdef:
        int              j, k
        double           gauss_max = t * speed_grow
        double *        _out = <double *> PyMem_Malloc(X.shape[1] * X.shape[2] * sizeof(double))
        double[:, ::1] __out = <double[:X.shape[1], :X.shape[2]]> _out
        #
    try:
        ### max value
        if gauss_max < 0.0:
            gauss_max = 0.0
        elif 1.0 < gauss_max:
            gauss_max = 1.0
        #
        ### run erk_multi_c one by one
        for j in prange(0, X.shape[1], nogil = True, schedule = 'static', chunksize = 1):
            for k in range(0, X.shape[2]):
                __out[j, k] = erk_multi_c(
                    t,
                    X[0, j, k],
                    X[1, j, k],
                    fieldlim[0],
                    fieldlim[1],
                    start_point[0],
                    start_point[1],
                    inv_sigma_theta,
                    theta,
                    gauss_max,
                    speed1d
                )
            #
        return(np.array(__out))
        #
    finally:
        PyMem_Free(_out)
        #
    #
    #

cpdef erk_2D_inc(
    double            t,
    double[:, ::1]    X,
    double[:, ::1]    inv_sigma_theta,
    double            speed1d,
    double            speed_grow,
    double[:]         start_point,
    double[:]         fieldlim,
    double            theta
):
    cdef:
        int              j
        double           gauss_max = t * speed_grow
        double *        _tmp = <double *> PyMem_Malloc(4 * sizeof(double))
        double[:, ::1] __tmp = <double[:2, :2]> _tmp
        double *        _out = <double *> PyMem_Malloc(X.shape[0] * sizeof(double))
        double[:]      __out = <double[:X.shape[0]]> _out
        #
    try:
        __tmp[:, :] = inv_sigma_theta
        ### max value
        if gauss_max < 0.0:
            gauss_max = 0.0
        elif 1.0 < gauss_max:
            gauss_max = 1.0
        #
        ### run erk_multi_c one by one
        for j in prange(0, X.shape[0], nogil = True, schedule = 'static', chunksize = 1):
                __out[j] = erk_multi_c(
                    t,
                    X[j, 0],
                    X[j, 1],
                    fieldlim[0],
                    fieldlim[1],
                    start_point[0],
                    start_point[1],
                    _tmp,
                    theta,
                    gauss_max,
                    speed1d
                )
            #
        return(np.array(__out))
        #
    finally:
        PyMem_Free(_out)
        #

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cpdef np.ndarray[np.float64_t, ndim = 3] ode_reader(
    np.ndarray[np.float64_t, ndim = 3] odeout,
    int t,
    int x,
    int y
):
    return(odeout[t, :].reshape((3, x, y)))
    #
    #


################
### WENO, use c file
################
cdef extern from "c_weno1d.h":
    ### weno
    void weno_1d_c(
      double *y_left,
      double *y_right,
      double  dx,
      double *out,
      int     size
    ) nogil

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef void weno_1d(
    double * y_left,
    double * y_right,
    double   dx,
    double * out,
    int      size
) nogil:
    weno_1d_c(y_left, y_right, dx, out, size)
    #


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef np.ndarray[np.float64_t, ndim = 2] weno_2d(
    np.ndarray[np.float64_t, ndim = 2] y_left,
    np.ndarray[np.float64_t, ndim = 2] y_right,
    double                             dx
):
    '''
    weno 2d, in 0th direction of input,
    transpose the result of this function if 1st direction is needed.
    #
    parallel implementation to run weno_1d here,
    to calculate finite difference manner of approximation
    '''
    cdef:
        int                                 int0     = y_left.shape[0]
        int                                 int1     = y_left.shape[1]
        #
        double *                           _out      = <double *> PyMem_Malloc(y_left.size * sizeof(double))
        double[:, ::1]                    __out      = <double[:int1, :int0]> _out
        np.ndarray[np.float64_t, ndim = 2]  out      = np.asarray(__out)
        double *                           _y_left2  = <double *> PyMem_Malloc(y_left.size * sizeof(double))
        double[:, ::1]                    __y_left2  = <double[:int1, :int0]> _y_left2
        np.ndarray[np.float64_t, ndim = 2]  y_left2  = np.asarray(__y_left2)
        double *                           _y_right2 = <double *> PyMem_Malloc(y_left.size * sizeof(double))
        double[:, ::1]                    __y_right2 = <double[:int1, :int0]> _y_right2
        np.ndarray[np.float64_t, ndim = 2]  y_right2 = np.asarray(__y_right2)
        #
        int                                 j
        #
    try:
        ### ATTETION, transposed for memory continuity in C manner
        y_left2[:, :]  = y_left.T
        y_right2[:, :] = y_right.T
        #
        ### parallel
        for j in prange(int1, nogil = True, schedule = 'static', chunksize = 1):
            ### preparation
            #
            ### run
            weno_1d(
                #_tmp1d_l,
                <double *> (<size_t> _y_left2  + (j * int0) * sizeof(double)),
                #_tmp1d_r,
                <double *> (<size_t> _y_right2 + (j * int0) * sizeof(double)),
                dx,
                #_tmp1d_o,
                <double *> (<size_t> _out      + (j * int0) * sizeof(double)),
                int0
            )
        #
        ### ATTENTION, out is output in tranposed shape
        return(np.array(__out).T)
        #
    finally:
        PyMem_Free(_out)
        PyMem_Free(_y_left2)
        PyMem_Free(_y_right2)
        #
    #


################
### diff over space
################
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef np.ndarray[np.float64_t, ndim = 2] di1(
    np.ndarray[np.float64_t, ndim = 2] field,
    int                                direction,
    double                             dx
):
    cdef:
        double *         _one_posi = <double *> PyMem_Malloc(field.size * sizeof(double))
        double[:, ::1]  __one_posi = <double[:field.shape[0], :field.shape[1]]> _one_posi
        double *         _one_nega = <double *> PyMem_Malloc(field.size * sizeof(double))
        double[:, ::1]  __one_nega = <double[:field.shape[0], :field.shape[1]]> _one_nega
        double *         _out      = <double *> PyMem_Malloc(field.size * sizeof(double))
        double[:, ::1]  __out      = <double[:field.shape[0], :field.shape[1]]> _out
        np.ndarray[np.float64_t, ndim = 2] one_posi = np.asarray(__one_posi)
        np.ndarray[np.float64_t, ndim = 2] one_nega = np.asarray(__one_nega)
        np.ndarray[np.float64_t, ndim = 2] out      = np.asarray(__out)
        #
    ### choose next indice
    one_posi[:, :] = np.roll(field, -1, axis = direction)
    one_nega[:, :] = np.roll(field,  1, axis = direction)
    #
    out[:, :] = (one_posi - one_nega) / (2.0 * dx)
    #
    try:
        return(np.array(__out))
    finally:
        PyMem_Free(_one_posi)
        PyMem_Free(_one_nega)
        PyMem_Free(_out)
    #
    #

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef np.ndarray[np.float64_t, ndim = 2] di2(
    np.ndarray[np.float64_t, ndim = 2] field,
    int                                direction,
    double                             dx
):
    cdef:
        double *         _one_posi = <double *> PyMem_Malloc(field.size * sizeof(double))
        double[:, ::1]  __one_posi = <double[:field.shape[0], :field.shape[1]]> _one_posi
        double *         _one_nega = <double *> PyMem_Malloc(field.size * sizeof(double))
        double[:, ::1]  __one_nega = <double[:field.shape[0], :field.shape[1]]> _one_nega
        double *         _out      = <double *> PyMem_Malloc(field.size * sizeof(double))
        double[:, ::1]  __out      = <double[:field.shape[0], :field.shape[1]]> _out
        np.ndarray[np.float64_t, ndim = 2] one_posi = np.asarray(__one_posi)
        np.ndarray[np.float64_t, ndim = 2] one_nega = np.asarray(__one_nega)
        np.ndarray[np.float64_t, ndim = 2] out      = np.asarray(__out)
        #
    ### choose next indice
    one_posi[:, :] = np.roll(field, -1, axis = direction)
    one_nega[:, :] = np.roll(field,  1, axis = direction)
    #
    out[:, :] = (one_posi - 2.0 * field + one_nega) * dx ** (-2.0)
    #
    try:
        return(np.array(__out))
    finally:
        PyMem_Free(_one_posi)
        PyMem_Free(_one_nega)
        PyMem_Free(_out)
    #
    #


################
### functions in models, but put outside of the class to get them into WENO
### add options (type = list) as input that were hiddenly pulled in class
###
### implement...
###     vx
###     vx diff
###     vx source
###
###     vy
###     vy diff
###     vy source
###
###     rho
###     rho diff
###
###     TEST_ZERO_nopt
###     TEST_ZERO_wopt
###
### cython class methods seems not accepted to be locate in other functions with self
################
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef np.ndarray[np.float64_t, ndim = 2] adv_mx_fx(
    np.ndarray[np.float64_t, ndim = 3] y
):
    '''
    takes vx, vy, rho, and returns momentum
    '''
    return((y[0, :, :] ** 2.0) * y[2, :, :])
    #
    #

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef np.ndarray[np.float64_t, ndim = 2] adv_mx_gy(
    np.ndarray[np.float64_t, ndim = 3] y
):
    '''
    takes vx, vy, rho, and returns momentum
    '''
    return(y[0, :, :] * y[1, :, :] * y[2, :, :])
    #
    #

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef np.ndarray[np.float64_t, ndim = 2] adv_mx_fx_diff(
    np.ndarray[np.float64_t, ndim = 3] y
):
    '''
    takes vx, vy, rho, and returns momentum diff
    '''
    return(y[0, :, :])
    #
    #


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef np.ndarray[np.float64_t, ndim = 2] adv_mx_gy_diff(
    np.ndarray[np.float64_t, ndim = 3] y
):
    '''
    takes vx, vy, rho, and returns momentum diff
    '''
    return(y[1, :, :])
    #
    #


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef np.ndarray[np.float64_t, ndim = 2] adv_my_fx(
    np.ndarray[np.float64_t, ndim = 3] y
):
    '''
    takes vx, vy, rho, and returns momentum
    '''
    return(y[0, :, :] * y[1, :, :] * y[2, :, :])
    #
    #

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef np.ndarray[np.float64_t, ndim = 2] adv_my_gy(
    np.ndarray[np.float64_t, ndim = 3] y
):
    '''
    takes vx, vy, rho, and returns momentum
    '''
    return((y[1, :, :] ** 2.0) * y[2, :, :])
    #
    #

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef np.ndarray[np.float64_t, ndim = 2] adv_my_fx_diff(
    np.ndarray[np.float64_t, ndim = 3] y
):
    '''
    takes vx, vy, rho, and returns momentum diff
    '''
    return(y[0, :, :])
    #
    #


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef np.ndarray[np.float64_t, ndim = 2] adv_my_gy_diff(
    np.ndarray[np.float64_t, ndim = 3] y
):
    '''
    takes vx, vy, rho, and returns momentum diff
    '''
    return(y[1, :, :])
    #
    #


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef np.ndarray[np.float64_t, ndim = 2] source_momentum(
    np.ndarray[np.float64_t, ndim = 2] y,
    list                               options
):
    '''
    source term of Vx
    options are...
        0 paras        list, [mu0, alpha, beta, k0, R0, eta, r]
        1 dx           double
        2 ERK          np.ndarray[np.float64_t, ndim = 2]
        3 rho          np.ndarray[np.float64_t, ndim = 2]
        4 model_indice list [0, 1, 2, 3, 4], [1, 2, 3, 4, 5] or [1, 2, 3, 4, 6]
        5 ERKdt        np.ndarray[np.float64_t, ndim = 2]
    #
    depends on di"n", n = 1 or 2,
        di"n"(
            np.ndarray[np.float64_t, ndim = 2] field,
            int                                direction,
            double                             dx
        )
    '''
    cdef:
        ### this gives dynamics of Vx
        double *         _vx        = <double *> PyMem_Malloc(y.size * sizeof(double))
        double[:, ::1]  __vx        = <double[:y.shape[0], :y.shape[1]]> _vx
        np.ndarray[np.float64_t, ndim = 2] vx = np.asarray(__vx)
        ### options given
        double *         _ERK       = <double *> PyMem_Malloc(y.size * sizeof(double))
        double[:, ::1]  __ERK       = <double[:y.shape[0], :y.shape[1]]> _ERK
        double *         _rho       = <double *> PyMem_Malloc(y.size * sizeof(double))
        double[:, ::1]  __rho       = <double[:y.shape[0], :y.shape[1]]> _rho
        #
        list                               paras     = options[0]
        double                             dx        = options[1]
        np.ndarray[np.float64_t, ndim = 2] ERK       = np.asarray(__ERK)
        np.ndarray[np.float64_t, ndim = 2] rho       = np.asarray(__rho)
        list                               mod_ind   = options[4]
        ### in paras
        double                             mu0       = paras[0]
        double                             alpha     = paras[1]
        double                             beta      = paras[2]
        double                             k0        = paras[3]
        double                             R0        = paras[4]
        double                             eta       = paras[5]
        double                             r         = paras[6]
        ### diff
        double *         _dERK_dx   = <double *> PyMem_Malloc(y.size * sizeof(double))
        double[:, ::1]  __dERK_dx   = <double[:y.shape[0], :y.shape[1]]> _dERK_dx
        double *         _dERK_dt   = <double *> PyMem_Malloc(y.size * sizeof(double))
        double[:, ::1]  __dERK_dt   = <double[:y.shape[0], :y.shape[1]]> _dERK_dt
        double *         _drho_dx   = <double *> PyMem_Malloc(y.size * sizeof(double))
        double[:, ::1]  __drho_dx   = <double[:y.shape[0], :y.shape[1]]> _drho_dx
        double *         _drho_dy   = <double *> PyMem_Malloc(y.size * sizeof(double))
        double[:, ::1]  __drho_dy   = <double[:y.shape[0], :y.shape[1]]> _drho_dy
        double *         _d2vx_dxdx = <double *> PyMem_Malloc(y.size * sizeof(double))
        double[:, ::1]  __d2vx_dxdx = <double[:y.shape[0], :y.shape[1]]> _d2vx_dxdx
        double *         _d2vx_dydy = <double *> PyMem_Malloc(y.size * sizeof(double))
        double[:, ::1]  __d2vx_dydy = <double[:y.shape[0], :y.shape[1]]> _d2vx_dydy
        double *         _dvx_dx    = <double *> PyMem_Malloc(y.size * sizeof(double))
        double[:, ::1]  __dvx_dx    = <double[:y.shape[0], :y.shape[1]]> _dvx_dx
        double *         _dvx_dy    = <double *> PyMem_Malloc(y.size * sizeof(double))
        double[:, ::1]  __dvx_dy    = <double[:y.shape[0], :y.shape[1]]> _dvx_dy
        double *         _dERK_dtdx = <double *> PyMem_Malloc(y.size * sizeof(double)) # 125
        double[:, ::1]  __dERK_dtdx = <double[:y.shape[0], :y.shape[1]]> _dERK_dtdx # 125
        np.ndarray[np.float64_t, ndim = 2] dERK_dx   = np.asarray(__dERK_dx)
        np.ndarray[np.float64_t, ndim = 2] dERK_dt   = np.asarray(__dERK_dt)
        np.ndarray[np.float64_t, ndim = 2] drho_dx   = np.asarray(__drho_dx)
        np.ndarray[np.float64_t, ndim = 2] drho_dy   = np.asarray(__drho_dy)
        np.ndarray[np.float64_t, ndim = 2] d2vx_dxdx = np.asarray(__d2vx_dxdx)
        np.ndarray[np.float64_t, ndim = 2] d2vx_dydy = np.asarray(__d2vx_dydy)
        np.ndarray[np.float64_t, ndim = 2] dvx_dx    = np.asarray(__dvx_dx)
        np.ndarray[np.float64_t, ndim = 2] dvx_dy    = np.asarray(__dvx_dy)
        np.ndarray[np.float64_t, ndim = 2] dERK_dtdx = np.asarray(__dERK_dtdx)
        ### tmp values
        int                                k
        ### output preparation, reshape to return
        list                               out_list
        #
        double *         _out       = <double *> PyMem_Malloc(y.size * sizeof(double))
        double[:, ::1]  __out       = <double[:y.shape[0], :y.shape[1]]> _out
        np.ndarray[np.float64_t, ndim = 2] out = np.asarray(__out)
        #
    try:
        ### init values
        ERK[:, :]       = options[2]
        rho[:, :]       = options[3]
        dERK_dt[:, :]   = options[5]
        ### see shape
        vx[:, :]        = y
        #
        #
        ### calculate diff
        dERK_dx[:, :]   = di1(ERK,     0, dx)
        drho_dx[:, :]   = di1(rho,     0, dx)
        drho_dy[:, :]   = di1(rho,     1, dx)
        d2vx_dxdx[:, :] = di2( vx,     0, dx)
        d2vx_dydy[:, :] = di2( vx,     1, dx)
        dvx_dx[:, :]    = di1( vx,     0, dx)
        dvx_dy[:, :]    = di1( vx,     1, dx)
        dERK_dtdx[:, :] = di1(dERK_dt, 0, dx) ### 125 190824
        #
        ### model
        out_list = [
            -1.0 * mu0 * np.exp(-1.0 * beta * ERK) * vx * rho,                # 0
            -2.0 * alpha * k0 * R0 * dERK_dx,                                 # 1
            -1.0 * k0 * drho_dx * rho**(-2.0),                                # 2
             eta * (d2vx_dxdx + d2vx_dydy) * rho**(-1.0),                     # 3
            -1.0 * eta * (drho_dx * dvx_dx + drho_dy * dvx_dy) * rho**(-2.0), # 4
            -1.0 * mu0 * np.exp(-1.0 * beta * dERK_dx) * vx * rho,            # 5
            -1.0 * mu0 * np.exp(-1.0 * beta * dERK_dt) * vx * rho,            # 6
            -2.0 * alpha * k0 * R0 * dERK_dtdx                                # 7
        ]
        '''
        ====================================
        below, make the output as sum of some in the list above.
        during test, terms should be tested one by one.
        make it able to be zeros.
        ====================================
        '''
        out[:, :]  = 0.0
        if len(mod_ind) > 0:
            out[:, :] += sum([out_list[k] for k in mod_ind])
        #
        return(np.array(__out))
        #
    finally:
        PyMem_Free(_vx)
        PyMem_Free(_ERK)
        PyMem_Free(_rho)
        PyMem_Free(_dERK_dx)
        PyMem_Free(_dERK_dt)
        PyMem_Free(_drho_dx)
        PyMem_Free(_drho_dy)
        PyMem_Free(_d2vx_dxdx)
        PyMem_Free(_d2vx_dydy)
        PyMem_Free(_dvx_dx)
        PyMem_Free(_dvx_dy)
        PyMem_Free(_dERK_dtdx)
        PyMem_Free(_out)
    #
    #

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef np.ndarray[np.float64_t, ndim = 2] adv_rho_fx(
    np.ndarray[np.float64_t, ndim = 3] y
):
    '''
    advection term for rho
    input y is rho, and returns v x rho
    v is input in option[0]
    '''
    return(y[2, :, :] * y[0, :, :])
    #
    #

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef np.ndarray[np.float64_t, ndim = 2] adv_rho_gy(
    np.ndarray[np.float64_t, ndim = 3] y
):
    '''
    advection term for rho
    input y is rho, and returns v x rho
    v is input in option[0]
    '''
    return(y[2, :, :] * y[1, :, :])
    #
    #

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef np.ndarray[np.float64_t, ndim = 2] adv_rho_fx_diff(
    np.ndarray[np.float64_t, ndim = 3] y
):
    '''
    advection term difference of rho
    '''
    return(y[0, :, :])
    #
    #


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef np.ndarray[np.float64_t, ndim = 2] adv_rho_gy_diff(
    np.ndarray[np.float64_t, ndim = 3] y
):
    '''
    advection term difference of rho
    '''
    return(y[1, :, :])
    #
    #


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef np.ndarray[np.float64_t, ndim = 2] TEST_ZERO_nopt(
    np.ndarray[np.float64_t, ndim = 2] y
):
    '''
    test function, no options as input
    '''
    return(np.zeros_like(y))
    #
    #

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef np.ndarray[np.float64_t, ndim = 2] TEST_ZERO_wopt(
    np.ndarray[np.float64_t, ndim = 2] y,
    list                               options
):
    '''
    test function, with options as input
    '''
    return(np.zeros_like(y))
    #
    #


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef np.ndarray[np.float64_t, ndim = 2] angle22(double theta):
    return(
        np.array([
            [cos(theta), -sin(theta)],
            [sin(theta),  cos(theta)]
        ])
    )


################
### class of simulation
################
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef class Diff_vrho_2D:
    '''
    class to run simulation of Euler discribed fields in WENO and Runge-Kutta
    #
    ### methods #######
    ### deftype    returns    dim    name
    ### cp         int        -      get_erk_store
    ### cp         ndarray    4      get_integrated
    ### cp         ndarray    3      get_ERK_blocked
    ### cp         list       -      get_fields
    ### cp         ndarray    2      get_times
    ### cp         ndarray    3      get_coord
    ### c          ndarray    2      line_gauss
    ### c          ndarray    2      line_gauss_dt
    ### c          ndarray    1      devo_vrho
    ### cp         -          -      run_sim
    ### cp         ndarray    3      get_ERK_3models
    ###################
    '''
    cdef:
        double     step_size
        double     dx
        int        imgx
        int        imgy
        list       paras
        double     tend
        #
        double *              _timerange
        double[:, ::1]       __timerange
        double *              _coord
        double[:, :, ::1]    __coord
        double *              _integrated
        double[:, :, :, ::1] __integrated
        np.ndarray timerange
        np.ndarray coord
        np.ndarray integrated
        #
        double     erk_store
        double     courant ### Courant number
        list       mod_ind
        #
        int        cut_out_length
        #
        double     amp_dt
        double     speed_grow
        #
        double     theta
        double *        _sigma_theta
        double[:, ::1] __sigma_theta
        #
        double *        _startpoints
        double[:]        startpoints
        double *        _fieldlims
        double[:]        fieldlims
        #
        #
    def __cinit__(self,
        double                             dt,
        double                             dx,
        int                                imgx,
        int                                imgy,
        list                               paras,
        double                             tend,
        double                             courant      = 0.2,
        bint                               noisy        = False,
        np.ndarray[np.float64_t, ndim = 3] add_init     = None,
        double                             t_init      = 0.0,
        list                               model_indice = [],
        double                             amp_dt       = 1.0,
        double                             theta_on_pi  = 0.25,
        list                               field_mod    = [64.0, 64.0],
        list                               startposi    = [0.0, 0.0],
        double                             speed_grow   = 0.5
    ):
        ################
        ### local values
        ################
        ### comment to notify that this is in cython
        print("initializing a class Diff_vrho_2D, coded in cython...")
        cdef:
            double *                _tmpt = <double *> PyMem_Malloc(int(tend / dt) * sizeof(double))
            double[::1]            __tmpt = <double[:int(tend / dt)]> _tmpt
            double *          _tmp_coordx = <double *> PyMem_Malloc(imgx * imgy * sizeof(double))
            double[:, ::1]   __tmp_coordx = <double[:imgx, :imgy]> _tmp_coordx
            double *          _tmp_coordy = <double *> PyMem_Malloc(imgx * imgy * sizeof(double))
            double[:, ::1]   __tmp_coordy = <double[:imgx, :imgy]> _tmp_coordy
            double *                 _yi0 = <double *> PyMem_Malloc(3 * imgx * imgy * sizeof(double))
            double[:, :, ::1]       __yi0 = <double[:3, :imgx, :imgy]> _yi0
            #
            np.ndarray[np.float64_t, ndim = 1] tmpt       = np.asarray(__tmpt)
            np.ndarray[np.float64_t, ndim = 2] tmp_coordx = np.asarray(__tmp_coordx)
            np.ndarray[np.float64_t, ndim = 2] tmp_coordy = np.asarray(__tmp_coordy)
            np.ndarray[np.float64_t, ndim = 3] yi0        = np.asarray(__yi0)
            #
            size_t sizet = sizeof(double)
            #
        #
        try:
            yi0[:, :, :] = np.zeros((3, imgx, imgy))
            #
            #
            #
            ################
            ### put input values
            ################
            self.step_size = dt
            self.dx        = dx
            self.imgx      = imgx ### pixels
            self.imgy      = imgy ### pixels
            self.paras     = paras
            self.tend      = tend
            self.mod_ind   = model_indice
            print(["accepted model indice are", self.mod_ind])
            #
            self.speed_grow = speed_grow
            #
            ### amplitudes
            self.amp_dt = amp_dt
            #
            ### theta
            self.theta   = theta_on_pi * pi
            #
            ### Courant Number max
            self.courant   = courant
            print("courant number is %.4f" % self.courant)
            #
            #
            ### timerange setting
            tmpt[:]              = np.arange(0.0, tend, self.step_size)
            ### initialize shapes of instance attributes
            self._timerange      = <double *> PyMem_Malloc(tmpt.size * sizeof(double))
            self.__timerange     = <double[:np.asarray(tmpt).shape[0], :1]> self._timerange
            self.timerange       = np.asarray(self.__timerange)
            ### init values
            self.timerange[:, 0] = np.array(__tmpt) + t_init
            #
            #
            ### hold xy corrdinates
            tmp_coordy[:, :], tmp_coordx[:, :] = np.meshgrid(
                np.arange(0, self.imgy, dtype = np.float64),
                np.arange(0, self.imgx, dtype = np.float64)
            )
            ### initialize shapes of instance attributes
            self._coord             = <double *> PyMem_Malloc(self.imgx * self.imgy * 2 * sizeof(double))
            self.__coord            = <double[:2, :self.imgx, :self.imgy]> self._coord
            self.coord              = np.asarray(self.__coord)
            ### init values
            self.coord[:, :, :]     = self.dx * np.concatenate(
                [
                    np.array(__tmp_coordx).reshape((1, self.imgx, self.imgy)),
                    np.array(__tmp_coordy).reshape((1, self.imgx, self.imgy))
                ],
                axis = 0
            )
            #
            ### initial value
            if noisy:
                yi0[2, :, :] = np.random.normal(paras[9], 0.02, (yi0.shape[1]*yi0.shape[2])).reshape((yi0.shape[1], yi0.shape[2]))
            else:
                yi0[2, :, :] = paras[9]
                #
            #
            if not add_init is None:
                yi0[:, :, :] += add_init
                #
            #
            ### result store
            ### initialize shapes of instance attributes
            self._integrated            = <double *> PyMem_Malloc(self.timerange.shape[0] * yi0.size * sizeof(double))
            self.__integrated           = <double[:self.timerange.shape[0], :3, :imgx, :imgy]> self._integrated
            self.integrated             = np.asarray(self.__integrated)
            ### init values
            self.integrated[:, :, :, :] = 0.0
            self.integrated[0, :, :, :] = np.array(__yi0)
            #
            ### to record ERK
            self.erk_store = 0.0
            #
            self.cut_out_length = self.timerange.shape[0]
            #
            #
            ### sigma theta
            self._sigma_theta  = <double *> PyMem_Malloc(4 * sizeof(double))
            self.__sigma_theta = <double[:2, :2]> self._sigma_theta
            np.asarray(self.__sigma_theta)[:, :] = angle22(self.theta).dot(np.array([
                [1.0 / self.paras[8], 0.0                ],
                [0.0,                 1.0 / self.paras[11]]
            ])).dot(angle22(-self.theta))
            #
            self._startpoints = <double *> PyMem_Malloc(2 * sizeof(double))
            self.startpoints  = <double[:2]> self._startpoints
            self.startpoints[0] = startposi[0]
            self.startpoints[1] = startposi[1]
            self._fieldlims   = <double *> PyMem_Malloc(2 * sizeof(double))
            self.fieldlims    = <double[:2]> self._fieldlims
            self.fieldlims[0] = field_mod[0]
            self.fieldlims[1] = field_mod[1]
            #
            #
        finally:
            PyMem_Free(_tmpt)
            PyMem_Free(_tmp_coordx)
            PyMem_Free(_tmp_coordy)
            PyMem_Free(_yi0)
        #
        #
        ################
        ### cinit end
        ################
    def __dealloc__(self):
        '''
        to deallocate memory on C level
        free depends on libc.stdlib
        instead, PyMem_Malloc, PyMem_Free from cpython.mem
        #
        only memoryviews are to be deallocated.
        '''
        PyMem_Free(self._timerange)
        PyMem_Free(self._coord)
        PyMem_Free(self._integrated)
        #print("deallocated")
        PyMem_Free(self._sigma_theta)
        #
        #
    cpdef get_erk_store(self):
        '''
        make it accessible from python
        '''
        return(np.array(self.__erk_store))
        #
        #
    cpdef get_integrated(self):
        '''
        make it accessible from python
        '''
        return(np.array(self.__integrated)[0:self.cut_out_length, :])
        #
        #
    cpdef get_ERK_blocked(self):
        '''
        timerange shoud be the output time points from ode solvers, 1-d array
        '''
        cdef:
            int j
            #
            double *           _erk_block = <double *> PyMem_Malloc(np.asarray(self.timerange).shape[0] * self.imgx * self.imgy * sizeof(double))
            double[:, :, :]   __erk_block = <double[:self.timerange.shape[0], :self.imgx, :self.imgy]> _erk_block
            np.ndarray[np.float64_t, ndim = 3] erk_block = np.asarray(__erk_block)
            #
        ### init values
        erk_block[:, :, :] = np.zeros((self.timerange.shape[0], self.imgx, self.imgy))
        ### put values
        for j in range(0, erk_block.shape[0]):
            erk_block[j, :, :] = self.line_gauss(self.timerange[j, 0])
            #
        try:
            return(np.array(__erk_block)[0:self.cut_out_length, :, :])
        finally:
            PyMem_Free(_erk_block)
        #
    cpdef get_ERK_dt_block(self):
        '''
        timerange shoud be the output time points from ode solvers, 1-d array
        '''
        cdef:
            int j
            #
            double *           _erk_block = <double *> PyMem_Malloc(np.asarray(self.timerange).shape[0] * self.imgx * self.imgy * sizeof(double))
            double[:, :, :]   __erk_block = <double[:self.timerange.shape[0], :self.imgx, :self.imgy]> _erk_block
            np.ndarray[np.float64_t, ndim = 3] erk_block = np.asarray(__erk_block)
            #
        ### init values
        erk_block[:, :, :] = np.zeros((self.timerange.shape[0], self.imgx, self.imgy))
        ### put values
        for j in range(0, erk_block.shape[0]):
            erk_block[j, :, :] = self.line_gauss_dt(self.timerange[j, 0])
            #
        try:
            return(np.array(__erk_block)[0:self.cut_out_length, :, :])
        finally:
            PyMem_Free(_erk_block)
        #
        #
    cpdef get_fields(self):
        '''
        for Time_values
        '''
        cdef:
            double *               _reshaped = <double *> PyMem_Malloc(self.integrated.shape[0] * 5 * self.imgx * self.imgy * sizeof(double))
            double[:, :, :, ::1]  __reshaped = <double[:self.integrated.shape[0], :5, :self.imgx, :self.imgy]> _reshaped
            np.ndarray[np.float64_t, ndim = 4] reshaped = np.asarray(__reshaped)
            #
        try:
            ### values
            reshaped[:, 0:3, :, :] = self.integrated.reshape((self.integrated.shape[0], 3, self.imgx, self.imgy))
            ### ERK
            reshaped[:, 3, :, :] = self.get_ERK_blocked()
            reshaped[:, 4, :, :] = self.get_ERK_dt_block()
            #
            return(np.array(__reshaped[0:self.cut_out_length, :, :, :]))
            #
        finally:
            PyMem_Free(_reshaped)
            #
        #
    cpdef get_times(self):
        return(np.array(self.__timerange)[0:self.cut_out_length, :])
        #
        #
    cpdef get_coord(self):
        return(np.array(self.__coord))
        #
        #
    cpdef get_in_erk_share(self):
        return([
            np.array(self.__sigma_theta),
            self.paras[7],
            self.speed_grow,
            self.startpoints,
            self.fieldlims,
            self.theta
        ])
        #
        #
        ################
        ### methods above are getting values in python, below are calculation
        ################
    cdef np.ndarray[np.float64_t, ndim = 2] line_gauss(self, double t):
        cdef:
            double            sigma_inner = self.paras[8]
            double            speed1d     = self.paras[7]
            double            speed_grow  = self.speed_grow
            double            e_e         = 64.0
            double            starts
            #
            double *         _erkfield    = <double *> PyMem_Malloc(self.imgx * self.imgy * sizeof(double))
            double[:, ::1]  __erkfield    = <double[:self.imgx, :self.imgy]> _erkfield
            #
            np.ndarray[np.float64_t, ndim = 2] erkfield    = np.asarray(__erkfield)
            #
            int j
            int k
            int len0 = self.imgx
            int len1 = self.imgy
            #
        try:
            erkfield[:, :] = erk_shared(
                t,
                self.__coord,
                self._sigma_theta,
                speed1d,
                speed_grow,
                self.startpoints,
                self.fieldlims,
                self.theta
            )
            return(np.array(__erkfield))
        finally:
            PyMem_Free(_erkfield)
        #
    cdef np.ndarray[np.float64_t, ndim = 2] line_gauss_dt(self, double t):
        cdef:
            double            sigma_inner = self.paras[8]
            double            speed1d     = self.paras[7]
            double            speed_grow  = self.speed_grow
            double            e_e         = 64.0
            double            starts
            #
            double *         _erkfield    = <double *> PyMem_Malloc(self.imgx * self.imgy * sizeof(double))
            double[:, ::1]  __erkfield    = <double[:self.imgx, :self.imgy]> _erkfield
            double *         _beginers    = <double *> PyMem_Malloc(int(float(self.imgx) / e_e) * sizeof(double))
            double[::1]     __beginers    = <double[:int(float(self.imgx) / e_e)]> _beginers
            #
            np.ndarray[np.float64_t, ndim = 2] erkfield    = np.asarray(__erkfield)
            np.ndarray[np.float64_t, ndim = 1] beginers    = np.asarray(__beginers)
            #
            int j
            int k
            int len0 = self.imgx
            int len1 = self.imgy
            #
            double *        _sigma = <double *> PyMem_Malloc(4 * sizeof(double))
            double[:, ::1] __sigma = <double[:2, :2]> _sigma
            double *        _tmp2d = <double *> PyMem_Malloc(len0 * len1 * sizeof(double))
            double[:, ::1] __tmp2d = <double[:len0, :len1]> _tmp2d
            np.ndarray[np.float64_t, ndim = 2] tmp2d = np.asarray(__tmp2d)
        #
        __sigma[:, :] = 0.0
        __sigma[0, 0] = 1.0 / sigma_inner
        __sigma[1, 1] = 2.0 / sigma_inner
        ### use erk_dt defined above
        beginers[:] = np.remainder(np.arange(0.0, float(self.imgx), e_e), float(self.imgx))
        erkfield[:, :] = sum([erk_dt_2d(t, self.coord, np.asarray(__sigma), speed1d, speed_grow, starts) for starts in np.concatenate((beginers, beginers - float(self.imgx)))])
        erkfield[:, :] = erkfield * self.amp_dt
        #
        try:
            return(np.array(__erkfield))
        finally:
            PyMem_Free(_erkfield)
            PyMem_Free(_beginers)
            PyMem_Free(_sigma)
            PyMem_Free(_tmp2d)
        #
        #
        ################
        ### model function put in WENO in Runge-Kutta
        ################
    cdef np.ndarray[np.float64_t, ndim = 3] devo_vrho(
        self,
        double t,
        np.ndarray[np.float64_t, ndim = 3] y
    ):
        '''
        t is the exact time
        y is v and rho, bound together
        put WENO here
        put this in Runge_kutta
        1. separate v and rho
        2. apply WENO with advection term and source term for each of v and rho
        3. return bound v and rho
        #
        depends on weno_2d
        cdef np.ndarray[np.float64_t, ndim = 2] weno_2d(
            np.ndarray[np.float64_t, ndim = 2] y,
            double                             dx,
            object                             advection,
            object                             adv_diff,
            object                             source      = None,
            list                               options     = [[], [], []]
        )
        '''
        cdef:
            double *                           _out      = <double *> PyMem_Malloc(3 * self.imgx * self.imgy * sizeof(double))
            double[:, :, ::1]                 __out      = <double[:3, :self.imgx, :self.imgy]> _out
            np.ndarray[np.float64_t, ndim = 3]  out      = np.asarray(__out)
            double *                           _y_left   = <double *> PyMem_Malloc(self.imgx * self.imgy * sizeof(double))
            double[:, ::1]                    __y_left   = <double[:self.imgx, :self.imgy]> _y_left
            np.ndarray[np.float64_t, ndim = 2]  y_left   = np.asarray(__y_left)
            double *                           _y_right  = <double *> PyMem_Malloc(self.imgx * self.imgy * sizeof(double))
            double[:, ::1]                    __y_right  = <double[:self.imgx, :self.imgy]> _y_right
            np.ndarray[np.float64_t, ndim = 2]  y_right  = np.asarray(__y_right)
            double *                           _tmp_2d   = <double *> PyMem_Malloc(self.imgx * self.imgy * sizeof(double))
            double[:, ::1]                    __tmp_2d   = <double[:self.imgx, :self.imgy]> _tmp_2d
            np.ndarray[np.float64_t, ndim = 2]  tmp_2d   = np.asarray(__tmp_2d)
            #
            double                              a_LF     ### alpha in LF
            #
        try:
            ### init values
            out[:, :, :] = 0.0
            #
            ################
            ### momentum x
            ################
            ### preparations direction x
            tmp_2d[:, :]  = adv_mx_fx_diff(y)
            a_LF          = np.amax(np.absolute(tmp_2d))
            ### f+ and f- preparation
            tmp_2d[:, :]  = adv_mx_fx(y)
            ### left
            y_left[:, :]  =         0.5 * (tmp_2d + a_LF * y[0, :, :] * y[2, :, :])
            ### right, next stencil
            y_right[:, :] = np.roll(0.5 * (tmp_2d - a_LF * y[0, :, :] * y[2, :, :]), -1, axis = 0)
            #
            out[0, :, :]  = weno_2d(
                y_left,
                y_right,
                self.dx
            )
            ### direction y
            tmp_2d[:, :]  = adv_mx_gy_diff(y)
            a_LF          = np.amax(np.absolute(tmp_2d))
            ### f+ and f- preparation
            tmp_2d[:, :]  = adv_mx_gy(y)
            ### left
            y_left[:, :]  =         0.5 * (tmp_2d + a_LF * y[0, :, :] * y[2, :, :])
            ### right, next stencil, ATTENTION... transpose for the y direction
            y_right[:, :] = np.roll(0.5 * (tmp_2d - a_LF * y[0, :, :] * y[2, :, :]), -1, axis = 1)
            ### ATTENTION... transpose, and include source term
            out[0, :, :] += weno_2d(
                y_left.T,
                y_right.T,
                self.dx
            ).T - source_momentum(
                y[0, :, :],
                [self.paras, self.dx, self.line_gauss(t), y[2, :, :], self.mod_ind, self.line_gauss_dt(t)]
            )
            #
            ################
            ### momentum y
            ################
            ### preparations direction x
            tmp_2d[:, :]  = adv_my_fx_diff(y)
            a_LF          = np.amax(np.absolute(tmp_2d))
            ### f+ and f- preparation
            tmp_2d[:, :]  = adv_my_fx(y)
            ### left
            y_left[:, :]  =         0.5 * (tmp_2d + a_LF * y[1, :, :] * y[2, :, :])
            ### right, next stencil
            y_right[:, :] = np.roll(0.5 * (tmp_2d - a_LF * y[1, :, :] * y[2, :, :]), -1, axis = 0)
            #
            out[1, :, :]  = weno_2d(
                y_left,
                y_right,
                self.dx
            )
            ### direction y
            tmp_2d[:, :]  = adv_my_gy_diff(y)
            a_LF          = np.amax(np.absolute(tmp_2d))
            ### f+ and f- preparation
            tmp_2d[:, :]  = adv_my_gy(y)
            ### left
            y_left[:, :]  =         0.5 * (tmp_2d + a_LF * y[1, :, :] * y[2, :, :])
            ### right, next stencil, ATTENTION... transpose for the y direction
            y_right[:, :] = np.roll(0.5 * (tmp_2d - a_LF * y[1, :, :] * y[2, :, :]), -1, axis = 1)
            ### ATTENTION... transpose, and include source term
            out[1, :, :] += weno_2d(
                y_left.T,
                y_right.T,
                self.dx
            ).T - source_momentum(
                y[1, :, :].T,
                [self.paras, self.dx, self.line_gauss(t).T, y[2, :, :].T, self.mod_ind, self.line_gauss_dt(t).T]
            ).T
            #
            ################
            ### rho
            ################
            ### preparations direction x
            tmp_2d[:, :]  = adv_rho_fx_diff(y)
            a_LF          = np.amax(np.absolute(tmp_2d))
            ### f+ and f- preparation
            tmp_2d[:, :]  = adv_rho_fx(y)
            ### left
            y_left[:, :]  =         0.5 * (tmp_2d + a_LF * y[2, :, :])
            ### right, next stencil
            y_right[:, :] = np.roll(0.5 * (tmp_2d - a_LF * y[2, :, :]), -1, axis = 0)
            #
            out[2, :, :]  = weno_2d(
                y_left,
                y_right,
                self.dx
            )
            ### direction y
            tmp_2d[:, :]  = adv_rho_gy_diff(y)
            a_LF          = np.amax(np.absolute(tmp_2d))
            ### f+ and f- preparation
            tmp_2d[:, :]  = adv_rho_gy(y)
            ### left
            y_left[:, :]  =         0.5 * (tmp_2d + a_LF * y[2, :, :])
            ### right, next stencil, ATTENTION... transpose for the y direction
            y_right[:, :] = np.roll(0.5 * (tmp_2d - a_LF * y[2, :, :]), -1, axis = 1)
            ### ATTENTION... transpose, and include source term
            out[2, :, :] += weno_2d(
                y_left.T,
                y_right.T,
                self.dx
            ).T
            if np.any(y[2, :, :] == 0.0):
                print("there is 0.0 in rho")
            #
            ### calculate velocity based on the momentum above
            out[0, :, :] = out[0, :, :] / y[2, :, :]
            out[1, :, :] = out[1, :, :] / y[2, :, :]
            #
            return(np.array(__out))
            #
        finally:
            PyMem_Free(_out)
            PyMem_Free(_y_left)
            PyMem_Free(_y_right)
            PyMem_Free(_tmp_2d)
        #
        #
        ################
        ### below, run simulation in Runge_Kutta with WENO
        ################
    cpdef run_sim(self):
        '''
        depends on weno_1d_LF
        '''
        cdef:
            int                                c = 1
            int                                j
            double                             timej
            double                             t_cur ### time
            double                             t_old ### time
            double                             dt    ### may smaller than self.step_size
            #
            double *              _tmp_counter = <double *> PyMem_Malloc(3 * self.imgx * self.imgy * sizeof(double))
            double [:, :, ::1]   __tmp_counter = <double[:3, :self.imgx, :self.imgy]> _tmp_counter
            double *              _u_cur       = <double *> PyMem_Malloc(3 * self.imgx * self.imgy * sizeof(double))
            double [:, :, ::1]   __u_cur       = <double[:3, :self.imgx, :self.imgy]> _u_cur
            double *              _u_pre       = <double *> PyMem_Malloc(3 * self.imgx * self.imgy * sizeof(double))
            double [:, :, ::1]   __u_pre       = <double[:3, :self.imgx, :self.imgy]> _u_pre
            np.ndarray[np.float64_t, ndim = 3] tmp_counter = np.asarray(__tmp_counter)
            np.ndarray[np.float64_t, ndim = 3] u_cur       = np.asarray(__u_cur)
            np.ndarray[np.float64_t, ndim = 3] u_pre       = np.asarray(__u_pre)
            #
            int                                cou_lp      = 0
            double                             kill_th     = 5e-4
            bint                               break_sim   = False
        #
        try:
            print("simulation starts, WENO with Lax-Friedrichs, in Runge-Kutta")
            #
            ### initialize time
            t_old = self.timerange[0, 0]
            t_cur = self.timerange[0, 0]
            #
            ### initialize integrating value
            u_cur[:, :, :] = self.integrated[0, :, :, :]
            #
            ### loop to calculate WENO in Runge-Kutta
            for j in range(1, self.timerange.shape[0]):
                timej = self.timerange[j, 0]
                #
                ### count loops in this step, controled by courant number
                cou_lp = 0
                #
                ### loop to reach timej, with smaller steps for keep courant number small
                while t_cur < timej:
                    '''
                    this loop end if one block tries to start with False
                    it does not mean that it breaks immediately after it make false
                    '''
                    ### decide dt
                    dt = self.courant * self.dx / np.amax(np.absolute(u_cur))
                    #
                    ### check if dt is too much
                    t_cur += dt
                    if timej <= t_cur:
                        '''
                        if True, this while loop ends at this end
                        '''
                        t_cur = timej
                        dt    = t_cur - t_old
                        #
                    elif dt <= kill_th:
                        ### check if dt is too small, i.e. some value goes like inf
                        break_sim = True
                        break
                        ### this breaks only the while loop
                        #
                    #
                    ### now dt was decided and exact timepoint needed is to be yealded
                    ### keep previous integrated value
                    u_pre[:, :, :] = np.array(__u_cur)
                    #
                    #
                    ### Runge Kutta, 1st stage
                    u_cur[:, :, :] = u_pre - dt * self.devo_vrho(t_cur, u_cur)
                    ### Runge Kutta, 2nd stage
                    u_cur[:, :, :] = 0.75 * u_pre        +  0.25       * (u_cur - dt * self.devo_vrho(t_cur, u_cur))
                    ### Runge Kutta, 3rd stage
                    u_cur[:, :, :] = (1.0 / 3.0) * u_pre + (2.0 / 3.0) * (u_cur - dt * self.devo_vrho(t_cur, u_cur))
                    #
                    if np.any(np.isnan(u_cur)) or np.any(np.isinf(u_cur)):
                        break_sim = True
                        break
                    #
                    ### renewal t_old
                    t_old = t_cur
                    #
                    ### courant number loop count
                    cou_lp += 1
                    #
                    #
                ### counter or broken
                if c >= 20 or break_sim:
                    tmp_counter[:, :, :] = self.integrated[(j - 1), :, :, :]
                    print("    sim %.3f, loop %d, sum (v %.4f, rho %.4f), inner %d" % (timej, j, np.sum(tmp_counter[0, :, :]), np.sum(tmp_counter[2, :, :]), cou_lp))
                    c = 0
                c += 1
                #
                ### escape from too small dt i.e. something goes to inf
                if break_sim:
                    print("    FAILED dt was %f\n    simulation failed\n    too small dt emerged i.e. less than 10 to the power of %.1f\nanyway," % (dt, np.log10(kill_th)))
                    ### cut integrated and timerange
                    self.cut_out_length = j
                    break
                #
                ### store the result
                self.integrated[j, :, :, :]   = np.array(__u_cur)
                #
                #
            print("simulation done")
            #
        finally:
            PyMem_Free(_tmp_counter)
            PyMem_Free(_u_cur)
            PyMem_Free(_u_pre)
            #
            #
        #
        ################
        ### below are for Lagrange-described simulations
        ################
    cpdef get_ERK_3models(self, double t):
        cdef:
            double *                           _out = <double *> PyMem_Malloc(3 * self.imgx * self.imgy * sizeof(double))
            double[:, :, ::1]                 __out = <double[:3, :self.imgx, :self.imgy]> _out
            np.ndarray[np.float64_t, ndim = 3]  out = np.asarray(__out)
        try:
            out[0, :, :] = self.line_gauss(t)
            out[1, :, :] = di1(out[0, :, :], 0, self.dx)
            out[2, :, :] = self.line_gauss_dt(t)
            return(np.array(__out))
        finally:
            PyMem_Free(_out)
        #
        #



################
###
################


################
###
################


################
###
################


###
