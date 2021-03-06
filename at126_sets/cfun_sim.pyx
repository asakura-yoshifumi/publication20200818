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


################
### ERK
################
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef np.ndarray[np.float64_t, ndim = 2] erk_multi(
    double                             t,
    np.ndarray[np.float64_t, ndim = 2] X,
    double                             sigma_inner,
    double                             speed1d,
    double                             speed_grow,
    double                             start_point
):
    cdef:
        double                               erk_max_t
        #
        double *                            _d           = <double *> PyMem_Malloc(X.size * sizeof(double))
        double[:, ::1]                     __d           = <double[:X.shape[0], :X.shape[1]]> _d
        np.ndarray[np.float64_t, ndim = 2]   d           = np.asarray(__d)
        #
        double *                            _erkfield_in = <double *> PyMem_Malloc(X.size * sizeof(double))
        double[:, ::1]                     __erkfield_in = <double[:X.shape[0], :X.shape[1]]> _erkfield_in
        np.ndarray[np.float64_t, ndim = 2]   erkfield_in = np.asarray(__erkfield_in)
        #
    #
    erk_max_t   = t * speed_grow
    if erk_max_t <= 0.0:
        erk_max_t = 0.0
    elif 1.0 <= erk_max_t:
        erk_max_t = 1.0
    #
    ### make a field of distance to wave top
    d[:, :] = X - (speed1d * t + start_point)
    #
    erkfield_in[:, :] = np.exp(-d**2.0 / (2.0 * sigma_inner**2.0)) * erk_max_t ### max is 1.0, at 022
    try:
        return(np.array(__erkfield_in))
    finally:
        PyMem_Free(_d)
        PyMem_Free(_erkfield_in)
    #
    #

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef np.ndarray[np.float64_t, ndim = 2] erk_dt(
    double                             t,
    np.ndarray[np.float64_t, ndim = 2] X,
    double                             sigma_inner,
    double                             speed1d,
    double                             speed_grow,
    double                             start_point
):
    cdef:
        double                               Amp_t
        double                               Amp_tdt     = 0.0
        #
        double *                            _d           = <double *> PyMem_Malloc(X.size * sizeof(double))
        double[:, ::1]                     __d           = <double[:X.shape[0], :X.shape[1]]> _d
        np.ndarray[np.float64_t, ndim = 2]   d           = np.asarray(__d)
        #
        double *                            _erkfield_in = <double *> PyMem_Malloc(X.size * sizeof(double))
        double[:, ::1]                     __erkfield_in = <double[:X.shape[0], :X.shape[1]]> _erkfield_in
        np.ndarray[np.float64_t, ndim = 2]   erkfield_in = np.asarray(__erkfield_in)
        #
    #
    ### make a field of distance to wave top
    d[:, :] = X - (speed1d * t + start_point)
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
    #
    erkfield_in[:, :] = np.exp(-d**2.0 / (2.0 * sigma_inner**2.0))\
                      * (Amp_tdt + Amp_t * speed1d * d / (sigma_inner**2.0))
    try:
        return(np.array(__erkfield_in))
    finally:
        PyMem_Free(_d)
        PyMem_Free(_erkfield_in)
    #
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
### WENO
################
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef np.ndarray[np.float64_t, ndim = 2] num_flux_k3_1d(
    np.ndarray[np.float64_t, ndim = 2] vc,
    bint                               left
):
    '''
    shared part to calculate numerical flux in WENO
    from stencil to weight
    '''
    cdef:
        double dr0     = 3.0 / 10.0
        double dr1     = 6.0 / 10.0
        double dr2     = 1.0 / 10.0
        double epsilon = 1e-6
        ### flux, left and right
        double *         _vl2        = <double *> PyMem_Malloc(vc.size * sizeof(double))
        double[:, ::1]  __vl2        = <double[:vc.shape[0], :vc.shape[1]]> _vl2
        double *         _vl1        = <double *> PyMem_Malloc(vc.size * sizeof(double))
        double[:, ::1]  __vl1        = <double[:vc.shape[0], :vc.shape[1]]> _vl1
        double *         _vr1        = <double *> PyMem_Malloc(vc.size * sizeof(double))
        double[:, ::1]  __vr1        = <double[:vc.shape[0], :vc.shape[1]]> _vr1
        double *         _vr2        = <double *> PyMem_Malloc(vc.size * sizeof(double))
        double[:, ::1]  __vr2        = <double[:vc.shape[0], :vc.shape[1]]> _vr2
        np.ndarray[np.float64_t, ndim = 2] vl2 = np.asarray(__vl2)
        np.ndarray[np.float64_t, ndim = 2] vl1 = np.asarray(__vl1)
        np.ndarray[np.float64_t, ndim = 2] vr1 = np.asarray(__vr1)
        np.ndarray[np.float64_t, ndim = 2] vr2 = np.asarray(__vr2)
        ### weight and mediator
        double *         _alpha0     = <double *> PyMem_Malloc(vc.size * sizeof(double))
        double[:, ::1]  __alpha0     = <double[:vc.shape[0], :vc.shape[1]]> _alpha0
        double *         _alpha1     = <double *> PyMem_Malloc(vc.size * sizeof(double))
        double[:, ::1]  __alpha1     = <double[:vc.shape[0], :vc.shape[1]]> _alpha1
        double *         _alpha2     = <double *> PyMem_Malloc(vc.size * sizeof(double))
        double[:, ::1]  __alpha2     = <double[:vc.shape[0], :vc.shape[1]]> _alpha2
        double *         _sum_alpha  = <double *> PyMem_Malloc(vc.size * sizeof(double))
        double[:, ::1]  __sum_alpha  = <double[:vc.shape[0], :vc.shape[1]]> _sum_alpha
        np.ndarray[np.float64_t, ndim = 2] alpha0 = np.asarray(__alpha0)
        np.ndarray[np.float64_t, ndim = 2] alpha1 = np.asarray(__alpha1)
        np.ndarray[np.float64_t, ndim = 2] alpha2 = np.asarray(__alpha2)
        np.ndarray[np.float64_t, ndim = 2] sum_alpha = np.asarray(__sum_alpha)
        #
        double *         _beta0      = <double *> PyMem_Malloc(vc.size * sizeof(double))
        double[:, ::1]  __beta0      = <double[:vc.shape[0], :vc.shape[1]]> _beta0
        double *         _beta1      = <double *> PyMem_Malloc(vc.size * sizeof(double))
        double[:, ::1]  __beta1      = <double[:vc.shape[0], :vc.shape[1]]> _beta1
        double *         _beta2      = <double *> PyMem_Malloc(vc.size * sizeof(double))
        double[:, ::1]  __beta2      = <double[:vc.shape[0], :vc.shape[1]]> _beta2
        np.ndarray[np.float64_t, ndim = 2] beta0 = np.asarray(__beta0)
        np.ndarray[np.float64_t, ndim = 2] beta1 = np.asarray(__beta1)
        np.ndarray[np.float64_t, ndim = 2] beta2 = np.asarray(__beta2)
        #
        double *         _weight0    = <double *> PyMem_Malloc(vc.size * sizeof(double))
        double[:, ::1]  __weight0    = <double[:vc.shape[0], :vc.shape[1]]> _weight0
        double *         _weight1    = <double *> PyMem_Malloc(vc.size * sizeof(double))
        double[:, ::1]  __weight1    = <double[:vc.shape[0], :vc.shape[1]]> _weight1
        double *         _weight2    = <double *> PyMem_Malloc(vc.size * sizeof(double))
        double[:, ::1]  __weight2    = <double[:vc.shape[0], :vc.shape[1]]> _weight2
        np.ndarray[np.float64_t, ndim = 2] weight0 = np.asarray(__weight0)
        np.ndarray[np.float64_t, ndim = 2] weight1 = np.asarray(__weight1)
        np.ndarray[np.float64_t, ndim = 2] weight2 = np.asarray(__weight2)
        ### polynomials
        double *         _poly0      = <double *> PyMem_Malloc(vc.size * sizeof(double))
        double[:, ::1]  __poly0      = <double[:vc.shape[0], :vc.shape[1]]> _poly0
        double *         _poly1      = <double *> PyMem_Malloc(vc.size * sizeof(double))
        double[:, ::1]  __poly1      = <double[:vc.shape[0], :vc.shape[1]]> _poly1
        double *         _poly2      = <double *> PyMem_Malloc(vc.size * sizeof(double))
        double[:, ::1]  __poly2      = <double[:vc.shape[0], :vc.shape[1]]> _poly2
        np.ndarray[np.float64_t, ndim = 2] poly0 = np.asarray(__poly0)
        np.ndarray[np.float64_t, ndim = 2] poly1 = np.asarray(__poly1)
        np.ndarray[np.float64_t, ndim = 2] poly2 = np.asarray(__poly2)
        #
        double *         _mat        = <double *> PyMem_Malloc(12 * sizeof(double))
        double[:, ::1]  __mat        = <double[:4, :3]> _mat
        np.ndarray[np.float64_t, ndim = 2] mat = np.asarray(__mat)
        int                                l
        int                                m
        int                                n
        list                               vec
        ### numerical flux
        double *         _h_out      = <double *> PyMem_Malloc(vc.size * sizeof(double))
        double[:, ::1]  __h_out      = <double[:vc.shape[0], :vc.shape[1]]> _h_out
        np.ndarray[np.float64_t, ndim = 2] h_out = np.asarray(__h_out)
        #
    #
    ### stencils
    vl2[:, :] = np.roll(vc,  2, axis = 0)
    vl1[:, :] = np.roll(vc,  1, axis = 0)
    vr1[:, :] = np.roll(vc, -1, axis = 0)
    vr2[:, :] = np.roll(vc, -2, axis = 0)
    #
    ### polynomials preparetion
    mat[:, :] = np.array(
        [   [ 2.0, -7.0, 11.0],  ### left only
            [-1.0,  5.0,  2.0],  ### both left and right
            [ 2.0,  5.0, -1.0],  ### both left and right
            [11.0, -7.0,  2.0]], ### right only
        dtype = np.float64
    ) / 6.0
    vec = [[vl2, vl1, vc ],
           [vl1, vc , vr1],
           [vc , vr1, vr2]]
    #
    ### polynomials
    n = vc.shape[0]
    if left:
        poly0[:, :], poly1[:, :], poly2[:, :] = [np.sum(np.concatenate(tuple([mat[l,       m] * vec[l][m] for m in range(0, 3)]), axis = 1), axis = 1).reshape((n, 1)) for l in range(0, 3)]
    else:
        poly0[:, :], poly1[:, :], poly2[:, :] = [np.sum(np.concatenate(tuple([mat[(l + 1), m] * vec[l][m] for m in range(0, 3)]), axis = 1), axis = 1).reshape((n, 1)) for l in range(0, 3)]
    #
    ### beta, smooth indicator
    beta0[:, :] = (13.0 / 12.0) * (vl2 - 2.0 * vl1 + vc)  ** 2.0 + 0.25 * (vl2      - 4.0 * vl1 + 3.0 * vc) ** 2.0
    beta1[:, :] = (13.0 / 12.0) * (vl1 - 2.0 * vc  + vr1) ** 2.0 + 0.25 * (vl1                  - vr1     ) ** 2.0
    beta2[:, :] = (13.0 / 12.0) * (vc  - 2.0 * vr1 + vr2) ** 2.0 + 0.25 * (3.0 * vc - 4.0 * vr1 + vr2     ) ** 2.0
    #
    ### np.power is slower
    #
    ### alpha, weights, here, power by np for easy read
    alpha0[:, :] = dr0 * np.power((epsilon + beta0), -2.0)
    alpha1[:, :] = dr1 * np.power((epsilon + beta1), -2.0)
    alpha2[:, :] = dr2 * np.power((epsilon + beta2), -2.0)
    #
    sum_alpha[:, :] = alpha0 + alpha1 + alpha2
    #
    ### weights
    weight0[:, :] = alpha0 / sum_alpha
    weight1[:, :] = alpha1 / sum_alpha
    weight2[:, :] = alpha2 / sum_alpha
    #
    ### numerical flux left
    h_out[:, :] = weight0 * poly0 + weight1 * poly1 + weight2 * poly2
    #
    try:
        return(np.array(__h_out))
    finally:
        PyMem_Free(_vl2)
        PyMem_Free(_vl1)
        PyMem_Free(_vr1)
        PyMem_Free(_vr2)
        PyMem_Free(_alpha0)
        PyMem_Free(_alpha1)
        PyMem_Free(_alpha2)
        PyMem_Free(_sum_alpha)
        PyMem_Free(_beta0)
        PyMem_Free(_beta1)
        PyMem_Free(_beta2)
        PyMem_Free(_weight0)
        PyMem_Free(_weight1)
        PyMem_Free(_weight2)
        PyMem_Free(_poly0)
        PyMem_Free(_poly1)
        PyMem_Free(_poly2)
        PyMem_Free(_mat)
        PyMem_Free(_h_out)
    #
    #


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef np.ndarray[np.float64_t, ndim = 1] weno_1d_LF(
    np.ndarray[np.float64_t, ndim = 1] y,
    double                             dx,
    object                             advection,
    object                             adv_diff,
    object                             source    = None,
    list                               options   = [[], [], []]
):
    '''
    options are to be passed to each function input, in the same order
    '''
    ### definition of types
    cdef:
        ### values depending on flux and field
        double            a
        #
        double *         _y2          = <double *> PyMem_Malloc(y.size * sizeof(double))
        double[:, ::1]  __y2          = <double[:y.shape[0], :1]> _y2
        np.ndarray[np.float64_t, ndim = 2] y2 = np.asarray(__y2)
        ### center vector
        double *         _vc          = <double *> PyMem_Malloc(y.size * sizeof(double))
        double[:, ::1]  __vc          = <double[:y.shape[0], :1]> _vc
        np.ndarray[np.float64_t, ndim = 2] vc = np.asarray(__vc)
        ### numerical flux and final output
        double *         _h_left      = <double *> PyMem_Malloc(y.size * sizeof(double))
        double[:, ::1]  __h_left      = <double[:y.shape[0], :1]> _h_left
        double *         _h_right     = <double *> PyMem_Malloc(y.size * sizeof(double))
        double[:, ::1]  __h_right     = <double[:y.shape[0], :1]> _h_right
        double *         _out         = <double *> PyMem_Malloc(y.size * sizeof(double))
        double[:, ::1]  __out         = <double[:y.shape[0], :1]> _out
        np.ndarray[np.float64_t, ndim = 2] h_left  = np.asarray(__h_left)
        np.ndarray[np.float64_t, ndim = 2] h_right = np.asarray(__h_right)
        np.ndarray[np.float64_t, ndim = 2] out     = np.asarray(__out) ### reshape to 1 dim in return
        ### temporal use for reshape, regarding functions input
        int               tmp_int
        #
        double *         _tmp_1_d     = <double *> PyMem_Malloc(y.size * sizeof(double))
        double[::1]     __tmp_1_d     = <double[:y.shape[0]]> _tmp_1_d
        double *         _source_term = <double *> PyMem_Malloc(y.size * sizeof(double))
        double[:, ::1]  __source_term = <double[:y.shape[0], :1]> _source_term
        np.ndarray[np.float64_t, ndim = 1] tmp_1_d     = np.asarray(__tmp_1_d)
        np.ndarray[np.float64_t, ndim = 2] source_term = np.asarray(__source_term)
        #
    #
    ### Lax Friedriches flux
    if len(options[1]) == 0:
        tmp_1_d[:] = adv_diff(y)
    else:
        tmp_1_d[:] = adv_diff(y, options[1])
    #
    tmp_int  = y.shape[0]
    y2[:, :] = y.reshape((tmp_int, 1))
    a        = np.amax(np.absolute(tmp_1_d))
    #
    #
    ################
    ### left, shared are, stencil - weight
    ################
    if len(options[0]) == 0:
        tmp_1_d[:] = advection(y)
    else:
        tmp_1_d[:] = advection(y, options[0])
    #
    tmp_int = y2.shape[0]
    vc[:, :] = 0.5 * (tmp_1_d.reshape((tmp_int, 1)) + a * y2)
    #
    ### numerical flux left
    h_left[:, :]  = num_flux_k3_1d(vc, left = True)
    #
    #
    ################
    ### right, shared are, stencil - weight
    ################
    # it can use the same tmp_1_d as left
    # it can use the same tmp_int as left
    vc[:, :] = np.roll(0.5 * (tmp_1_d.reshape((tmp_int, 1)) - a * y2), -1, axis = 0)
    #
    ### numerical flux right
    h_right[:, :] = num_flux_k3_1d(vc, left = False)
    #
    #
    ################
    ### output
    ################
    if source is None:
        source_term[:, :] = np.zeros_like(y2)
    else:
        tmp_int           = y2.shape[0]
        ### source term function must get options any time
        source_term[:, :] = source(y2.reshape(-1), options[2]).reshape((tmp_int, 1))
        #
    out[:, :] = (h_left - np.roll(h_left, 1, axis = 0) + h_right - np.roll(h_right, 1, axis = 0)) / dx - source_term
    #
    try:
        return(np.array(__out).reshape(-1))
    finally:
        PyMem_Free(_y2)
        PyMem_Free(_vc)
        PyMem_Free(_h_left)
        PyMem_Free(_h_right)
        PyMem_Free(_out)
        PyMem_Free(_tmp_1_d)
        PyMem_Free(_source_term)

    #
    #


################
### diff over space
################
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
cdef np.ndarray[np.float64_t, ndim = 1] adv_vx(
    np.ndarray[np.float64_t, ndim = 1] y
):
    '''
    advection term of x direction
    '''
    return(0.5 * np.power(y, 2.0))
    #
    #

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef np.ndarray[np.float64_t, ndim = 1] adv_vx_diff(
    np.ndarray[np.float64_t, ndim = 1] y
):
    '''
    difference of advection term
    '''
    return(y)
    #
    #

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef np.ndarray[np.float64_t, ndim = 1] source_x(
    np.ndarray[np.float64_t, ndim = 1] y,
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
        double[:, ::1]  __vx        = <double[:y.shape[0], :1]> _vx
        np.ndarray[np.float64_t, ndim = 2] vx = np.asarray(__vx)
        ### options given
        double *         _ERK       = <double *> PyMem_Malloc(y.size * sizeof(double))
        double[:, ::1]  __ERK       = <double[:y.shape[0], :1]> _ERK
        double *         _rho       = <double *> PyMem_Malloc(y.size * sizeof(double))
        double[:, ::1]  __rho       = <double[:y.shape[0], :1]> _rho
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
        double[:, ::1]  __dERK_dx   = <double[:y.shape[0], :1]> _dERK_dx
        double *         _dERK_dt   = <double *> PyMem_Malloc(y.size * sizeof(double))
        double[:, ::1]  __dERK_dt   = <double[:y.shape[0], :1]> _dERK_dt
        double *         _drho_dx   = <double *> PyMem_Malloc(y.size * sizeof(double))
        double[:, ::1]  __drho_dx   = <double[:y.shape[0], :1]> _drho_dx
        double *         _d2vx_dxdx = <double *> PyMem_Malloc(y.size * sizeof(double))
        double[:, ::1]  __d2vx_dxdx = <double[:y.shape[0], :1]> _d2vx_dxdx
        double *         _d2vx_dydy = <double *> PyMem_Malloc(y.size * sizeof(double))
        double[:, ::1]  __d2vx_dydy = <double[:y.shape[0], :1]> _d2vx_dydy
        double *         _dvx_dx    = <double *> PyMem_Malloc(y.size * sizeof(double))
        double[:, ::1]  __dvx_dx    = <double[:y.shape[0], :1]> _dvx_dx
        double *         _dERK_dtdx = <double *> PyMem_Malloc(y.size * sizeof(double)) # 125
        double[:, ::1]  __dERK_dtdx = <double[:y.shape[0], :1]> _dERK_dtdx # 125
        np.ndarray[np.float64_t, ndim = 2] dERK_dx   = np.asarray(__dERK_dx)
        np.ndarray[np.float64_t, ndim = 2] dERK_dt   = np.asarray(__dERK_dt)
        np.ndarray[np.float64_t, ndim = 2] drho_dx   = np.asarray(__drho_dx)
        np.ndarray[np.float64_t, ndim = 2] d2vx_dxdx = np.asarray(__d2vx_dxdx)
        np.ndarray[np.float64_t, ndim = 2] d2vx_dydy = np.asarray(__d2vx_dydy)
        np.ndarray[np.float64_t, ndim = 2] dvx_dx    = np.asarray(__dvx_dx)
        np.ndarray[np.float64_t, ndim = 2] dERK_dtdx = np.asarray(__dERK_dtdx)
        ### tmp values
        int                                in_shape0
        int                                k
        ### output preparation, reshape to return
        list                               out_list
        #
        double *         _out       = <double *> PyMem_Malloc(y.size * sizeof(double))
        double[:, ::1]  __out       = <double[:y.shape[0], :1]> _out
        np.ndarray[np.float64_t, ndim = 2] out = np.asarray(__out)
        #
    ### init values
    ERK[:, :]       = options[2]
    rho[:, :]       = options[3]
    dERK_dt[:, :]   = options[5]
    ### see shape
    in_shape0       = y.shape[0]
    vx[:, :]        = y.reshape((in_shape0, 1))
    #
    #
    ### calculate diff
    dERK_dx[:, :]   = di1(ERK,     0, dx)
    drho_dx[:, :]   = di1(rho,     0, dx)
    d2vx_dxdx[:, :] = di2( vx,     0, dx)
    d2vx_dydy[:, :] = di2( vx,     1, dx)
    dvx_dx[:, :]    = di1( vx,     0, dx)
    dERK_dtdx[:, :] = di1(dERK_dt, 0, dx)
    #
    ### model
    out_list = [
        -1.0 * mu0 * np.exp(-1.0 * beta * ERK) * vx,      # 0
        -2.0 * alpha * k0 * R0 * dERK_dx / rho,           # 1
        -1.0 * k0 * drho_dx * rho**(-3.0),                # 2
         eta * (d2vx_dxdx + d2vx_dydy) * rho**(-2.0),     # 3
        -1.0 * eta * drho_dx * dvx_dx * rho**(-3.0),      # 4
        -1.0 * mu0 * np.exp(-1.0 * beta * dERK_dx) * vx,  # 5
        -1.0 * mu0 * np.exp(-1.0 * beta * dERK_dt) * vx,  # 6
        -2.0 * alpha * k0 * R0 * dERK_dtdx / rho          # 7
    ]
    '''
    ====================================
    below, make the output as sum of some in the list above.
    during test, terms should be tested one by one.
    make it able to be zeros.
    ====================================
    '''
    out[:, :]  = np.zeros_like(vx)
    if len(mod_ind) > 0:
      out[:, :] += sum([out_list[k] for k in mod_ind])
    #
    try:
        return(np.array(__out).reshape(-1))
    finally:
        PyMem_Free(_vx)
        PyMem_Free(_ERK)
        PyMem_Free(_rho)
        PyMem_Free(_dERK_dx)
        PyMem_Free(_dERK_dt)
        PyMem_Free(_drho_dx)
        PyMem_Free(_d2vx_dxdx)
        PyMem_Free(_d2vx_dydy)
        PyMem_Free(_dvx_dx)
        PyMem_Free(_dERK_dtdx)
        PyMem_Free(_out)
    #
    #

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef np.ndarray[np.float64_t, ndim = 1] adv_rho(
    np.ndarray[np.float64_t, ndim = 1] y,
    list                               options
):
    '''
    advection term for rho
    input y is rho, and returns v x rho
    v is input in option[0]
    '''
    cdef:
        double *      _out = <double *> PyMem_Malloc(y.size * sizeof(double))
        double[::1]  __out = <double[:y.shape[0]]> _out
        np.ndarray[np.float64_t, ndim = 1] out = np.asarray(__out)
    #
    out[:] = options[0] * y
    try:
        return(np.array(__out))
    finally:
        PyMem_Free(_out)
    #
    #

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef np.ndarray[np.float64_t, ndim = 1] adv_rho_diff_x(
    np.ndarray[np.float64_t, ndim = 1] y,
    list                               options
):
    '''
    advection term difference of rho
    '''
    cdef:
        double *      _out = <double *> PyMem_Malloc(y.size * sizeof(double))
        double[::1]  __out = <double[:y.shape[0]]> _out
        np.ndarray[np.float64_t, ndim = 1] out = np.asarray(__out)
        #
    out[:] = options[0]
    #
    try:
        return(np.array(__out))
    finally:
        PyMem_Free(_out)
    #
    #


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef np.ndarray[np.float64_t, ndim = 1] TEST_ZERO_nopt(
    np.ndarray[np.float64_t, ndim = 1] y
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
cdef np.ndarray[np.float64_t, ndim = 1] TEST_ZERO_wopt(
    np.ndarray[np.float64_t, ndim = 1] y,
    list                               options
):
    '''
    test function, with options as input
    '''
    return(np.zeros_like(y))
    #
    #


################
### class of simulation
################
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef class Diff_vrho:
    '''
    class to run simulation of Euler discribed fields in WENO and Runge-Kutta
    #
    ### methods #######
    ### deftype    returns    dim    name
    ### cp         int        -      get_erk_store
    ### cp         ndarray    2      get_integrated
    ### cp         ndarray    3      get_ERK_blocked
    ### cp         list       -      get_fields
    ### cp         ndarray    2      get_times
    ### cp         ndarray    3      get_coord
    ### cp         ndarray    4      get_source
    ### c          ndarray    2      line_gauss
    ### c          ndarray    2      line_gauss_dt
    ### c          ndarray    1      devo_vrho
    ### cp         -          -      run_sim
    ### cp         ndarray    1      source_xcp
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
        double[:, ::1]       __integrated
        double *              _source
        double[:, :, :, ::1] __source
        np.ndarray timerange
        np.ndarray coord
        np.ndarray integrated
        np.ndarray source
        #
        double     erk_store
        double     courant ### Courant number
        list       mod_ind
        #
        int        cut_out_length
        #
        double     amp_dt
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
        list                               model_indice = [],
        double                             amp_dt       = 1.0
    ):
        ################
        ### local values
        ################
        ### comment to notify that this is in cython
        print("initializing a class Diff_vrho, coded in cython...")
        cdef:
            double *                _tmpt = <double *> PyMem_Malloc(int(tend / dt) * sizeof(double))
            double[::1]            __tmpt = <double[:int(tend / dt)]> _tmpt
            double *          _tmp_coordx = <double *> PyMem_Malloc(imgx * imgy * sizeof(double))
            double[:, ::1]   __tmp_coordx = <double[:imgx, :imgy]> _tmp_coordx
            double *          _tmp_coordy = <double *> PyMem_Malloc(imgx * imgy * sizeof(double))
            double[:, ::1]   __tmp_coordy = <double[:imgx, :imgy]> _tmp_coordy
            double *                 _yi0 = <double *> PyMem_Malloc(3 * imgx * imgy * sizeof(double))
            double[:, :, ::1]       __yi0 = <double[:3, :imgx, :imgy]> _yi0
            double *                _tmpy = <double *> PyMem_Malloc(__yi0.size * sizeof(double))
            double[::1]            __tmpy = <double[:__yi0.size]> _tmpy
            #
            np.ndarray[np.float64_t, ndim = 1] tmpt       = np.asarray(__tmpt)
            np.ndarray[np.float64_t, ndim = 2] tmp_coordx = np.asarray(__tmp_coordx)
            np.ndarray[np.float64_t, ndim = 2] tmp_coordy = np.asarray(__tmp_coordy)
            np.ndarray[np.float64_t, ndim = 3] yi0        = np.asarray(__yi0)
            np.ndarray[np.float64_t, ndim = 1] tmpy       = np.asarray(__tmpy)
        #
        try:
            yi0[:, :, :] = np.zeros((3, imgx, imgy))
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
            ### amplitudes
            self.amp_dt = amp_dt
            #
            ### Courant Number max
            self.courant   = courant
            print("courant number is %.4f" % self.courant)
            #
            #
            ### timerange setting
            tmpt[:]              = np.arange(0.0, tend, self.step_size)
            ### initialize shapes of instance attributes
            self._timerange      = <double *> PyMem_Malloc(np.asarray(tmpt).size * sizeof(double))
            self.__timerange     = <double[:np.asarray(tmpt).shape[0], :1]> self._timerange
            self.timerange       = np.asarray(self.__timerange)
            ### init values
            self.timerange[:, :] = np.array(__tmpt).reshape((tmpt.shape[0], 1))
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
            self.coord[:, :, :]     = self.dx * np.concatenate([
                    np.array(__tmp_coordx).reshape((1, self.imgx, self.imgy)),
                    np.array(__tmp_coordy).reshape((1, self.imgx, self.imgy))],
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
            tmpy[:] = np.array(__yi0).reshape(-1)
            #
            ### result store
            ### initialize shapes of instance attributes
            self._integrated      = <double *> PyMem_Malloc(np.asarray(self.timerange).shape[0] * np.asarray(tmpy).shape[0] * sizeof(double))
            self.__integrated     = <double[:np.asarray(self.timerange).shape[0], :np.asarray(tmpy).shape[0]]> self._integrated
            self.integrated       = np.asarray(self.__integrated)
            ### init values
            self.integrated[:, :] = np.zeros((self.timerange.shape[0], tmpy.shape[0]))
            self.integrated[0, :] = np.array(__tmpy)
            #
            ### source term
            self._source          = <double *> PyMem_Malloc(self.timerange.shape[0] * 2 * self.imgx * self.imgy * sizeof(double))
            self.__source         = <double[:self.timerange.shape[0], :2, :self.imgx, :self.imgy]> self._source
            self.source           = np.asarray(self.__source)
            #
            ### to record ERK
            self.erk_store = 0.0
            #
            self.cut_out_length = self.timerange.shape[0]
            #
        finally:
            PyMem_Free(_tmpt)
            PyMem_Free(_tmp_coordx)
            PyMem_Free(_tmp_coordy)
            PyMem_Free(_yi0)
            PyMem_Free(_tmpy)
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
        PyMem_Free(self._source)
        #print("deallocated")
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
            int                                j
            list                               tmp
            #
            double *               _reshaped = <double *> PyMem_Malloc(self.integrated.shape[0] * 3 * self.imgx * self.imgy * sizeof(double))
            double[:, :, :, ::1]  __reshaped = <double[:self.integrated.shape[0], :3, :self.imgx, :self.imgy]> _reshaped
            np.ndarray[np.float64_t, ndim = 4] reshaped = np.asarray(__reshaped)
            #
        ### init values
        reshaped[:, :, :, :] = self.integrated.reshape((self.integrated.shape[0], 3, self.imgx, self.imgy))
        ### fields
        tmp = [np.array(__reshaped)[0:self.cut_out_length, j, :, :] for j in range(0, 3)]
        tmp.append(self.get_ERK_blocked())
        tmp.append(self.get_ERK_dt_block())
        #
        try:
            return(tmp)
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
    cpdef get_source(self):
        return(np.array(self.__source)[0:self.cut_out_length, :, :, :])
        #
        #
        ################
        ### methods above are getting values in python, below are calculation
        ################
    cdef np.ndarray[np.float64_t, ndim = 2] line_gauss(self, double t):
        cdef:
            double            sigma_inner = self.paras[8]
            double            speed1d     = self.paras[7]
            double            speed_grow  = 0.5
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
        ### use erk_multi defined above
        beginers[:] = np.remainder(np.arange(0.0, float(self.imgx), e_e), float(self.imgx))
        erkfield[:, :] = sum([erk_multi(t, self.coord[0, :, :], sigma_inner, speed1d, speed_grow, starts) for starts in np.concatenate((beginers, beginers - float(self.imgx)))])
        #
        try:
            return(np.array(__erkfield))
        finally:
            PyMem_Free(_erkfield)
            PyMem_Free(_beginers)
        #
    cdef np.ndarray[np.float64_t, ndim = 2] line_gauss_dt(self, double t):
        cdef:
            double            sigma_inner = self.paras[8]
            double            speed1d     = self.paras[7]
            double            speed_grow  = 0.5
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
        ### use erk_dt defined above
        beginers[:] = np.remainder(np.arange(0.0, float(self.imgx), e_e), float(self.imgx))
        erkfield[:, :] = sum([erk_dt(t, self.coord[0, :, :], sigma_inner, speed1d, speed_grow, starts) for starts in np.concatenate((beginers, beginers - float(self.imgx)))])
        erkfield[:, :] = erkfield * self.amp_dt
        #
        try:
            return(np.array(__erkfield))
        finally:
            PyMem_Free(_erkfield)
            PyMem_Free(_beginers)
        #
        #
        ################
        ### model function put in WENO in Runge-Kutta
        ################
    cdef np.ndarray[np.float64_t, ndim = 1] devo_vrho(
        self,
        double t,
        np.ndarray[np.float64_t, ndim = 1] y
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
        depends on weno_1d_LF
        weno_1d_LF takes and returns 1 dimensional array
            np.ndarray[np.float64_t, ndim = 1] y
            double                             dx
            object                             advection
            object                             adv_diff
            object                             source    = None
            list                               options   = [[], [], []]
        '''
        cdef:
            double *           _y2  = <double *> PyMem_Malloc(3 * self.imgx * self.imgy * sizeof(double))
            double[:, :, ::1] __y2  = <double[:3, :self.imgx, :self.imgy]> _y2
            double *           _vx  = <double *> PyMem_Malloc(self.imgx * self.imgy * sizeof(double))
            double[::1]       __vx  = <double[:self.imgx * self.imgy]> _vx
            double *           _vy  = <double *> PyMem_Malloc(self.imgx * self.imgy * sizeof(double))
            double[::1]       __vy  = <double[:self.imgx * self.imgy]> _vy
            double *           _rho = <double *> PyMem_Malloc(self.imgx * self.imgy * sizeof(double))
            double[::1]       __rho = <double[:self.imgx * self.imgy]> _rho
            double *           _out = <double *> PyMem_Malloc(3 * self.imgx * self.imgy * sizeof(double))
            double[::1]       __out = <double[:3 * self.imgx * self.imgy]> _out
            ### temporal
            double *           _tmp = <double *> PyMem_Malloc(self.imgx * self.imgy * sizeof(double))
            double[:, ::1]    __tmp = <double[:self.imgx ,:self.imgy]> _tmp
            #
            np.ndarray[np.float64_t, ndim = 3] y2  = np.asarray(__y2)
            np.ndarray[np.float64_t, ndim = 1] vx  = np.asarray(__vx)
            np.ndarray[np.float64_t, ndim = 1] vy  = np.asarray(__vy)
            np.ndarray[np.float64_t, ndim = 1] rho = np.asarray(__rho)
            np.ndarray[np.float64_t, ndim = 1] out = np.asarray(__out)
            ### temporal
            np.ndarray[np.float64_t, ndim = 2] tmp = np.asarray(__tmp)
            #
        ### init values
        vx[:]       = np.zeros((self.imgx, self.imgy)).reshape(-1)
        vy[:]       = np.zeros((self.imgx, self.imgy)).reshape(-1) ### modify this if 2D
        rho[:]      = np.zeros((self.imgx, self.imgy)).reshape(-1)
        ### v and rho separation
        y2[:, :, :] = y.reshape((3, self.imgx, self.imgy))
        #
        ### apply WENO velocity takes itself only
        #       regarding source term of V
        #       options are...
        #           paras        list, [mu0, alpha, beta, k0, R0, eta, r]
        #           dx           double
        #           ERK          np.ndarray[np.float64_t, ndim = 1]
        #           rho          np.ndarray[np.float64_t, ndim = 1]
        #           model_indice list like [0, 1, 2, 3, 4] or [1, 2, 3, 4, 5]
        ################################
        ### direction x
        vx[:] = weno_1d_LF(
            y2[0, :, :].reshape(-1),
            self.dx,
            advection = adv_vx,
            adv_diff  = adv_vx_diff,
            source    = source_x,
            options   = [
                [],
                [],
                [self.paras, self.dx, self.line_gauss(t), y2[2, :, :], self.mod_ind, self.line_gauss_dt(t)]
            ]
        )
        ### direction y
        #vy  = weno_1d_LF(y2[1, :, :].reshape(-1), self.dx)
        #
        #
        ### apply WENO rho, takes velocity and rho
        #       regarding advection of rho
        #       options are...
        #           v in advection
        #           v in adv_diff
        ################################
        tmp[:, :] = y2[0, :, :] ### this is vx
        rho[:] = weno_1d_LF(
            y2[2, :, :].reshape(-1),
            self.dx,
            #
            advection = adv_rho,
            adv_diff  = adv_rho_diff_x,
            ### above,,, true rho dynamics
            ### below,,, during tests
            #advection = TEST_ZERO_wopt,
            #adv_diff  = TEST_ZERO_wopt,
            #
            source    = None,
            options   = [[tmp.reshape(-1)], [tmp.reshape(-1)], []]
        )
        #
        #
        ### make output
        out[:] = np.concatenate([vx, vy, rho], axis = 0).reshape(-1)
        try:
            return(np.array(__out))
        finally:
            PyMem_Free(_y2)
            PyMem_Free(_vx)
            PyMem_Free(_vy)
            PyMem_Free(_rho)
            PyMem_Free(_out)
            PyMem_Free(_tmp)
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
            double *              _u_cur       = <double *> PyMem_Malloc(self.integrated.shape[1] * sizeof(double))
            double [::1]         __u_cur       = <double[:self.integrated.shape[1]]> _u_cur
            double *              _u_pre       = <double *> PyMem_Malloc(self.integrated.shape[1] * sizeof(double))
            double [::1]         __u_pre       = <double[:self.integrated.shape[1]]> _u_pre
            np.ndarray[np.float64_t, ndim = 3] tmp_counter = np.asarray(__tmp_counter)
            np.ndarray[np.float64_t, ndim = 1] u_cur       = np.asarray(__u_cur)
            np.ndarray[np.float64_t, ndim = 1] u_pre       = np.asarray(__u_pre)
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
            u_cur[:] = self.integrated[0, :]
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
                    u_pre[:] = np.array(__u_cur)
                    #
                    #
                    ### Runge Kutta, 1st stage
                    u_cur[:] = u_pre - dt * self.devo_vrho(t_cur, u_cur)
                    ### Runge Kutta, 2nd stage
                    u_cur[:] = 0.75 * u_pre        +  0.25       * (u_cur - dt * self.devo_vrho(t_cur, u_cur))
                    ### Runge Kutta, 3rd stage
                    u_cur[:] = (1.0 / 3.0) * u_pre + (2.0 / 3.0) * (u_cur - dt * self.devo_vrho(t_cur, u_cur))
                    #
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
                    tmp_counter[:, :, :] = self.integrated[(j - 1), :].reshape((3, self.imgx, self.imgy))
                    print("    sim %.3f, loop %d, sum (v %.4f, rho %.4f), inner %d" % (timej, j, np.sum(tmp_counter[0, :, :]), np.sum(tmp_counter[2, :, :]), cou_lp))
                    c = 0
                c += 1
                #
                ### escape from too small dt i.e. something goes to inf
                if break_sim:
                    print("    FAILED dt was %f\n    simulation failed\n    too small dt emerged i.e. less than 10 to the power of %.1f\nanyway," % (dt, np.log10(kill_th)))
                    ### cut integrated and timerange
                    #self.integrated = self.integrated[0:j, :]
                    #self.timerange  = self.timerange[0:j, :]
                    self.cut_out_length = j
                    break
                #
                ### store the result
                self.integrated[j, :]   = np.array(__u_cur)
                self.source[j, 0, :, :] = source_x(
                    u_cur.reshape((3, self.imgx, self.imgy))[0, :, :].reshape(-1),
                    [
                        self.paras,
                        self.dx,
                        self.line_gauss(t_cur),
                        u_cur.reshape((3, self.imgx, self.imgy))[2, :, :],
                        self.mod_ind,
                        self.line_gauss_dt(t_cur)
                    ]
                ).reshape((self.imgx, self.imgy))
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
    cpdef source_xcp(self,
        double t
    ):
        cdef:
            double *                           _integ = <double *> PyMem_Malloc(3 * self.imgx * self.imgy * sizeof(double))
            double[:, :, ::1]                 __integ = <double[:3, :self.imgx, :self.imgy]> _integ
            np.ndarray[np.float64_t, ndim = 3]  integ = np.asarray(__integ)
            double *                           _out   = <double *> PyMem_Malloc(2 * self.imgx * self.imgy * sizeof(double))
            double[:, :, ::1]                 __out   = <double[:2, :self.imgx, :self.imgy]> _out
            np.ndarray[np.float64_t, ndim = 3]  out   = np.asarray(__out) # x, y
            int                                 ti
            int                                 tj
            double                              tf
            double                              wl
            double                              wr
            int                                 ix    = self.imgx
            int                                 iy    = self.imgy
            #
        #
        try:
            ### init values
            out[:, :, :] = 0.0
            #
            ### prepare v, rho
            ti = np.argmin(np.abs(self.timerange[:, 0] - t))
            if t < self.timerange[ti, 0]:
                ### since self.timerange[0] <= t, min ti can be 0.0
                ti -= 1
            ### float part
            tf = t - self.timerange[ti, 0]
            #
            ### weight of left and right on time line
            if self.cut_out_length <= ti + 1:
                tj = ti
                wl = 1.0
            else:
                tj = ti + 1
                wl = tf / (self.timerange[tj] - self.timerange[ti])
            wr = 1.0 - wl
            #
            integ[:, :, :] = self.integrated[ti, :].reshape((3, ix, iy)) * wl +\
                             self.integrated[tj, :].reshape((3, ix, iy)) * wr
            #
            ### returning source_x
            out[0, :, :] = source_x(
                integ[0, :, :].reshape(-1),
                [
                    self.paras,
                    self.dx,
                    self.line_gauss(t),
                    integ[2, :, :],
                    self.mod_ind,
                    self.line_gauss_dt(t)
                ]
            ).reshape((ix, iy))
            #
            return(np.array(__out))
            #
        finally:
            PyMem_Free(_integ)
            PyMem_Free(_out)
            #
        #
        #
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
