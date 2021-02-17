### on cython
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

from scipy.integrate import RK45


from cython.parallel cimport prange



cdef extern from "c_extern.hpp":
    double gauss_one_dim(
      double   mu_t,
      double   x,
      double   sigma_inner,
      double   startx,
      double   gauss_max
    ) nogil

cdef extern from "./c_skew.hpp":
    double skew_normal_1d(
      double   mu_t,
      double   x,
      double   sigma_inner,
      double   startx,
      double   gauss_max,
      double   skew_shape
    ) nogil

################
### ERK
################
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef double erk_wrap_select(
  double   mu_t,
  double   x,
  double   sigma_inner,
  double   startx,
  double   gauss_max,
  double   skew_shape
) nogil:
    ### use gauss_one_dim or skew_normal_1d
    if skew_shape == 0.0:
        return(
            gauss_one_dim(
                mu_t,
                x,
                sigma_inner,
                startx,
                gauss_max
            )
        )
    else:
        return(
            skew_normal_1d(
                mu_t,
                x,
                sigma_inner,
                startx,
                gauss_max,
                skew_shape
            )
        )

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef double erk_on(
    double t,
    double X,
    double sigma_inner,
    double speed1d,
    double skew_shape = 0.0,
    double speed_grow = 0.5,
    double mod_lim    = 64.0,
):
    cdef:
        double erk_max_t
        double c_ind
        list   dists
        double d
        double out
        #
    #
    erk_max_t   = t * speed_grow
    if erk_max_t <= 0.0:
        erk_max_t = 0.0
    elif 1.0 <= erk_max_t:
        erk_max_t = 1.0
    #
    ### make a field of distance to wave top
    out = sum([
        erk_wrap_select(
            (t * speed1d) % mod_lim, ### attention
            X % mod_lim,             ### attention
            sigma_inner,
            mod_lim * c_ind,         ### attention
            erk_max_t,
            skew_shape
        )
        for c_ind in [-1.0, 0.0, 1.0]
    ])
    return(out)
    #

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef double erk_on_dt(
    double t,
    double X,
    double sigma_inner,
    double speed1d,
    double speed_grow = 0.5,
    double mod_lim    = 64.0
):
    cdef:
        double Amp_t
        double Amp_tdt     = 0.0
        double c_ind
        list   dists
        double d
        double out
        #
    #
    ### make a field of distance to wave top
    dists = [(X % mod_lim) - (mod_lim * c_ind + (speed1d * t) % mod_lim) for c_ind in [-1.0, 0.0, 1.0]]
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
    out = sum([np.exp(-d**2.0 / (2.0 * sigma_inner**2.0)) * (Amp_tdt + Amp_t * speed1d * d / (sigma_inner**2.0)) for d in dists])
    return(out)
    #

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef double in_left(
    int    j,
    int    k,
    double d_or_s,
    int    min_index
):
    '''
    implement in non-periodic condition
    '''
    cdef:
        double out = 0.0
        #
    if j == k and not j == min_index:
        out = 1.0
    elif j == k + 1 and not j == min_index:
        out = d_or_s
    #
    return(out)
    #

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef double in_right(
    int    j,
    int    k,
    double d_or_s,
    int    max_index
):
    '''
    implement in non-periodic condition
    '''
    cdef:
        double out = 0.0
        #
    if j == k and not j == max_index - 1:
        out = d_or_s
    elif j == k - 1 and not j == max_index - 1:
        out = 1.0
    #
    return(out)
    #

################
### simulator
################
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef class Sim_Part:
    '''
    ################################
    ### simulator for particle model
    ### methods
    ### py     -       - __cinit__
    ### py     -       - __dealloc__
    ### cp     ndarray 1 step
    ### cp     -       - run_sim
    ### cp     ndarray 3 get_integrated
    ### cp     ndarray 2 get_timerange ### the timerange is to be set in cinit
    ### cp     ndarray 2 get_ERK_on
    ### cp     ndarray 3 get_xve
    ###
    ################################
    '''
    cdef:
        double *           _integrated
        double[:, :, ::1] __integrated
        np.ndarray          integrated
        double *           _timerange
        double[:, ::1]    __timerange
        np.ndarray          timerange
        double *           _ERK_on
        double[:, ::1]    __ERK_on
        np.ndarray          ERK_on
        ### diff and sum, left and right
        double *           _dl
        double[:, ::1]    __dl
        np.ndarray          dl
        double *           _dr
        double[:, ::1]    __dr
        np.ndarray          dr
        double *           _sl
        double[:, ::1]    __sl
        np.ndarray          sl
        double *           _sr
        double[:, ::1]    __sr
        np.ndarray          sr
        #
        list                paras
        double              step_size
        double              tend
        int                 num_cell
        int                 num_val
        list                model_choice
        double              amp_dt
        #
        double              skew_shape
        double              mod_lim
        #
    def __cinit__(self,
        np.ndarray[np.float64_t, ndim = 2] timerange,
        np.ndarray[np.float64_t, ndim = 2] cell_init, ### cell x (x, v)
        list                               paras,
        list                               model_choice,
        double                             amp_dt = 1.0
    ):
        cdef:
            int num_cell = cell_init.shape[0]
            int mat_size = num_cell * num_cell
            #
            int j
            int k
            #
        ### init
        self.paras           = paras
        self.model_choice    = model_choice
        print("Particle model simulation starts\naccepted model indice are")
        print(self.model_choice)
        #
        self.amp_dt          = amp_dt
        #
        self._timerange      = <double *> PyMem_Malloc(timerange.size * sizeof(double))
        self.__timerange     = <double[:timerange.shape[0], :timerange.shape[1]]> self._timerange
        self.timerange       = np.asarray(self.__timerange)
        self.timerange[:, :] = timerange
        #
        self._integrated     = <double *> PyMem_Malloc(timerange.shape[0] * cell_init.size * sizeof(double))
        self.__integrated    = <double[:timerange.shape[0], :num_cell, :cell_init.shape[1]]> self._integrated
        self.integrated      = np.asarray(self.__integrated)
        #
        self._ERK_on         = <double *> PyMem_Malloc(timerange.shape[0] * num_cell * sizeof(double))
        self.__ERK_on        = <double[:timerange.shape[0], :num_cell]> self._ERK_on
        self.ERK_on          = np.asarray(self.__ERK_on)
        #
        ### make arrays to calculate diff or sum
        self._dl             = <double *> PyMem_Malloc(mat_size * sizeof(double))
        self.__dl            = <double[:num_cell, :num_cell]> self._dl
        self.dl              = np.asarray(self.__dl)
        self._dr             = <double *> PyMem_Malloc(mat_size * sizeof(double))
        self.__dr            = <double[:num_cell, :num_cell]> self._dr
        self.dr              = np.asarray(self.__dr)
        self._sl             = <double *> PyMem_Malloc(mat_size * sizeof(double))
        self.__sl            = <double[:num_cell, :num_cell]> self._sl
        self.sl              = np.asarray(self.__sl)
        self._sr             = <double *> PyMem_Malloc(mat_size * sizeof(double))
        self.__sr            = <double[:num_cell, :num_cell]> self._sr
        self.sr              = np.asarray(self.__sr)
        #
        self.dl[:, :] = np.array([[ in_left(j, k, -1.0,        0) for k in range(0, num_cell)] for j in range(0, num_cell)])
        self.dr[:, :] = np.array([[in_right(j, k, -1.0, num_cell) for k in range(0, num_cell)] for j in range(0, num_cell)])
        self.sl[:, :] = np.array([[ in_left(j, k,  1.0,        0) for k in range(0, num_cell)] for j in range(0, num_cell)])
        self.sr[:, :] = np.array([[in_right(j, k,  1.0, num_cell) for k in range(0, num_cell)] for j in range(0, num_cell)])
        #
        ### results and first timepoint
        self.integrated[:, :, :] = np.zeros_like(self.integrated)
        self.integrated[0, :, :] = cell_init[:, :]
        #
        self.ERK_on[:, :]       = np.zeros_like(self.ERK_on)
        #
        self.num_cell           = num_cell
        self.num_val            = cell_init.shape[1]
        #
        ### skew
        self.skew_shape = paras[11]
        self.mod_lim    = paras[12]
        print(">>> skew value is %f" % self.skew_shape)
        #
        #
    def __dealloc__(self):
        PyMem_Free(self._integrated)
        PyMem_Free(self._timerange)
        PyMem_Free(self._ERK_on)
        PyMem_Free(self._dl)
        PyMem_Free(self._dr)
        PyMem_Free(self._sl)
        PyMem_Free(self._sr)
        #
        #
    cpdef np.ndarray[np.float64_t, ndim = 1] step(Sim_Part self,
        double                   t,
        np.ndarray[np.float64_t, ndim = 1] y
    ):
        '''
        put this in scipy.integrate
        '''
        cdef:
            double *                           _y2  = <double *> PyMem_Malloc(y.size * sizeof(double))
            double[:, ::1]                    __y2  = <double[:self.num_cell, :self.num_val]> _y2
            np.ndarray[np.float64_t, ndim = 2]  y2  = np.asarray(__y2)
            double *                           _out = <double *> PyMem_Malloc(y.size * sizeof(double))
            double[:, ::1]                    __out = <double[:self.num_cell, :self.num_val]> _out
            np.ndarray[np.float64_t, ndim = 2]  out = np.asarray(__out)
            #
            double *                           _xs    = <double *> PyMem_Malloc(self.num_cell * sizeof(double))
            double[:, :]                      __xs    = <double[:self.num_cell, :1]> _xs
            np.ndarray[np.float64_t, ndim = 2]  xs    = np.asarray(__xs)
            double *                           _vs    = <double *> PyMem_Malloc(self.num_cell * sizeof(double))
            double[:, :]                      __vs    = <double[:self.num_cell, :1]> _vs
            np.ndarray[np.float64_t, ndim = 2]  vs    = np.asarray(__vs)
            double *                           _Rs    = <double *> PyMem_Malloc(self.num_cell * sizeof(double))
            double[:, :]                      __Rs    = <double[:self.num_cell, :1]> _Rs
            np.ndarray[np.float64_t, ndim = 2]  Rs    = np.asarray(__Rs)
            double *                           _Ms    = <double *> PyMem_Malloc(self.num_cell * sizeof(double))
            double[:, :]                      __Ms    = <double[:self.num_cell, :1]> _Ms
            np.ndarray[np.float64_t, ndim = 2]  Ms    = np.asarray(__Ms)
            double *                           _ERKs  = <double *> PyMem_Malloc(self.num_cell * sizeof(double))
            double[:, :]                      __ERKs  = <double[:self.num_cell, :1]> _ERKs
            np.ndarray[np.float64_t, ndim = 2]  ERKs  = np.asarray(__ERKs)
            double *                           _ERKt  = <double *> PyMem_Malloc(self.num_cell * sizeof(double))
            double[:, :]                      __ERKt  = <double[:self.num_cell, :1]> _ERKt
            np.ndarray[np.float64_t, ndim = 2]  ERKt  = np.asarray(__ERKt)
            #
            ### in paras
            double                    mu0   = self.paras[0]
            double                    alpha = self.paras[1]
            double                    beta  = self.paras[2]
            double                    k0    = self.paras[3]
            double                    R0    = self.paras[4]
            double                    eta   = self.paras[5]
            double                    r     = self.paras[6]
            double                    speed = self.paras[7]
            double                    sigma = self.paras[8]
            #
            double                    tmp_x
            #
        try:
            ### init value
            y2[:, :] = y.reshape((self.num_cell, self.num_val))
            xs[:, 0] = y2[:, 0]
            vs[:, 0] = y2[:, 1]
            #
            out[:, 0] = vs[:, 0]
            #
            ERKs[:, 0] = np.array([   erk_on(t, tmp_x, sigma, speed, skew_shape = self.skew_shape, speed_grow = 0.5, mod_lim = self.mod_lim) for tmp_x in xs[:, 0]])
            ERKt[:, 0] = np.array([erk_on_dt(t, tmp_x, sigma, speed                                                                        ) for tmp_x in xs[:, 0]]) * self.amp_dt
            #
            ### model choice
            if 0 in self.model_choice:
                Ms[:, :] = -mu0 * np.exp(-beta * ERKs)
            elif 6 in self.model_choice:
                Ms[:, :] = -mu0 * np.exp(-beta * ERKt)
            else:
                Ms[:, :] = 0.0
            #
            if 1 in self.model_choice:
                Rs[:, :] = R0 * (1.0 + alpha * ERKs)
            elif 7 in self.model_choice:
                Rs[:, :] = R0 * (1.0 + alpha * ERKt)
            else:
                Rs[:, :] = 0.0
            #
            ### model, differentiation
            out[:, 1:2] = Ms  * vs\
                        - k0  * (self.sr.dot(Rs) - self.dr.dot(xs))\
                        + k0  * (self.sl.dot(Rs) - self.dl.dot(xs))\
                        + eta * (self.dr.dot(vs) - self.dl.dot(vs))
            #
            return(np.array(__out).reshape(-1))
            #
        finally:
            PyMem_Free(_y2)
            PyMem_Free(_out)
            PyMem_Free(_xs)
            PyMem_Free(_vs)
            PyMem_Free(_Rs)
            PyMem_Free(_Ms)
            PyMem_Free(_ERKs)
            PyMem_Free(_ERKt)
            #
    cpdef run_sim(self):
        cdef:
            int    c      = 1
            int    j      = 0
            int    shape0 = self.integrated.shape[1]
            int    shape1 = self.integrated.shape[2]
            double speed  = self.paras[7]
            double sigma  = self.paras[8]
            #
            ### rk,  this is python instance
            ### den, this is python instance
            #
            double timej
            double tmp_x
            #
        print("particle model simulation on cython running...")
        #
        ### integrator instance
        rk = RK45(self.step, self.timerange[0, 0], self.integrated[0, :, :].reshape(-1), np.max(self.timerange))
        rk.step()
        #
        for j in range(1, self.timerange.shape[0]):
            timej = self.timerange[j, 0]
            while rk.status == "running":
                if rk.t_old < timej and timej <= rk.t:
                    break
                    #
                rk.step()
                #
            #
            den = rk.dense_output()
            self.integrated[j, :, :] = den(timej).reshape((shape0, shape1))
            #
            ### ERK
            self.ERK_on[j, :] = np.array([erk_on(timej, tmp_x, sigma, speed, skew_shape = self.skew_shape, speed_grow = 0.5, mod_lim = self.mod_lim) for tmp_x in self.integrated[j, :, 0]])
            #
            if c >= 20:
                print("    %d th loop and the time is %s" % (j, timej))
                c = 0
            c += 1
            #
        #
        #
    cpdef get_integrated(self):
        return(np.array(self.__integrated))
        #
    cpdef get_timerange(self):
        return(np.array(self.__timerange))
        #
    cpdef get_ERK_on(self):
        return(np.array(self.__ERK_on))
        #
    cpdef get_xve(self):
        cdef:
            double *                           _out = <double *> PyMem_Malloc(self.integrated.shape[0] * self.num_cell * (self.num_val + 1) * sizeof(double))
            double[:, :, ::1]                 __out = <double[:self.integrated.shape[0], :self.num_cell, :(self.num_val + 1)]> _out
            np.ndarray[np.float64_t, ndim = 3]  out = np.asarray(__out)
            #
        try:
            out[:, :, 0:2] = self.integrated
            out[:, :, 2]   = self.ERK_on
            #
            return(np.array(__out))
            #
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


################
###
################


###
