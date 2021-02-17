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

### on cython
'''
to run simulation of 2 dimentional particle model
'''


################
### import
################
cimport cython
cimport numpy as np
import numpy as np

from numpy cimport ndarray

from cpython.mem cimport PyMem_Malloc, PyMem_Free

from scipy.spatial   import Voronoi

from cython.parallel cimport prange

from libc.math cimport exp, sin, cos, pi

################
### voronoi
################
cdef extern from "c_qhull/c_voronoi.h":
    void neighbor_pairs(double *in_arr, int *out_arr, int TOTpoints)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef void voronoi_pair(
    double * in_arr,
    int *    out_arr,
    int      TOTpoints
):
    '''
    ### wrap neighbor_pairs written in C
    ### to avoid memory alooc and delloc many times, give pointer directry.
    ### in the out_arr,
    #       [0, 0] ... number of pairs
    #       [0, 1] ... always 0
    #       later  ... each colmun indicates indice of pairs in the input.
    ### informative part in the array is,
    #       out[1:(1+out[0, 0]), :]
    '''
    #
    neighbor_pairs(in_arr, out_arr, TOTpoints)
    #


################
### ERK
################
cdef extern from "c_erk_share.h":
    double expo(
        double x
    ) nogil
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
    double[:, ::1]    X,
    double *          inv_sigma_theta,
    double            speed1d,
    double            speed_grow,
    double[:]         start_point,
    double[:]         fieldlim,
    double            theta
):
    cdef:
        int              j
        double           gauss_max = t * speed_grow
        double *        _out = <double *> PyMem_Malloc(X.shape[0] * sizeof(double))
        double[:, ::1] __out = <double[:X.shape[0], :1]> _out
        #
    try:
        ### max value
        if gauss_max < 0.0:
            gauss_max = 0.0
        elif 1.0 < gauss_max:
            gauss_max = 1.0
        #
        ### run erk_multi_c one by one
        for j in prange(0, X.shape[0], nogil = True, schedule = 'static', chunksize = 1):
            __out[j, 0] = erk_multi_c(
                t,
                X[j, 0],
                X[j, 1],
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



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef np.ndarray[np.float64_t, ndim = 1] force(
    np.ndarray[np.float64_t, ndim = 1] xy,
    double                             ymin,
    double                             ymax
):
    '''
    external force
    '''
    cdef:
        double th00 = 0.0
        double th01 = 4.0
        double *                           _out = <double *> PyMem_Malloc(2 * sizeof(double))
        double[::1]                       __out = <double[:2]> _out
        np.ndarray[np.float64_t, ndim = 1]  out = np.asarray(__out)
        #
    try:
        out[:] = 0.0
        #
        if xy[1] < ymin:
            out[1] = ymin - xy[1]
        elif ymax < xy[1]:
            out[1] = xy[1] - ymax
        #
        ### RETURN
        return(np.array(__out))
        #
    finally:
        PyMem_Free(_out)
        #
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
### class to run simulations
################
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef class Two_Part:
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
        #
        list                paras
        list                model_choice
        double              amp_dt
        #
        int                 num_cell
        int                 num_val
        #
        double              ERK_grow
        double              dt_inner
        #
        double              y_center_ERK
        double *           _sigma
        double[:, ::1]    __sigma
        np.ndarray          sigma
        #
        int *              _pairs
        #
        ### moved from step_c
        double *           _y2
        double *           _xy_only
        double *           _yT
        double *           _out
        double *           _rel_x
        double *           _rel_v
        double *           _dis
        bint   *           _far_th
        double *           _nat_2
        double *           _ERK
        double *           _nat_l
        double *           _dis2
        double *           _extF
        double *           _tmp_out
        #
        bint firstloop
        bint ext_force
        #
        double ymin
        double ymax
        #
        #
        double *        _inv_sigma
        double[:, ::1] __inv_sigma
        double *        _startpoints
        double[:]        startpoints
        double *        _fieldlims
        double[:]        fieldlims
        double           theta
        #
        #
        int              count_vor_after
        int              vor_span
        #
        bint             inh_thre
        double           thre_x_nat
        #
        #
    def __cinit__(self,
        np.ndarray[np.float64_t, ndim = 2] timerange,
        np.ndarray[np.float64_t, ndim = 2] cell_init, ### cell x (x, y, vx, vy)
        list                               paras,
        list                               model_choice,
        double                             amp_dt    = 1.0,
        double                             dt_inner  = 1.0 * (10.0 ** (-1.0)),
        bint                               ext_force = False,
        list                               field_mod = [64.0, 64.0],
        list                               startposi = [0.0, 0.0],
        double                             theta_on_pi  = 0.25,
        double                             speed_grow   = 0.5,
        int                                vor_span     = 2,
        bint                               inh_thre     = True,
        double                             thre_x_nat   = 2.5
    ):
        cdef:
            int j
            int k
            int num_cell = cell_init.shape[0]
            int lentime  = timerange.shape[0]
            #
        ###
        self.count_vor_after = 0
        self.vor_span        = vor_span
        self.inh_thre        = inh_thre   # if true, threshold is ignored
        self.thre_x_nat      = thre_x_nat # threshold x natural length
        #
        ### setting the inputs
        self.paras    = paras
        self.amp_dt   = amp_dt
        self.dt_inner = dt_inner
        #
        self._integrated  = <double *> PyMem_Malloc(lentime * num_cell * 4 * sizeof(double))
        self.__integrated = <double[:lentime, :num_cell, :4]> self._integrated
        self.integrated   = np.asarray(self.__integrated)
        self.integrated[0, :, :] = cell_init[:, :]
        #
        self._timerange   = <double *> PyMem_Malloc(timerange.size * sizeof(double))
        self.__timerange  = <double[:lentime, :timerange.shape[1]]> self._timerange
        self.timerange    = np.asarray(self.__timerange)
        self.timerange[:, :] = timerange[:, :]
        #
        self._ERK_on         = <double *> PyMem_Malloc(timerange.shape[0] * num_cell * sizeof(double))
        self.__ERK_on        = <double[:timerange.shape[0], :num_cell]> self._ERK_on
        self.ERK_on          = np.asarray(self.__ERK_on)
        self.ERK_on[:, :]    = 0.0
        #
        self.num_cell        = num_cell
        self.num_val         = cell_init.shape[1]
        #
        self.ERK_grow        = speed_grow
        #
        self.y_center_ERK    = 0.5 * np.max(cell_init[:, 1])
        self._sigma          = <double *> PyMem_Malloc(4 * sizeof(double))
        self.__sigma         = <double[:2, :2]> self._sigma
        self.sigma           = np.asarray(self.__sigma)
        self.sigma[:, :]     = np.array([[1.0 / self.paras[8], 0.0], [0.0, 4.0 / self.paras[8]]])
        #
        ### for voronoi
        self._pairs          = <int *> PyMem_Malloc(2 * 3 * num_cell * sizeof(int))
        #
        ### moved from step_c
        self._y2        = <double *> PyMem_Malloc(self.num_cell * 4 * sizeof(double))
        self._xy_only   = <double *> PyMem_Malloc(self.num_cell * 2 * sizeof(double))
        self._yT        = <double *> PyMem_Malloc(self.num_cell * 4 * sizeof(double))
        self._out       = <double *> PyMem_Malloc(self.num_cell * 4 * sizeof(double))
        self._rel_x     = <double *> PyMem_Malloc(self.num_cell * self.num_cell * 2 * sizeof(double))
        self._rel_v     = <double *> PyMem_Malloc(self.num_cell * self.num_cell * 2 * sizeof(double))
        self._dis       = <double *> PyMem_Malloc(self.num_cell ** 2 * sizeof(double))
        self._far_th    = <bint   *> PyMem_Malloc(self.num_cell ** 2 * sizeof(bint))
        self._nat_2     = <double *> PyMem_Malloc(self.num_cell ** 2 * sizeof(double))
        self._ERK       = <double *> PyMem_Malloc(self.num_cell * sizeof(double))
        self._nat_l     = <double *> PyMem_Malloc(self.num_cell * sizeof(double))
        self._dis2      = <double *> PyMem_Malloc(self.num_cell ** 2 * sizeof(double))
        self._extF      = <double *> PyMem_Malloc(self.num_cell * 2 * sizeof(double))
        self._tmp_out   = <double *> PyMem_Malloc(self.num_cell * 4 * sizeof(double))
        #
        self.firstloop = True
        self.ext_force = ext_force
        #
        self.ymin = np.amin(cell_init[:, 1])
        self.ymax = np.amax(cell_init[:, 1])
        #
        #
        self.theta = pi * theta_on_pi
        self._inv_sigma  = <double *> PyMem_Malloc(4 * sizeof(double))
        self.__inv_sigma = <double[:2, :2]> self._inv_sigma
        np.asarray(self.__inv_sigma)[:, :] = angle22(self.theta).dot(np.array([
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
    def __dealloc__(self):
        PyMem_Free(self._integrated)
        PyMem_Free(self._timerange)
        PyMem_Free(self._ERK_on)
        PyMem_Free(self._sigma)
        PyMem_Free(self._pairs)
        #
        ### moved from step_c
        PyMem_Free(self._y2)
        PyMem_Free(self._yT)
        PyMem_Free(self._xy_only)
        PyMem_Free(self._out)
        PyMem_Free(self._rel_x)
        PyMem_Free(self._rel_v)
        PyMem_Free(self._dis)
        PyMem_Free(self._far_th)
        PyMem_Free(self._nat_2)
        PyMem_Free(self._ERK)
        PyMem_Free(self._nat_l)
        PyMem_Free(self._dis2)
        PyMem_Free(self._extF)
        PyMem_Free(self._tmp_out)
        #
        #
    cpdef get_integrated(self):
        return(np.array(self.__integrated))
        #
    cpdef get_timerange(self):
        return(np.array(self.__timerange))
        #
    cpdef get_ERK_on(self):
        cdef int j
        for j in range(self.__timerange.shape[0]):
            self.ERK_on[j, :] = self.erk_self(self.__timerange[j, 0], self.__integrated[j, :, 0:2])[:, 0]
        return(np.array(self.__ERK_on))
        #
        #
    cdef np.ndarray[np.float64_t, ndim = 2] erk_self(Two_Part self,
        double         t,
        double[:, ::1] y
    ):
        return(
            erk_shared(
                t,
                y,
                self._inv_sigma,
                self.paras[7],
                self.ERK_grow,
                self.startpoints,
                self.fieldlims,
                self.theta
            )
        )
        #
    cdef np.ndarray[np.float64_t, ndim = 1] step_c(Two_Part self,
        double                             t,
        np.ndarray[np.float64_t, ndim = 1] y
    ):
        '''
        put this in scipy.integrate
        '''
        cdef:
            ### settings
            double                              c_th      = 1.41 # const thre
            ### arrays
            double[:, ::1]                    __y2        = <double[:self.num_cell, :self.num_val]> self._y2
            np.ndarray[np.float64_t, ndim = 2]  y2        = np.asarray(__y2)
            double[:, ::1]                    __xy_only   = <double[:self.num_cell, :2]> self._xy_only
            np.ndarray[np.float64_t, ndim = 2]  xy_only   = np.asarray(__xy_only)
            double[:, ::1]                    __yT        = <double[:self.num_val, :self.num_cell]> self._yT
            np.ndarray[np.float64_t, ndim = 2]  yT        = np.asarray(__yT) ### transposed for C continuity
            double[:, ::1]                    __out       = <double[:self.num_cell, :self.num_val]> self._out
            np.ndarray[np.float64_t, ndim = 2]  out       = np.asarray(__out)
            #
            double[:, ::1]                    __tmp_out   = <double[:self.num_val, :self.num_cell]> self._tmp_out
            np.ndarray[np.float64_t, ndim = 2]  tmp_out   = np.asarray(__tmp_out)
            #
            double[:, :, ::1]                 __rel_x     = <double[:self.num_cell, :2, :self.num_cell]> self._rel_x
            np.ndarray[np.float64_t, ndim = 3]  rel_x     = np.asarray(__rel_x)
            double[:, :, ::1]                 __rel_v     = <double[:self.num_cell, :2, :self.num_cell]> self._rel_v
            np.ndarray[np.float64_t, ndim = 3]  rel_v     = np.asarray(__rel_v)
            double[:, ::1]                    __dis       = <double[:self.num_cell, :self.num_cell]> self._dis
            np.ndarray[np.float64_t, ndim = 2]  dis       = np.asarray(__dis)
            #
            bint[:, ::1]                      __far_th    = <bint[:self.num_cell, :self.num_cell]> self._far_th
            #
            double[:, ::1]                    __nat_2     = <double[:self.num_cell, :self.num_cell]> self._nat_2
            np.ndarray[np.float64_t, ndim = 2]  nat_2     = np.asarray(__nat_2)
            double[:, ::1]                    __ERK       = <double[:self.num_cell, :1]> self._ERK
            np.ndarray[np.float64_t, ndim = 2]  ERK       = np.asarray(__ERK)
            double[:, ::1]                    __nat_l     = <double[:self.num_cell, :1]> self._nat_l
            np.ndarray[np.float64_t, ndim = 2]  nat_l     = np.asarray(__nat_l)
            double[:, ::1]                    __dis2      = <double[:self.num_cell, :self.num_cell]> self._dis2
            np.ndarray[np.float64_t, ndim = 2]  dis2      = np.asarray(__dis2)
            #
            double[:, ::1]                    __extF      = <double[:self.num_cell, :2]> self._extF
            np.ndarray[np.float64_t, ndim = 2]  extF      = np.asarray(__extF)
            #
            ### voronoi
            int[:, ::1]                       __pairs     = <int[:(self.num_cell * 3), :2]> self._pairs
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
            ### tmp
            int                       j     = 0
            int                       k     = 0
            int                       l     = 0
            int                       tmp_int
            #
            double                    tmp_double
            double                    tmp_d2
            double                    tmp_pres
            double                    tmp_visc
            #
            bint                      external_force = self.ext_force #False
            #
            double                    gauss_max
            #
        ################
        ### y preparation
        ################
        y2[:, :] = y.reshape((self.num_cell, 4))
        yT[:, :] = y2.T
        xy_only[:, :] = y2[:, 0:2]
        #
        #
        ################
        ### check if a pair of cells are neighbor_pairs
        ################
        #
        ### voronoi, set self._pairs as an output pointer
        if self.count_vor_after == 0:
            ### initialize the arrays
            __pairs[...]  = 0
            __far_th[...] = False ### based on voronoi below
            #
            ### run voronoi
            voronoi_pair(self._xy_only, self._pairs, self.num_cell)
            #
            '''
            ### informative part in the array is,
            ###     out[1:(1+out[0, 0]), :]
            '''
            tmp_int = __pairs[0, 0]
            #
            #for k in prange(1, (1 + tmp_int), nogil = True, schedule = 'static', chunksize = 1):
            for k in range(1, (1 + tmp_int)):
                ### not parallel, because of random access
                ### 1 sec faster per 0.5 sim time
                ### indice
                j = __pairs[k, 0]
                l = __pairs[k, 1]
                ### record the pair
                __far_th[j, l] = True
                __far_th[l, j] = True
                #
                #
            ### otherwise, previous calculated __pair and __far_th is used
        self.count_vor_after += 1
        if self.vor_span <= self.count_vor_after:
            self.count_vor_after = 0
        #
        ################
        ### ERK and natural length
        ################
        gauss_max      = t * self.ERK_grow
        if gauss_max < 0.0:
            gauss_max = 0.0
        elif 1.0 < gauss_max:
            gauss_max = 1.0
        #
        for j in range(0, self.num_cell):
            __ERK[j, :] = gauss_multi(
                t,
                __y2[j, 0],
                __y2[j, 1],
                self.fieldlims[0],
                self.fieldlims[1],
                self.startpoints[0],
                self.startpoints[1],
                self._inv_sigma,
                self.theta,
                gauss_max,
                speed #self.paras[7],
            )
            #
            '''
            below cannot be in prange. it depends on other loop.
            '''
            ### natural length
            tmp_d2 = R0 * (1.0 + alpha * __ERK[j, 0])
            __nat_l[j, 0] = tmp_d2
            #
            for l in range(0, j): ### this makes l < j
                ### pair of natural length
                __nat_2[j, l] = tmp_d2 + __nat_l[l, 0]
                __nat_2[l, j] = __nat_2[j, l]
                #
        #
        ################
        ### force (in model) calculation
        ################
        __tmp_out[...] = 0.0 ### moved from below
        for j in prange(
            0, self.num_cell,
            nogil = True, schedule = 'static', chunksize = 1
        ):
            for l in range(self.num_cell):
                for k in range(2):
                    __rel_x[j, k, l] = __yT[ k,      l] - __y2[j,  k]
                    __rel_v[j, k, l] = __yT[(k + 2), l] - __y2[j, (k + 2)]
                #
                ### distance between 2 cells
                __dis[j, l] = (__rel_x[j, 0, l] ** 2.0 + __rel_x[j, 1, l] ** 2.0) ** 0.5
                #
                ### make sure not divided by 0.0
                tmp_double   = __dis[j, l]
                __dis2[j, l] = tmp_double + float(tmp_double == 0.0)
                #
            #
            #
            ''' originally, some lines below were out of prange of j '''
            ### there was voronoi calculation here, out-of-prange
            ################
            ### force
            ################
            for k in range(2, 4):
                __out[j, k] = -mu0 * expo(-beta * __ERK[j, 0]) * __y2[j, k]
            #
            #
            ''' originally out-of-prange part end '''
            for k in range(2):
                for l in range((j + 1), self.num_cell): ### instead of j < l
                    if __far_th[j, l] \
                    and (\
                         self.inh_thre \
                         or \
                         __dis[j, l] <= self.thre_x_nat * __nat_2[j, l] \
                    ):
                        ### pressure and viscosity
                        tmp_pres = k0  * __rel_x[j, k, l] * ((__dis[j, l] - __nat_2[j, l]) / __dis2[j, l])
                        tmp_visc = eta * __rel_v[j, k, l]
                        tmp_d2   = tmp_pres + tmp_visc
                        #
                        __out[j, (k + 2)] += tmp_d2
                        __tmp_out[(k + 2), l] -= tmp_d2 ### c continuity
                        #
        out[:, :] += tmp_out.T
        #
        #
        if external_force:
            out[:, 2:4] += np.array([force(y2[j, 0:2], self.ymin, self.ymax) for j in range(y2.shape[0])]) * 1.6
        #
        #
        ################
        ### xy change rate is velocity
        ################
        out[:, 0:2] = y2[:, 2:4]
        #
        #
        return(np.array(__out).reshape(-1))
        #
        #
    cpdef run_sim_c(Two_Part self):
        cdef:
            int c = 1
            int j = 0
            int    shape0 = self.integrated.shape[1]
            int    shape1 = self.integrated.shape[2]
            double speed  = self.paras[7]
            double sigma  = self.paras[8]
            #
            double timej
            #
            double dt     = self.dt_inner
            double timed  = self.timerange[0, 0]
            #
            double *                           _k1 = <double *> PyMem_Malloc(self.num_cell * self.num_val * sizeof(double))
            double[::1]                       __k1 = <double[:(self.num_cell * self.num_val)]> _k1
            np.ndarray[np.float64_t, ndim = 1]  k1 = np.asarray(__k1)
            double *                           _k2 = <double *> PyMem_Malloc(self.num_cell * self.num_val * sizeof(double))
            double[::1]                       __k2 = <double[:(self.num_cell * self.num_val)]> _k2
            np.ndarray[np.float64_t, ndim = 1]  k2 = np.asarray(__k2)
            double *                           _k3 = <double *> PyMem_Malloc(self.num_cell * self.num_val * sizeof(double))
            double[::1]                       __k3 = <double[:(self.num_cell * self.num_val)]> _k3
            np.ndarray[np.float64_t, ndim = 1]  k3 = np.asarray(__k3)
            double *                           _k4 = <double *> PyMem_Malloc(self.num_cell * self.num_val * sizeof(double))
            double[::1]                       __k4 = <double[:(self.num_cell * self.num_val)]> _k4
            np.ndarray[np.float64_t, ndim = 1]  k4 = np.asarray(__k4)
            #
            double *                           _tmp_x = <double *> PyMem_Malloc(self.num_cell * self.num_val * sizeof(double))
            double[::1]                       __tmp_x = <double[:(self.num_cell * self.num_val)]> _tmp_x
            np.ndarray[np.float64_t, ndim = 1]  tmp_x = np.asarray(__tmp_x)
            #
        print("particle model in 2 dimensions on cython cdef starts.")
        #
        ### initial value
        tmp_x[:] = self.integrated[0, :, :].reshape(-1)
        #
        for j in range(1, self.timerange.shape[0]):
            timej = self.timerange[j, 0]
            while timed < timej:
                '''
                step forward with smaller step,
                with getting timed and tmp_x forward
                '''
                ### time inside
                if timed + dt < timej:
                    dt = self.dt_inner
                else:
                    ### in this case, the while loop breaks at this end
                    dt = timej - timed
                    #
                ### runge kutta
                k1[:] = self.step_c( timed,              tmp_x)
                k2[:] = self.step_c((timed + 0.5 * dt), (tmp_x + 0.5 * dt * k1))
                k3[:] = self.step_c((timed + 0.5 * dt), (tmp_x + 0.5 * dt * k2))
                k4[:] = self.step_c((timed +       dt), (tmp_x +       dt * k3))
                #
                ### the integrated result is at timed + dt
                tmp_x[:] += dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
                #
                ### in the end of while loop,
                timed += dt
                #print("%8.3f" % timed)
                #
            ### allocate the integrated into the result array
            self.integrated[j, :, :] = tmp_x.reshape((self.num_cell, self.num_val))
            #
            #
            if c >= 20 or self.firstloop:
                #if True:
                #
                print("    %d th loop and the time is %s" % (j, timej))
                c = 0
                if self.firstloop:
                    self.firstloop = False
                    c = 1
            c += 1
            #
            #
        PyMem_Free(_k1)
        PyMem_Free(_k2)
        PyMem_Free(_k3)
        PyMem_Free(_k4)
        PyMem_Free(_tmp_x)
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
