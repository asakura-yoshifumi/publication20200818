#!/usr/bin/env python
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


### to define functions to run on the ode

################
### imports
################
import numpy as np
import time
import sys

import os
import shutil

import pyper

from scipy.integrate import RK45, Radau

import math
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib import animation as ani

import pandas as pd
import traceback as tb

from numba import jit, jitclass
from numba import f8, i4, void

### handwrite
from .fun_dir import di1
from .fun_dir import di2
from .fun_dir import dir_reset

from .fun_old import Euler_diff as ED
from .fun_manual import R_manual

################
### animation of results
################
### moved out to another file


################
###
################
class Time_values:
    yet_ptintR = True ### class value
    def __init__(self, timerange, fields, paras, dx, coord, draw_steps = 400, ind_VxyRhoERK = [0, 1, 2, 3]):
        '''
        timerange is oberved time in input field
        '''
        self.dx    = dx
        self.coord = coord
        #
        print("initializing Time_values...")
        #
        ### for slicing
        tmp = divmod(timerange.shape[0], draw_steps)
        self.slice_step = tmp[0]
        if tmp[1] > 0.5:
            self.slice_step += 1
        #
        ### prepare for graph, inputs
        self.time_tosee = timerange[0:timerange.shape[0]:self.slice_step, 0]
        #
        ### order of input
        self.fields = []
        for j in range(0, len(ind_VxyRhoERK)):
            self.fields.append(fields[ind_VxyRhoERK[j]][0:timerange.shape[0]:self.slice_step, :, :])
        #
        ### for graph drawing
        self.set_by_paras = "mu0=%04.3f, alpha=%04.3f, beta=%04.3f\nk0=%04.3f, R0=%04.3f, eta=%04.3f\nr=%04.3f, erkspeed=%04.3f, sigma=%04.3f" % tuple([x for x in paras[0:9]])
        #
        print("Time_values were initialized")
        #
        #
    def draw_heat(self, outname, Vxname = None, Vyname = None, Rhoname = None, ERKname = None, choice_var = [0, 1, 2, 3], Rdata = None, another_color = {}):
        testing = False
        print("drawing...")
        #
        r = pyper.R(use_numpy = "True")
        r("library('tidyverse')")
        ### assignmant
        r.assign("time_tosee", self.time_tosee)
        #
        ### prepare to draw
        cen = int(self.fields[0].shape[2] / 2)
        fields_draw = [fs[:, :, cen] for fs in [self.fields[j] for j in choice_var]]
        if testing:
            print([fs.shape for fs in self.fields])
            print([fs.shape for fs in fields_draw])
        #
        #
        ### rho, erk, vx, vy
        factors = [["Vx", "Vy", "Rho", "ERK"][j] for j in choice_var]
        #
        ### add on R one by one
        for j in range(0, len(factors)):
            r.assign("df", fields_draw[j])
            comg = [
                "df <- as.data.frame(df)",
                "colnames(df) <- sapply(formatC(1:length(df), width = 4, flag = '0'), function(x){return(paste('', x, sep = ''))})",
                "df$time <- time_tosee",
                "df <- tidyr::gather(df, key = x, value = %s, -time)" % factors[j]
            ]
            if j == 0:
                comg.append("df0 <- df")
            else:
                comg.append("df0 <- cbind(df0, df[c(-1, -2)])")
                #
            if testing:
                print(r("\n".join(comg)))
                print(r("print(str(df0))"))
            else:
                r("\n".join(comg))
            print("%s assigned" % factors[j])
            #
        #
        ### add factors one by one
        comg = ["df0 <- tidyr::gather(df0, key = factors, value = value,"]
        for j in range(0, len(factors)):
            comg.append(" %s," % factors[j])
        comg.append(")")
        comg  = ["".join(comg)] ### addtion end
        comg.append("df0$x <- as.integer(df$x) * %s" % self.dx)
        if testing:
            print(r("\n".join(comg)))
        else:
            r("\n".join(comg))
        '''
        if not Rdata is None:
            tmp_dir = Rdata.rsplit("/", 1)[0]
            dir_reset(tmp_dir, False)
            #r("save(df0, file = '%s')" % Rdata)
            if os.path.exists(Rdata):
                print("saved %s" % Rdata)
            else:
                print("failed %s" % Rdata)
        '''
        #
        #print(r("print(str(df0))"))
        ### graph
        scales = [
            ") + scale_fill_viridis_c(",
            "\n".join([
                ") + scale_fill_gradient2(",
                "  low = 'blue', high = 'red', mid = 'white', midpoint = 0.0"
            ])
        ]; #print(scales)
        eachname = [
            [outname, another_color.get("all")],
            [Vxname,  another_color.get("Vx")],
            [Vyname,  another_color.get("Vy")],
            [Rhoname, another_color.get("Rho")],
            [ERKname, another_color.get("ERK")]
        ]
        for j in [0, 1]:
            outname_in, Vxname_in, Vyname_in, Rhoname_in, ERKname_in = [tmp[j] for tmp in eachname]
            #print([outname_in, Vxname_in, Vyname_in, Rhoname_in, ERKname_in])
            scale_string = scales[j]
            comg = [
                "e000 <- ggplot(df0, aes(x, time)) + geom_tile(",
                "  aes(fill = value)",
                ") + facet_wrap(",
                "  . ~ factors, ncol = 2", ### ncol and nrow
                ") + scale_y_reverse(",
                ")",
                "g000 <- e000 + theme_classic(",
                scale_string,
                ") + labs(",
                "  title = '%s'" % self.set_by_paras,
                ")",
                "e010   <- g000",
                "df_eul <- df0"
            ]
            if testing:
                print(r("\n".join(comg)))
            else:
                r("\n".join(comg))
            #
            if not Rdata is None and j == 0:
                R_manual([
                    "\n################",
                    "### euler",
                    "################\n"
                ])
                R_manual(comg) ### be sure no change of comg
                tmp_dir = Rdata.rsplit("/", 1)[0]
                dir_reset(tmp_dir, False)
                #r("save(df0, file = '%s')" % Rdata)
                r("save(list = c('df_eul', 'e000', 'e010'), file = '%s')" % Rdata)
                if os.path.exists(Rdata):
                    print("saved %s" % Rdata)
                else:
                    print("failed %s" % Rdata)
                    #
                #
            #
            if not outname_in is None:
                comg = [
                    "ggsave(plot = g000, file = '%s', width = 160, height = 120, unit = 'mm')" % outname_in
                ]
                r("\n".join(comg))
                ### check the drawn
                if os.path.exists(outname_in):
                    print("saved %s" % outname_in)
                else:
                    print("failed to save %s" % outname_in)
                #
            ### print values set on R only one time
            if Time_values.yet_ptintR:
                Time_values.yet_ptintR = False
                r("uni <- unique(df0$x)")
                r("print(list(all(uni == sort(uni)), uni))")
            #
            if not Vxname_in is None:
                comg = [
                    "df <- subset(df0, factors == 'Vx')",
                    "g000 <- ggplot(df, aes(x, time)) + geom_tile(",
                        "aes(fill = value)",
                    ") + theme_classic(",
                    ") + facet_wrap(",
                        ". ~ factors",
                    scale_string,
                    ") + scale_y_reverse(",
                    ") + labs(",
                        "title = '%s'" % self.set_by_paras,
                    ")",
                    "ggsave(plot = g000, file = '%s', width = 160, height = 120, unit = 'mm')" % Vxname_in
                ]
                r("\n".join(comg))
                if os.path.exists(Vxname_in):
                    print("saved %s" % Vxname_in)
                else:
                    print("failed %s" % Vxname_in)
            else:
                print("Vx not selected")
                print(Vxname_in)
            #
            if not Vyname_in is None:
                comg = [
                    "df <- subset(df0, factors == 'Vy')",
                    "g000 <- ggplot(df, aes(x, time)) + geom_tile(",
                        "aes(fill = value)",
                    ") + theme_classic(",
                    ") + facet_wrap(",
                        ". ~ factors",
                    scale_string,
                    ") + scale_y_reverse(",
                    ") + labs(",
                        "title = '%s'" % self.set_by_paras,
                    ")",
                    "ggsave(plot = g000, file = '%s', width = 160, height = 120, unit = 'mm')" % Vyname_in
                ]
                r("\n".join(comg))
                if os.path.exists(Vyname_in):
                    print("saved %s" % Vyname_in)
                else:
                    print("failed %s" % Vyname_in)
            #
            if not Rhoname_in is None:
                comg = [
                    "df <- subset(df0, factors == 'Rho')",
                    "g000 <- ggplot(df, aes(x, time)) + geom_tile(",
                        "aes(fill = value)",
                    ") + theme_classic(",
                    ") + facet_wrap(",
                        ". ~ factors",
                    scale_string,
                    ") + scale_y_reverse(",
                    ") + labs(",
                        "title = '%s'" % self.set_by_paras,
                    ")",
                    "ggsave(plot = g000, file = '%s', width = 160, height = 120, unit = 'mm')" % Rhoname_in
                ]
                r("\n".join(comg))
                if os.path.exists(Rhoname_in):
                    print("saved %s" % Rhoname_in)
                else:
                    print("failed %s" % Rhoname_in)
            #
            if not ERKname_in is None:
                comg = [
                    "df <- subset(df0, factors == 'ERK')",
                    "g000 <- ggplot(df, aes(x, time)) + geom_tile(",
                        "aes(fill = value)",
                    ") + theme_classic(",
                    ") + facet_wrap(",
                        ". ~ factors",
                    scale_string,
                    ") + scale_y_reverse(",
                    ") + labs(",
                        "title = '%s'" % self.set_by_paras,
                    ")",
                    "ggsave(plot = g000, file = '%s', width = 160, height = 120, unit = 'mm')" % ERKname_in
                ]
                r("\n".join(comg))
                if os.path.exists(ERKname_in):
                    print("saved %s" % ERKname_in)
                else:
                    print("failed %s" % ERKname_in)
        #
        #


################
###
################
@jit(nopython = True)
def field1_over_time(field_t, xy_chr, dx):
    f0 = np.zeros_like(field_t)
    for k in range(0, f0.shape[0]):
        f0[k, :, :] = di1(field_t[k, :, :], xy_chr, dx)
    return(f0)
    #
    #

@jit(nopython = True)
def field2_over_time(field_t, xy_chr, dx):
    f0 = np.zeros_like(field_t)
    for k in range(0, f0.shape[0]):
        f0[k, :, :] = di2(field_t[k, :, :], xy_chr, dx)
    return(f0)
    #
    #

def sep_if(x, y):
    pass
    '''
    ### define similar as a class method
    xy = [x, y]; #print(xy)
    sep_xy = np.frompyfunc(math.modf, 1, 2)(xy)
    #
    ### indice
    out = [int(sep_xy[1][0]), int(sep_xy[1][1])]
    #
    fx = sep_xy[0][0]
    fy = sep_xy[0][1]
    #
    ### area weight, (j, k), (j+1, k), (j, k+1), (j+1, k+1)
    out.append((1.0 - fx) * (1.0 - fy))
    out.append(fx         * (1.0 - fy))
    out.append((1.0 - fx) *         fy)
    out.append(fx         *         fy)
    #
    #print(out)
    return(out)
    '''
    #

@jit(nopython = True)
def loc_value_in(sepout, field2d):
    ### here, sepout expects an output of sep_if function, over nparray, through frompyfunc
    out = np.zeros((sepout.shape[0], 1))
    shape0 = field2d.shape[0]
    shape1 = field2d.shape[1]
    for c in range(0, len(sepout)):
        ### loop for cells
        sepc = sepout[c, :]
        ### values, counterparts of indice in the sepc, wight
        ### periodic
        i00 =  int(sepc[0])      % shape0
        i01 = (int(sepc[0]) + 1) % shape0
        i10 =  int(sepc[1])      % shape1
        i11 = (int(sepc[1]) + 1) % shape1
        #
        v2 = sepc[2] * field2d[i00, i10]
        v3 = sepc[3] * field2d[i01, i10]
        v4 = sepc[4] * field2d[i00, i11]
        v5 = sepc[5] * field2d[i01, i11]
        #
        out[c, 0] = v2 + v3 + v4 + v5
        #
    return(out)
    #
    #


################
### using Radau after 142 (tested in 141 only in ssh)
################
spec = [
    ("num_cell",      i4),
    ("ERK",           f8[:, :, :]),
    ("dx",            f8),
    ("timerange",     f8[:, :]),
    ("vxy",           f8[:, :, :, :]),
    ("integrated",    f8[:, :]),
    ("loc_erk_store", f8[:, :])
]
#@jitclass(spec)
class Pon_Euler:
    def __init__(
        self,
        initial_XY,
        fields,
        timerange,
        #dt,
        dx,
        ind_Vxy = [0, 1],
        ind_erk = 3
    ):
        self.num_cell  = initial_XY.shape[0]
        self.ERK       = fields[:, ind_erk, :, :] #ERK
        #
        self.dx        = dx
        self.timerange = timerange; #print(self.timerange.shape)
        #
        self.vxy       = np.zeros((
            2,
            fields.shape[0],
            fields.shape[2],
            fields.shape[3]
        ))
        for j in range(0, len(ind_Vxy)):
            ### list comprehension refers the 0th index as like lists
            self.vxy[j, :, :, :] = fields[:, ind_Vxy[j], :, :]
            #
        #
        #
        print("particles simulation starts")
        #
        ### time
        #self.step_size = dt
        #
        ### first v setting
        j = 0
        tmp_to_save = initial_XY[:, 0:2]
        loc_ws = np.array([self.sep_if(tmp_to_save[k0, 0], tmp_to_save[k0, 1]) for k0 in range(tmp_to_save.shape[0])])
        #
        loc_v       = [loc_value_in(loc_ws, fs[j, :, :]) for fs in self.vxy]
        tmp_to_save = np.concatenate(
            (tmp_to_save, loc_v[0], loc_v[1]),
            axis = 1
        )
        tmp         = tmp_to_save.reshape(-1); #print(tmp)
        #
        self.integrated       = np.zeros((self.timerange.shape[0], tmp.shape[0]))
        self.integrated[0, :] = tmp
        #
        self.loc_erk_store    = np.zeros((self.timerange.shape[0], self.num_cell))
        #
        #
    def get_integ(self):
        return(self.integrated.reshape((self.timerange.shape[0], self.num_cell, 4)))
        #
    def get_time(self):
        return(self.timerange)
        #
    def get_erk(self):
        return(self.loc_erk_store)
        #
    def sep_if(self, x, y):
        xy = [j / self.dx for j in [x, y]]
        fx, ix = math.modf(xy[0])
        fy, iy = math.modf(xy[1])
        #
        ### index
        out = [ix, iy] ### float
        #
        ### area weight, (j, k), (j+1, k), (j, k+1), (j+1, k+1)
        out.append((1.0 - fx) * (1.0 - fy))
        out.append(fx         * (1.0 - fy))
        out.append((1.0 - fx) *         fy)
        out.append(fx         *         fy)
        #
        return(np.array(out))
        #
    def step(self, t, y):
        ### y is list of 2, [x, y] diff to t
        xyvv    = y.reshape((self.num_cell, 2))
        loc_ws  = np.array([
            self.sep_if(xyvv[k0, 0], xyvv[k0, 1])
            for k0 in range(self.num_cell)
        ])
        #
        ### calculate local values on the cells positions
        ### index
        ti = np.argmin(np.abs(self.timerange - t))
        if t < self.timerange[ti]:
            ### since self.timerange[0] <= t, min ti can be 0.0
            ti -= 1
        ### float part
        tf = t - self.timerange[ti]
        #
        ### weight of left and right on time line
        if self.timerange.shape[0] - 1 <= ti:
            ti = self.timerange.shape[0] - 1
            tj = ti
            wl = 1.0
        else:
            tj = ti + 1
            wl = tf / (self.timerange[tj] - self.timerange[ti])
        wr = 1.0 - wl
        #
        ### list of 2, shape (cells x 1)
        loc_vxy = [loc_value_in(loc_ws, fs[ti, :, :]) for fs in self.vxy]
        ### list of 2, shape (cells x 1)
        lod_vxy = [loc_value_in(loc_ws, fs[tj, :, :]) for fs in self.vxy]
        #
        loc_vxy = np.concatenate(tuple(loc_vxy), axis = 1)
        lod_vxy = np.concatenate(tuple(lod_vxy), axis = 1)
        #
        w_sum   = loc_vxy * wl + lod_vxy * wr
        return(w_sum.reshape(-1))
        #
    def run_sim(self):
        '''
        run simulation
        with step that accepts only coordinate
        v is given
        '''
        print("integration starts for particles...")
        #
        coord_only = self.integrated[0, :].reshape((self.num_cell, 4))[:, 0:2].reshape(-1)
        c = 1
        #
        ### prepare the integrator
        #rk = RK45(self.step, self.timerange[0, 0], coord_only, np.max(self.timerange))
        rk = Radau(self.step, self.timerange[0, 0], coord_only, np.max(self.timerange))
        #
        ### ERK
        j = 0
        xyvv = self.integrated[j, :].reshape((self.num_cell, 4))
        self.loc_erk_store[j, :] = loc_value_in(
            np.array([self.sep_if(xyvv[k0, 0], xyvv[k0, 1]) for k0 in range(self.num_cell)]),
            self.ERK[j, :, :]
        ).reshape(-1)
        #
        ### preparetion for loop simulation, first step makes t_old as 0.0
        rk.step(); #print([rk.t, rk.t_old])
        #
        ### run simulation
        no_error_yet = True
        for j in range(1, self.timerange.shape[0]):
            timej = self.timerange[j, 0]
            while rk.status == "running":
                if rk.t_old < timej and timej <= rk.t:
                    break
                    #
                ### the step below is skipped
                ### while timej is calculated by dense_output
                try:
                    rk.step()
                except:
                    if no_error_yet:
                        no_error_yet = False
                        print("\n".join([
                            "error occured during run_sim",
                            "    at loop %d and time %f"  % (j, timej),
                            "    on RK: old %f and now %f" % (rk.t_old, rk.t),
                        ]))
                        #
                        print("\n")
                        tb.print_exc()
                        print("\n")
                    else:
                        break
            den = rk.dense_output()
            tmp_to_save = den(timej).reshape(-1)
            #
            ### save the result
            tmp_to_save = tmp_to_save.reshape((self.num_cell, 2))
            loc_ws      = np.frompyfunc(self.sep_if, 2, 1)(
                tmp_to_save[:, 0],
                tmp_to_save[:, 1]
            )
            loc_ws      = np.array([
                self.sep_if(tmp_to_save[k0, 0], tmp_to_save[k0, 1])
                for k0 in range(self.num_cell)
            ])
            loc_v       = [loc_value_in(loc_ws, fs[j, :, :]) for fs in self.vxy]
            tmp_to_save = np.concatenate(
                (tmp_to_save, loc_v[0], loc_v[1]),
                axis = 1
            )
            self.integrated[j, :] = tmp_to_save.reshape(-1)
            #
            ### ERK
            xyvv = self.integrated[j, :].reshape((self.num_cell, 4))
            self.loc_erk_store[j, :] = loc_value_in(
                np.array([self.sep_if(xyvv[k0, 0], xyvv[k0, 1]) for k0 in range(self.num_cell)]),
                self.ERK[j, :, :]
            ).reshape(-1)
            #
            if c >= 20:
                print("    %d th loop and the time is %s" % (j, timej))
                c = 0
            c += 1
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
