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
import subprocess as sbp

import os
import shutil

import pyper

import math

import pandas as pd

from .cfun_weno2 import erk_2D_inc
from .fun_dir    import dir_reset

################
### function for initial positions
################
def init_hex_close(xmax, ynum, velo_add = True):
    a = np.arange(0.0, xmax, dtype = np.float64)
    a = a.reshape((a.shape[0], 1))
    o = np.ones_like(a)
    #
    out = np.concatenate(
        [
            np.concatenate(
                [
                    (a + 0.5 ** (j % 2)),
                    (o * float(j) * 0.5 * (3.0 ** 0.5))
                ],
                axis = 1
            ) for j in range(ynum)
        ],
        axis = 0
    )
    if velo_add:
        out = np.concatenate([out, np.zeros_like(out)], axis = 1)
        #
    #
    return(out)
    #

def init_hex_circle(center, radius, velo_add = True):
    '''
    center is list or 1D np array
    '''
    xmax = center[0] + radius * 1.2
    ynum = int((center[1] + radius * 1.2) / (0.5 * 3.0**0.5) + 0.5) + 1
    #
    tmparr = init_hex_close(xmax, ynum, velo_add); #print(tmparr.shape[0])
    #
    out = []
    for j in range(tmparr.shape[0]):
        lenj = np.linalg.norm(tmparr[j, 0:2] - np.array(center))
        if lenj <= radius:
            out.append(tmparr[j, :])
            #
        #
    #
    #print(len(out))
    #
    return(np.array(out))
    #
    #


################
### class to draw
################
def bind_ec(arr, t, ind, lab_arr = ["x", "y", "ERK"]):
    '''
    bind time and cell index
    '''
    df  = pd.DataFrame(arr, columns = lab_arr)
    tmp = pd.DataFrame({"time": t, "cell": "cell%04d" % ind})
    #
    df  = pd.concat([df, tmp], axis = 1)
    return(df)
    #

def angle22(theta):
    return(
        np.array([
            [math.cos(theta), -math.sin(theta)],
            [math.sin(theta),  math.cos(theta)]
        ])
    )

def tf1(j, k):
    if k <= j:
        return(1)
    else:
        return(0)

def ld1(intlen):
    return(
        np.array([[tf1(j, k) for k in range(intlen)] for j in range(intlen)])
    )

def choose_a_cell(S, C, theta, cells, calib = None):
    '''
    inputs
        S       start point of ERK wave   array of shape (2, 1)
        C       center of drawing region  array of shape (2, 1)
        theta   angle of ERK wave [rad]   double
        cells   initial cells locations   array of shape (num_cell, 2) -> x,y
    output
        index   of a cell choosen         int
    '''
    ### target location
    if calib is None:
        calibration = np.zeros((2, 1))
    else:
        calibration = calib
    eq_right  = C - S
    if np.linalg.norm(eq_right) <= 1.0:
        targ = C
    else:
        mat_angle = np.array([
            [np.cos(theta),  np.sin(theta)],
            [np.sin(theta), -np.cos(theta)]
        ])
        tmp  = np.linalg.solve(mat_angle, eq_right)
        targ = S + tmp[0, 0] * np.array([[np.cos(theta)], [np.sin(theta)]])
    targ += calibration
    #
    ### choose a cell
    rel = cells - np.array([targ.reshape(-1)] * cells.shape[0])
    sq  = [rel[j, 0] ** 2 + rel[j, 1] ** 2 for j in range(rel.shape[0])]
    out = np.argmin(sq)
    return(out)
    #

class Draw_Part2:
    def __init__(self, integrated, timerange, ERK, interval = 100, lagrange = False):
        self.integrated = np.array(integrated)
        self.timerange  = timerange
        self.ERK        = ERK
        #
        self.interval   = interval
        self.r          = None
        #
        self.lagrange   = lagrange
        #
        self.dt = self.timerange[1, 0] - self.timerange[0, 0]
        #
    def draw(
        self,
        filename,
        Rdata       = None,
        xylims      = None,
        snap_dir    = None,
        points      = 0,
        erk_start   = None,
        erk_theta   = None,
        calib       = None,
        track_color = None,
        man_snap_time = [],
        make_traj_mov = False,
        bash_ff     = "tmp_ff_bash.txt"
    ):
        ### define animation file time [sec]
        frame_rate = self.interval * 0.001 ### seconds
        anim_time  = self.timerange.shape[0] * frame_rate #* 20
        #
        dfname     = "%s.csv" % filename.split(".")[0]
        arrow      = "%s_arrow.png" % filename.split(".")[0]
        #
        traj_name  = "%s_trajectory.mp4" % filename.split(".")[0]
        #
        if track_color is None:
            track_color = 'time'
        #
        ### file name select
        if self.lagrange:
            p_or_l = "lagrange"
            p_or_c = "continuum"
        else:
            p_or_l = "particle"
            p_or_c = "particle"
            #
        ##########
        # preparation for data frames
        ##########
        erk      = self.ERK.reshape(self.ERK.shape + (1,))
        integ    = np.concatenate([self.integrated[:, :, 0:2], erk], axis = 2)
        num_cell = integ.shape[1]
        if xylims is None:
            xymin = np.min(self.integrated[:, :, 0:2])
            xymax = np.max(self.integrated[:, :, 0:2])
            tuple_R = (xymin, xymax, xymin, xymax)
        elif len(xylims) == 2:
            xymin = xylims[0]
            xymax = xylims[1]
            tuple_R = (xymin, xymax, xymin, xymax)
        elif len(xylims) == 4:
            xymin = xylims[0]
            xymax = xylims[1]
            tuple_R = tuple(xylims)
            #
        #
        if (not erk_start is None) and (not erk_theta is None):
            track  = True
            if calib is None:
                calibration = np.zeros((2, 1))
            else:
                calibration = calib
            #center = np.ones((2, 1)) * (xymax + xymin) * 0.5
            center = (np.array([[tuple_R[0]], [tuple_R[3]]]) + np.array([[tuple_R[1]], [tuple_R[2]]])) * 0.5
            choosen_cell_ind = choose_a_cell(
                erk_start,
                center,
                erk_theta,
                integ[0, :, 0:2],
                calibration
            )
            df_choosen = pd.DataFrame(
                integ[:, choosen_cell_ind, 0:2],
                columns = ["x", "y"]
            )
            if track_color == "time":
                df_choosen["time"] = self.timerange[:, 0]
                time_min = self.timerange.min()
                time_max = self.timerange.max()
                if len(man_snap_time) > 0:
                    mindex = min(man_snap_time)
                    maxdex = max(man_snap_time)
                    time_min = self.timerange[mindex:maxdex, :].min()
                    time_max = self.timerange[mindex:maxdex, :].max()
                ### else do not add this column to save volume on memory
        else:
            track  = False
        #
        ### snap shots
        if points >= 1 and not snap_dir is None:
            snaps = [0]
            if points >= 2:
                snaps.append(self.timerange.shape[0] - 1)
            if points >= 3:
                step  = int(self.timerange.shape[0] / points) + 1
                snaps = list(
                    range(0, self.timerange.shape[0], step)
                ) + [
                    self.timerange.shape[0] - 1
                ]
            snaps = man_snap_time + snaps
            #
            ### ERK max min
            emax = np.amax(erk)
            emin = np.amin(erk)
            #
            ### draw snapshots
            r = pyper.R(use_numpy = True, use_pandas = True)
            comg = ["library(tidyverse)", "library(gganimate)"]
            r("\n".join(comg))
            #
            def filej_name(snap_dir, p_or_l, j):
                return("%s/snap_%s_%08d.png" % (snap_dir, p_or_l, j))
            if track:
                r.assign("df_choosen", df_choosen)
                max_rows = 2 * 10**8
                skips = (int(num_cell * integ.shape[0] / max_rows) + 1) * 2
                tmp_r = range(0, self.timerange.shape[0], skips)
                tmp_d = "%s/tmp" % snap_dir
                dir_reset(tmp_d, False)
                df_manage = pd.DataFrame({
                    "ind": list(tmp_r),
                    "ori": [filej_name(snap_dir, p_or_l, j) for j in tmp_r]
                })
                df_manage["tmp"]  = ["%s/%s" % (tmp_d, filej_name(snap_dir, p_or_l, j).rsplit("/", 1)[1]) for j in range(df_manage.shape[0])]
                df_manage["keep"] = [(j in snaps) for j in tmp_r]
                #
                tmp1 = [os.path.exists(j) for j in df_manage["ori"]]
                tmp2 = [os.path.exists(j) for j in df_manage["tmp"]]
                #
                if make_traj_mov:
                    snaps = np.unique(snaps + list(tmp_r))
                else:
                    snaps = np.unique(snaps)
                #
                #
            else:
                tmp2 = [True]
            #
            for j in snaps:
                filej = filej_name(snap_dir, p_or_l, j)
                #r = pyper.R(use_numpy = True, use_pandas = True)
                #comg = ["library(tidyverse)", "library(gganimate)"]
                #r("\n".join(comg))
                df = pd.concat([bind_ec(integ[j:(j+1), k, :], self.timerange[j:(j+1), 0], k) for k in range(num_cell)], axis = 0).reset_index(drop = True)
                r.assign("df", df)
                comg = [
                    "g000 <- ggplot() + geom_point(",
                    "  data = df, aes(x, y, color = ERK)",
                    ") + coord_fixed(",
                    "  ratio = 1, xlim = c(%f, %f), ylim = c(%f, %f)" % tuple_R,
                    ") + theme_classic(",
                    ") + scale_color_continuous(",
                    "  limits = c(%f, %f)" % (emin, emax),
                    ")"
                ]
                if track:
                    if track_color == "time":
                        comg += [
                            "library(ggnewscale)",
                            "g000 <- g000 + new_scale_color(",
                            ") + geom_point(",
                            "  data = df_choosen[1:%d, ], aes(x, y, color = time)" % (j+1),
                            ") + scale_color_gradientn(",
                            "  colours = heat.colors(100), limits = c(%f, %f)" % (time_min, time_max),
                            ")"
                        ]
                    else:
                        comg += [
                            "g000 <- g000 + geom_path(",
                            "  data = df_choosen[1:%d, ], aes(x, y), color = '%s', size = 2" % ((j+1), track_color),
                            ")"
                        ]
                #
                comg += [
                    "ggsave(plot = g000, file = '%s', height = 90, width = 120, unit = 'mm')" % filej,
                    "ggsave(plot = g000, file = '%s', height = 90, width = 120, unit = 'mm')" % ".".join([filej.rsplit(".", 1)[0], "pdf"]),
                    "save(list = c('df', 'g000'), file = '%s')" % ".".join([filej.rsplit(".", 1)[0], "Rdata"]),
                    "rm(df)",
                    "rm(g000)" ### reset
                ]
                r("\n".join(comg))
                #del r ### reset
                if os.path.exists(filej):
                    print("saved %s" % filej)
                else:
                    print("failed %s" % filej)
                    #
                #
            ### run ffmpeg and manage files
            ### df_manage has...
            ###     int  ind
            ###     str  ori
            ###     str  tmp
            ###     bool keep
            if not all(tmp2):
                for j in range(df_manage.shape[0]):
                    src = df_manage["ori"][j]
                    dst = df_manage["tmp"][j]
                    if os.path.exists(src):
                        if df_manage["keep"][j]:
                            shutil.copy(src, dst)
                        else:
                            shutil.move(src, dst)
                    #
            run_ff = " ".join([
                "ffmpeg",
                "-r",
                "20",
                "-i",
                "".join(["%s/snap_%s" % (tmp_d, p_or_l), "_%08d.png"]),
                "-pix_fmt",
                "yuv420p",
                "-r",
                "20",
                traj_name
            ])
            run_ff = " ".join([
                run_ff,
                "&&",
                "rm -rf %s" % tmp_d
            ])
            with open(bash_ff, mode = 'a') as f:
                if make_traj_mov:
                    f.write("\n" + run_ff)
                else:
                    f.write("\n### not to make %s" % traj_name)
            #
            del r ### reset
            #
        #
        ### movie file
        if self.r is None:
            r = pyper.R(use_numpy = True, use_pandas = True)
            comg = [
                "library(tidyverse)",
                "library(gganimate)"
            ]
            r("\n".join(comg))
            #
            ### limit the data volume
            #max_rows = 2000000
            max_rows = 2 * 10**8
            #
            skips = (int(num_cell * integ.shape[0] / max_rows) + 1) * 2
            fps   = int(20 * 2 / skips)
            nframes = int(self.timerange.shape[0] / skips)
            #
            df = pd.concat([bind_ec(integ[::skips, j, :], self.timerange[::skips, 0], j) for j in range(num_cell)], axis = 0).reset_index(drop = True)
            r.assign("df", df)
            comg = [
                ### for animation
                "f <- function(x){return(formatC(x, format = 'f', flag = '0', width = 8, digit = 2))}",
                "df$time_f <- sapply(df$time, f)"
            ]
            r("\n".join(comg))
            #
            self.r = r
            #
        else:
            r = self.r
            #
        ##########
        # draw on R
        ##########
        if ".gif" in filename:
            renderer = "gifski_renderer()"
        elif ".mp4" in filename:
            renderer = "ffmpeg_renderer()"
        else:
            print("filename is not specified collectly;\n    %s" % filename)
            return()
            #
        comg = [
            ### save the df
            "write.csv(df, '%s', row.names = F, quote = F)" % dfname,
            ### basic graph
            "g000 <- ggplot(df, aes(x, y)) + geom_point(",
            "  aes(color = ERK)",
            ")",
            ### themes
            "g010 <- g000 + theme_bw(",
            ") + coord_fixed(",
            "  ratio = 1, xlim = c(%f, %f), ylim = c(%f, %f)" % tuple_R,
            ")",
            ### animation make
            "a010 <- g010 + transition_manual(",
            "  time_f, cumulative = F",
            ") + labs(",
            "  title = '%s time: {current_frame}'" % p_or_c,
            ")"
        ]
        if not Rdata is None:
            if self.lagrange:
                comg = comg + [
                    "df_l2 <- df",
                    "l200  <- g000",
                    "l210  <- g010",
                    "save(list = c('df_l2', 'l200', 'l210'), file = '%s')" % Rdata
                ]
            else:
                comg = comg + [
                    "df_p2 <- df",
                    "p200  <- g000",
                    "p210  <- g010",
                    "save(list = c('df_p2', 'p200', 'p210'), file = '%s')" % Rdata
                ]
        comg = comg + [
            "anim_save(",
            "  filename  = '%s'," % filename,
            "  animation = a010,",
            "  renderer  = %s," % renderer,
            "  nframes   = %d," % nframes,
            "  fps       = %d," % fps,
            "  width     = 120, height = 120, unit = 'mm', res = 270",
            ")"
        ]
        '''
        ### arrow file, start and end
        comg = comg + [
            "df_start <- subset(df, time == min(df$time))",
            "df_end   <- subset(df, time == max(df$time))",
            "df_arrow <- rbind(df_start, df_end)",
            "arr0 <- ggplot(df_arrow, aes(x, y)) + geom_path(",
            "  aes(group = cell),",
            "  arrow = arrow(angle = 10, length = unit(1, 'mm'), type = 'closed')",
            ") + theme_classic(",
            ") + coord_fixed(",
            "  ratio = 1, xlim = c(%f, %f), ylim = c(%f, %f)" % tuple_R,
            ")",
            "ggsave(plot = arr0, file = '%s', width = 160, height = 120, unit = 'mm')" % arrow
        ]
        '''
        print("start rendering...")
        r("\n".join(comg))
        if os.path.exists(filename):
            print("saved %s" % filename)
        else:
            print("failed %s" % filename)
        #
    def x_t(self, filename, list_for_erk, new_O = [0.0, 0.0], theta = None, xlim_draw = None, th_newY = 1.2):
        ### ERK wave direction
        if theta is None:
            the_in = list_for_erk[5]
        else:
            the_in = theta
        #
        ### df
        dfname     = "%s.csv"     % filename.split(".")[0]
        df_erk     = "%s_erk.csv" % filename.split(".")[0]
        #
        ### calculate xy positions to show
        xy  = self.integrated[:, :, 0:2] ### overwrite it later
        xy[:, :, 0] -= new_O[0]
        xy[:, :, 1] -= new_O[1]
        xy  = np.array([xy[t, :, :].dot(angle22(-the_in).T) for t in range(xy.shape[0])])
        xy_new = xy ### for later
        #print(xy_new)
        meanx = np.mean(xy[:, :, 0])
        xy = np.where(np.abs(xy[:, :, 1]) <= th_newY, xy[:, :, 0], meanx); #print(xy)
        #xy  = xy[ind, :, :]
        if xlim_draw is None:
            draw_lim_in = [
                np.amin(xy[:, :]),
                np.amax(xy[:, :])
            ]
        else:
            draw_lim_in = xlim_draw
            #
        ### calculate coordinate ERK values based on new coord
        step = draw_lim_in[1] - draw_lim_in[0]
        xy = np.arange(draw_lim_in[0] - step * 0.1, draw_lim_in[1] + step * 0.1, 0.005 * step)
        xy = xy.reshape(xy.shape + (1,))
        xy = np.concatenate((xy, np.zeros_like(xy, dtype = np.float64)), axis = 1)
        #
        ### for erk
        xs = xy[:, 0]
        #
        ### rotate and slide
        xy = xy.dot(angle22(the_in).T)
        xy[:, 0] += new_O[0]
        xy[:, 1] += new_O[1] ### now xy has its line coord on original position
        #
        ### get erk values over time, (time) x (coord along new x axis)
        draw_frame = 400
        step = int(self.timerange.shape[0] / draw_frame)
        if step == 0:
            step = 1
        df = pd.concat(
            [
                pd.DataFrame({"x": xs, "ERK": erk_2D_inc(*[t, xy] + list_for_erk), "time": t})
                for t in self.timerange[::step, 0]
            ],
            axis = 0
        ).reset_index(drop = True)
        ### on R
        if self.lagrange:
            color_select = "black" #"blue"
        else:
            color_select = "black"
            #
        #
        r = pyper.R(use_numpy = True, use_pandas = True)
        comg = [
            "library(tidyverse)",
            "library(gganimate)"
        ]
        r("\n".join(comg))
        #
        r.assign("back_erk", df); #print(df.shape)
        #
        ### particles
        df = pd.DataFrame()
        shape1 = xy_new.shape[1]
        #
        dt_t = step * self.dt
        for j in range(shape1):
            ndf = pd.DataFrame({
                "x":    xy_new[0:self.timerange.shape[0]:step, j, 0],
                "y":    xy_new[0:self.timerange.shape[0]:step, j, 1],
                "time": self.timerange[0:self.timerange.shape[0]:step, 0],
            })
            ndf = ndf.query('-@th_newY < y and y < @th_newY').reset_index(drop = True)
            ### check the continuity
            tj = ndf["time"].to_numpy()
            tend = tj.shape[0]
            tj = np.where(tj[1:tend] - tj[0:(tend-1)] == dt_t, 0, 1).reshape(((tend-1), 1))
            tj = ld1(tend-1).dot(tj)
            cj = [
                "cell%08d_set%03d" % (j, 0)
            ] + [
                "cell%08d_set%03d" % (j, tj[k, 0])
                for k in range(tend-1)
            ]
            ndf["cell"] = cj
            df = pd.concat([df, ndf], axis = 0).reset_index(drop = True)
            #
        time_only = df["time"]
        ylim = [time_only.min(), time_only.max()]
        #
        r.assign("df", df); #print(df.shape)
        print("prepared 2D rotated x-t graph")
        #
        #
        ### draw on R
        comg = [
            "write.csv(df,       '%s', quote = F, row.names = F)" % dfname,
            "write.csv(back_erk, '%s', quote = F, row.names = F)" % df_erk,
            "g000 <- ggplot() + geom_tile(",
            "  data = back_erk, aes(x, time, fill = ERK), alpha = 0.4",
            ") + geom_path(",
            "  data = df, aes(x, time, group = cell), color = '%s', alpha = 0.4" % color_select,
            ") + theme_classic(",
            ") + scale_fill_viridis_c(",
            ") + coord_cartesian(",
            "  expand = F,",
            "  xlim = c(%f, %f), ylim = c(%f, %f)" % tuple(draw_lim_in + ylim),
            ") + scale_y_reverse(",
            ")",
            "ggsave(plot = g000, file = '%s', width = 160, height = 120, unit = 'mm')" % filename,
            "ggsave(plot = g000, file = '%s', width = 160, height = 120, unit = 'mm')" % ".".join([filename.rsplit(".", 1)[0], "pdf"]),
            "xt <- df",
            "save(list = c('xt', 'back_erk', 'g000'), file = '%s')" % ".".join([filename.rsplit(".", 1)[0], "Rdata"])
        ]
        #print(r("\n".join(comg)))
        r("\n".join(comg))
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
