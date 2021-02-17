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

import math

from matplotlib import animation as ani

import pandas as pd


################
### draw snapshots
################
def draw_2D_fluid(
    dir,
    arr,
    coord,
    times,
    points = 0,
    man_points = [],
    movfile = None,
    valueind = [0, 1, 2],
    ERK = None,
    lim2d = None,
    rect_time = [],
    rect_tups = []
):
    ind_v_in = [j for j in valueind]
    ### values
    vals       = ["Vx", "Vy", "Rho", "ERK"]
    scale_fill = [
        "scale_fill_gradient2(high = 'red', low = 'blue', mid = 'white', midpoint = 0",
        "scale_fill_gradient2(high = 'red', low = 'blue', mid = 'white', midpoint = 0",
        "scale_fill_viridis_c(",
        "scale_fill_viridis_c("
    ]
    collims    = [
        ", limits = c(%f, %f)" % (np.amin(arr[:, j, :, :]), np.amax(arr[:, j, :, :]))
        for j in ind_v_in
    ] + [
        ", limits = c(%f, %f)" % (np.amin(ERK), np.amax(ERK))
    ]
    if not ERK is None:
        ind_v_in.append(3)
    scale_fill = [scale_fill[j] + collims[j] for j in ind_v_in]
    #
    r = pyper.R()
    r("library(tidyverse); library(gganimate)")
    #
    ### coord
    cox = coord[0, :, 0]
    coy = coord[1, 0, :]
    coy_str = ["%09.4f" % j for j in coy]
    #
    if lim2d is None:
        dmin = cox.min()
        dmax = cox.max()
        tuple_R = (dmin, dmax, dmin, dmax)
    elif len(lim2d) == 2:
        dmin = lim2d[0]
        dmax = lim2d[1]
        tuple_R = (dmin, dmax, dmin, dmax)
    elif len(lim2d) == 4:
        tuple_R = tuple(lim2d)
    #
    ### get points
    if 0 < points or 0 < len(man_points) or 0 < len(rect_time):
        snaps = [0] ### index
        #
        if points >= 2:
            snaps.append(times.shape[0] - 1)
        if points >= 3:
            step  = int(times.shape[0] / points) + 1
            snaps = list(
                range(0, times.shape[0], step)
            ) + [
                times.shape[0] - 1
            ]
            #
        snaps += man_points + rect_time
        snaps = np.unique(snaps)
        #
        ### draw snapshots
        for j in snaps:
            for k in ind_v_in:
                ### ERK below, not here
                #print("    loop %d %d" % (j, k))
                #
                filek = "%s/snap_%s_%08d.png" % (dir, vals[k], j)
                file_rect = "%s/rect_%s_%08d.png" % (dir, vals[k], j)
                #
                if k < 3:
                    arrk = arr[j, k, :, :]
                elif k == 3 and not ERK is None:
                    arrk = ERK[j, :, :]
                #
                df = pd.DataFrame(arrk, columns = coy_str)
                df["x"] = cox
                #
                r.assign("df", df)
                r.assign("t", times[j, 0])
                comg = [
                    "df   <- gather(df, key = y, value = %s, -x)" % vals[k],
                    "df$y <- sapply(df$y, function(y){return(as.numeric(str_sub(y, 2)))})",
                    "df$time <- t",
                    "g000 <- ggplot() + geom_tile(",
                    "  data = df, aes(x, y, fill = %s)" % vals[k],
                    ") + theme_classic(",
                    ") + %s" % scale_fill[k],
                    ") + coord_fixed(",
                    "  ratio = 1, xlim = c(%f, %f), ylim = c(%f, %f)" % tuple_R,
                    ")",
                    "ggsave(file = '%s', plot = g000, height = 120, width = 160, unit = 'mm')" % filek,
                    "ggsave(file = '%s', plot = g000, height = 120, width = 160, unit = 'mm')" % ".".join([filek.rsplit(".", 1)[0], "pdf"]),
                    "savelist <- c('g000', 'df')"
                ]
                if j in rect_time:
                    comg += [
                    "g010 <- g000 + geom_polygon(",
                    "  aes(",
                    "    x = c(%f, %f, %f, %f, %f)," % rect_tups[0],
                    "    y = c(%f, %f, %f, %f, %f)"  % rect_tups[1],
                    "  ),",
                    "  fill = NA, color = '#ffffff'",
                    ")",
                    "ggsave(file = '%s', plot = g010, height = 120, width = 160, unit = 'mm')" % file_rect,
                    "ggsave(file = '%s', plot = g010, height = 120, width = 160, unit = 'mm')" % ".".join([file_rect.rsplit(".", 1)[0], "pdf"]),
                    "savelist <- append(savelist, c('g010'))"
                    ]
                comg += [
                    "save(list = savelist, file = '%s')" % ".".join([filek.rsplit(".", 1)[0], "Rdata"])
                ]
                r("\n".join(comg))
                if os.path.exists(filek):
                    print("saved %s" % filek)
                else:
                    print("failed %s" % filek)
                #
                ### reset R
                del r
                r = pyper.R()
                r("library(tidyverse); library(gganimate)")
                #
            #
        #
    #
    if not movfile is None:
        ### threshold to pass all the data on R
        #thr = 200000000 ### 2 * 10**8, R can take 2^31 data at most
        thr = 2 * 10**8
        ### count the number of data
        datacount = arr[:, 0, :, :].size
        #
        step = (int(datacount / thr) + 1) * 2
        fps  = int(20 * 2 / step)
        nframes = int(times.shape[0] / step)
        #
        print("    data size %d ~ 10^(%05.2f) ~ 2^(%05.2f)\n    step %d" % (datacount, math.log10(datacount), math.log2(datacount), step)); #sys.exit()
        #
        #anim_time = times.shape[0] * 0.1; print(anim_time); #sys.exit()
        #
        movhead = movfile.split(".")[0]
        #
        for k in ind_v_in:
            filek = "%s_%s.mp4" % (movhead, vals[k])
            csv_k = "%s_%s.csv" % (movhead, vals[k])
            #
            ### reset R
            r = pyper.R()
            r("library(tidyverse); library(gganimate)")
            #
            ### data frame preparation on pandas
            df = pd.DataFrame()
            print("preparing data frame to pass to R")
            for j in range(0, times.shape[0], step):
                ### data frame preparation
                if k < 3:
                    arrk = arr[j, k, :, :]
                elif k == 3 and not ERK is None:
                    arrk = ERK[j, :, :]
                #
                df_tmp         = pd.DataFrame(arrk, columns = coy_str)
                df_tmp["x"]    = cox
                df_tmp["time"] = times[j, 0]
                #
                df = pd.concat([df, df_tmp], axis = 0).reset_index(drop = True)
                #
            r.assign("df", df)
            print("--> prepared, start rendering %s" % filek)
            #
            comg = [
                "df   <- gather(df, key = y, value = %s, -x, -time)" % vals[k],
                "df$y <- sapply(df$y, function(y){return(as.numeric(str_sub(y, 2)))})",
                "df$time_f <- sapply(df$time, function(x){return(sprintf('%09.3f', x))})",
                ### for extension
                "write.csv(df, '%s', quote = F, row.names = F)" % csv_k,
                ### animation
                "a000 <- ggplot(df, aes(x, y)) + geom_tile(",
                "  aes(fill = %s)" % vals[k],
                ") + theme_classic(",
                ") + %s" % scale_fill[k],
                ") + coord_fixed(",
                "  ratio = 1, xlim = c(%f, %f), ylim = c(%f, %f)" % tuple_R,
                ") + transition_manual(",
                "  time_f",
                ") + labs(",
                "  title = 'time: {current_frame}'",
                ")",
                "anim_save(",
                "  filename = '%s'," % filek,
                "  animation = a000,",
                "  renderer  = ffmpeg_renderer(),",
                "  nframes   = %d," % nframes,
                "  fps       = %d," % fps,
                #"  duration  = %f," % anim_time,
                "  width = 120, height = 120, unit = 'mm', res = 360",
                ")"
            ]
            #print(r("\n".join(comg))); sys.exit()
            r("\n".join(comg))
            #
            if os.path.exists(filek):
                print("saved %s" % filek)
            else:
                print("failed %s" % filek)
            #
        #
    #
    #
    #

def build_tuples(new_O, theta, xlim_draw, th_newY = 1.2):
    out = [] ### of 2 tuples
    #
    ### left right, top bottom
    lb = [xlim_draw[0], -th_newY]
    rb = [xlim_draw[1], -th_newY]
    rt = [xlim_draw[1],  th_newY]
    lt = [xlim_draw[0],  th_newY]
    #
    ### square
    sq = np.array([lb, rb, rt, lt, lb]).T
    #
    ### rotate
    sq = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ]).dot(sq)
    sq[0, :] += new_O[0]
    sq[1, :] += new_O[1]
    #
    out.append(tuple(sq[0, :].tolist()))
    out.append(tuple(sq[1, :].tolist()))
    #
    return(out)
    #


################
### drawing of one movie
################
### reconstructed density, grid
def draw_one_2D(
    movfile,
    den_arr,
    coord,
    times,
    add_label = None,
    lim2d     = None
):
    ### drawing only one field in a movie file
    #thr = 200000000 ### 2 * 10**8, R can take 2^31 data at most
    thr = 2 * 10**8
    ### count the number of data
    datacount = den_arr.size
    #
    max_num   = np.amax(den_arr)
    if add_label is None:
        add_title_r = ""
    else:
        add_title_r = add_label
    #
    step = (int(datacount / thr) + 1) * 2
    fps  = int(20 * 2 / step)
    nframes = int(times.shape[0] / step)
    #
    print("    data size %d ~ 10^(%05.2f) ~ 2^(%05.2f)\n    step %d" % (datacount, math.log10(datacount), math.log2(datacount), step))
    #
    anim_time = times.shape[0] * 0.1
    #
    cox = coord[0, :, 0]
    coy = coord[1, 0, :]
    coy_str = ["%09.4f" % j for j in coy]
    #
    if lim2d is None:
        dmin = cox.min()
        dmax = cox.max()
    else:
        dmin = lim2d[0]
        dmax = lim2d[1]
    #
    #### reset R
    r = pyper.R()
    r("library(tidyverse); library(gganimate)")
    #
    ### data frame preparation on pandas
    df = pd.DataFrame()
    print("preparing data frame to pass to R")
    for j in range(0, times.shape[0], step):
        ### data frame preparation
        df_tmp         = pd.DataFrame(den_arr[j, :, :], columns = coy_str)
        df_tmp["x"]    = cox
        df_tmp["time"] = times[j, 0]
        #
        df = pd.concat([df, df_tmp], axis = 0).reset_index(drop = True)
        #
    r.assign("df", df)
    print("--> prepared, start rendering %s" % movfile)
    #
    comg = [
        "max_num <- %f" % max_num,
        "max_title <- sprintf('max %09.5f', max_num)",
        "df   <- gather(df, key = y, value = density, -x, -time)",
        "df$y <- sapply(df$y, function(y){return(as.numeric(str_sub(y, 2)))})",
        "df$time_f <- sapply(df$time, function(x){return(sprintf('%09.3f', x))})",
        "a000 <- ggplot(df, aes(x, y)) + geom_tile(",
        "  aes(fill = density)",
        ") + theme_classic(",
        ") + scale_fill_viridis_c(",
        ") + coord_fixed(",
        "  ratio = 1, xlim = c(%f, %f), ylim = c(%f, %f)" % (dmin, dmax, dmin, dmax),
        ") + transition_manual(",
        "  time_f",
        ") + labs(",
        "  title = paste('%s', max_title, 'time: {current_frame}', sep = '\n')" % add_title_r,
        ")",
        "anim_save(",
        "  filename = '%s'," % movfile,
        "  animation = a000,",
        "  renderer  = ffmpeg_renderer(),",
        "  nframes   = %d," % nframes,
        "  fps       = %d," % fps,
        #"  duration  = %f," % anim_time,
        "  width = 160, height = 120, unit = 'mm', res = 360",
        ")"
    ]
    #print(r("\n".join(comg)))
    r("\n".join(comg))
    #
    if os.path.exists(movfile):
        print("saved %s" % movfile)
    else:
        print("failed %s" % movfile)
    #
    #

### reconstructed density, cell
def draw_den_p2D(
    movfile,
    den_arr,
    coord,
    times,
    add_label = None,
    center_0  = False,
    lim2d     = None
):
    #print(np.mean(den_arr[:, :, 1]))
    ### drawing only one field in a movie file
    #thr = 200000000 ### 2 * 10**8, R can take 2^31 data at most
    thr = 2 * 10**8 ### 2 * 10**7
    ### count the number of data
    datacount = den_arr.size
    #
    max_num   = np.amax(den_arr[:, :, 2])
    if add_label is None:
        add_title_r = ""
    else:
        add_title_r = add_label
    #
    step = (int(datacount / thr) + 1) * 2
    fps  = int(20 * 2 / step)
    nframes = int(times.shape[0] / step)
    #
    print("    data size %d ~ 10^(%05.2f) ~ 2^(%05.2f)\n    step %d" % (datacount, math.log10(datacount), math.log2(datacount), step))
    #
    anim_time = times.shape[0] * 0.1
    #
    cox = coord[0, :, 0]
    coy = coord[1, 0, :]
    #
    if lim2d is None:
        dmin = cox.min()
        dmax = cox.max()
    else:
        dmin = lim2d[0]
        dmax = lim2d[1]
    #
    #### reset R
    r = pyper.R()
    r("library(tidyverse); library(gganimate)")
    #
    ### data frame preparation on pandas
    df = pd.DataFrame()
    print("preparing data frame to pass to R")
    for j in range(0, times.shape[0], step):
        ### data frame preparation
        df_tmp         = pd.DataFrame(den_arr[j, :, :], columns = ["x", "y", "density"])
        df_tmp["time"] = times[j, 0]
        #
        df = pd.concat([df, df_tmp], axis = 0).reset_index(drop = True)
        #
    r.assign("df", df)
    print("--> prepared, start rendering %s" % movfile)
    #print(np.mean(df["y"].to_numpy()))
    #print(np.mean(den_arr[:, :, 1]))
    #
    if center_0:
        scale = "\n".join([
            ") + scale_color_gradient2(",
            "  low = 'blue', high = 'red', mid = 'white', midpoint = 0"
        ])
    else:
        scale = "\n".join([
            ") + scale_color_viridis_c("
        ])
    #
    comg = [
        "max_num <- %f" % max_num,
        "max_title <- sprintf('max %09.5f', max_num)",
        "df$time_f <- sapply(df$time, function(x){return(sprintf('%09.3f', x))})",
        "a000 <- ggplot(df, aes(x, y)) + geom_point(",
        "  aes(color = density)",
        ") + theme_classic(",
        #") + scale_color_viridis_c(",
        scale,
        ") + coord_fixed(",
        "  ratio = 1, xlim = c(%f, %f), ylim = c(%f, %f)" % (dmin, dmax, dmin, dmax),
        #"  xlim = c(%f, %f)," % (np.amin(cox), np.amax(cox)),
        #"  ylim = c(%f, %f)"  % (np.amin(coy), np.amax(coy)),
        ") + transition_manual(",
        "  time_f",
        ") + labs(",
        "  title = paste('%s', max_title, 'time: {current_frame}', sep = '\n')" % add_title_r,
        ")",
        "anim_save(",
        "  filename = '%s'," % movfile,
        "  animation = a000,",
        "  renderer  = ffmpeg_renderer(),",
        "  nframes   = %d," % nframes,
        "  fps       = %d," % fps,
        #"  duration  = %f," % anim_time,
        "  width = 160, height = 120, unit = 'mm', res = 360",
        ")"
    ]
    #print(r("\n".join(comg)))
    r("\n".join(comg))
    #
    if os.path.exists(movfile):
        print("saved %s" % movfile)
    else:
        print("failed %s" % movfile)
    #
    #

################
###
################



################
###
################



###
