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

from scipy.integrate import RK45

import math
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib import animation as ani

import pandas as pd

### handwrite
from .fun_dir import di1
from .fun_dir import di2

from .fun_dir import Time_keeper


################
### animation of results
################
class Animation_results:
    def __init__(self, integrated, coord, timerange, loopend = 20, interval = 10, repeat = True):
        self.integrated = integrated
        self.coord      = coord
        self.end        = min([loopend, self.integrated.shape[0]])
        self.timerange  = timerange
        self.max_time   = self.timerange[(self.end - 1), 0]
        self.interval   = interval
        self.repeat     = repeat
        #
        integ2          = self.integrated.reshape((self.integrated.shape[0], 3, self.coord.shape[1], self.coord.shape[2]))
        tmp_mat         = [integ2[:, j, :, :] for j in [0, 2]]
        self.values     = tmp_mat
        #
        self.tmkpr      = Time_keeper()
        #
        #
    def drawR(self, filename = None, res_choice = "low", format = [0, 1], cap_name = None):
        testing  = False
        #
        print("\ndrawing animations... (%s)" % self.tmkpr.get_elapsed())
        colnames = ["Vx", "Rho"]
        #
        #
        if res_choice == "low":
            resolution = 270
        elif res_choice == "mid":
            resolution = 360
        elif res_choice == "high":
            resolution = 480
        else:
            resolution = 270
        #
        ### define animation file time [sec]
        frame_rate = self.interval * 0.001 ### seconds
        anim_time  = self.timerange.shape[0] * frame_rate; #print(anim_time); sys.exit()
        #
        #
        ################
        ### on R
        ################
        r = pyper.R(use_numpy = True, use_pandas = True)
        comg = [
            "library(tidyverse)",
            "library(gganimate)"
        ]
        r("\n".join(comg))
        #
        r("df <- data.frame()")
        #
        ### data preparation, select y = 0 like no dimension
        for j in range(0, len(self.values)):
            dfn = self.values[j][:, :, 0].T ### 0 vx or rho along x, 1 time
            dfp = pd.DataFrame(dfn)
            #
            dfp["x"] = self.coord[0, :, 0]
            #
            ### assign on R
            r.assign("df_tmp", dfp)
            #
            ### time as column names
            if False:
                for k in range(0, self.timerange.shape[0]):
                    r("colnames(df_tmp)[%d] <- '%f'" % ((k + 1), self.timerange[k, 0]))
            else:
                r("colnames(df_tmp) <- c(%s, 'x')" % ",".join(["%f" % l for l in self.timerange[:, 0]]))
            #
            ### gather for ggplot2
            comg = [
                "df_tmp <- gather(df_tmp, key = 'time', value = '%s', - x)" % colnames[j],
                "df_tmp$time <- as.numeric(df_tmp$time)"
            ]
            if j > 0:
                comg = ["print(str(df_tmp))"] + comg
                comg.append("df <- left_join(df, df_tmp, by = c('x' = 'x', 'time' = 'time'))")
            else:
                comg.append("df <- df_tmp")
            #
            if testing:
                print(r("\n".join(comg)))
            else:
                r("\n".join(comg))
            #
            #
        comg =[
            "df <- gather(df, key = variables, value = dynamics, %s, %s)" % tuple(colnames),
            "df$variables <- factor(df$variables, levels = unique(df$variables))",
            #"print(str(df)); print(str(na.omit(df)))",
            "f <- function(x){return(formatC(x, format = 'f', flag = '0', width = 8, digit = 2))}",
            "df$time_f <- sapply(df$time, f)"
        ]
        if testing:
            comg.append(
                "print(str(df))"
            )
            print(r("\n".join(comg)))
        else:
            r("\n".join(comg))
        #
        print("rendering... (%s)" % self.tmkpr.get_elapsed())
        comg = [
            "g000 <- ggplot(df, aes(x, dynamics)) + geom_line(",
            "  aes(group = variables, color = variables)",
            #") + theme_classic(",
            ") + theme_bw(",
            #") + scale_color_viridis_d(",
            ") + facet_grid(",
            "  variables ~ .",
            ") + coord_cartesian(",
            "  ylim = c(-2, 2)",
            ")",
            "g001 <- g000 + transition_manual(", ### this line and below are gganimate
            "  time",
            ") + labs(",
            "  title = 'time: {current_frame}'",
            ")",
            ########
            # test, for formatting
            ########
            "g010 <- g000 + transition_manual(",
            "  time_f",
            ") + labs(",
            "  title = 'time: {current_frame}'",
            ")"
        ]
        if not filename is None:
            ### loop to make both
            for extchoice in format:
                ext = [".gif", ".mp4"]
                filenamej = "".join([filename, ext[extchoice]])
                comg = comg + [
                    ########
                    # save
                    ########
                    "anim_save(",
                    "  filename  = '%s'," % filenamej,
                    "  animation = g010,",
                    "  renderer  = %s," % ["gifski_renderer()", "ffmpeg_renderer()"][extchoice],
                    "  duration  = %f," % anim_time,
                    "  width     = 160, height = 120, unit = 'mm', res = %s" % resolution,
                    ")"
                ]
                try:
                    if testing:
                        comg.append("save.image(file = '%s')" % "zzz_test_gganimate.Rdata")
                        print(r("\n".join(comg)))
                    else:
                        r("\n".join(comg))
                    #
                except:
                    pass
                #
                if os.path.exists(filenamej):
                    print("saved animation in %s\nit took %s" % (filenamej, self.tmkpr.get_elapsed()))
                else:
                    print("failed animation %s\nit took %s" % (filenamej, self.tmkpr.get_elapsed()))
                #
            #
        else:
            r("\n".join(comg))
            print("animation movie file was not made.")
        ### last moment
        try:
            if not cap_name is None:
                comg = [
                    "subdf <- subset(df, time == max(df$time))",
                    "s000 <- ggplot(subdf, aes(x, dynamics)) + geom_line(",
                    "  aes(group = variables, color = variables)",
                    ") + theme_bw(",
                    ") + facet_grid(",
                    "  variables ~ .",
                    ") + coord_cartesian(",
                    "  ylim = c(-2, 2)",
                    ") + labs(",
                    "  title = paste('time:', subdf$time_f[0])",
                    ")",
                    "ggsave(plot = s000, file = '%s', width = 160, height = 120, unit = 'mm')" % cap_name
                ]
                r("\n".join(comg))
                if os.path.exists(cap_name):
                    print("saved %s" % cap_name)
                else:
                    print("failed %s" % cap_name)
                #
            #
        except:
            pass
            #
        #


################
### ERK wave drawing
################
def draw_erks(filename, ERK3D, coord):
    '''
    draw ERK 3 types waves,
    ERK,
    dERK/dx,
    dERK/dt
    '''
    coordx = coord[0, :, 0]
    df     = pd.DataFrame({
        "x"     : coordx,
        "ERK"   : ERK3D[0, :, 0],
        "ERKdx" : ERK3D[1, :, 0],
        "ERKdt" : ERK3D[2, :, 0]
    })
    r = pyper.R(use_pandas = True)
    r("library(tidyverse)")
    r.assign("df", df)
    comg = [
        "df <- gather(df, key = type, value = amp, -x)",
        "df$type <- factor(df$type, levels = unique(df$type))",
        #"print(str(df))",
        "g000 <- ggplot(df, aes(x, amp)) + geom_line(",
        "  aes(color = type)",
        ") + facet_grid(",
        "  type ~ .",
        ") + theme_bw(",
        ") + theme(",
        "  legend.position = 'none'",
        ") + labs(",
        "  y = 'amplitude'",
        ")",
        "ggsave(plot = g000, file = '%s', width = 160, height = 120, unit = 'mm')" % filename
    ]
    r("\n".join(comg))
    if os.path.exists(filename):
        print("saved %s" % filename)
    else:
        print("failed %s" % filename)
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
