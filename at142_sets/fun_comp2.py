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

import pandas as pd


################
### 2D comarinson
################
class Compare2D:
    def __init__(self, R_lag, R_part):
        self.r = pyper.R()
        comg   = [
            "library(tidyverse)",
            "library(gganimate)",
            "load('%s')" % R_lag,
            "load('%s')" % R_part,
            "df_p2$model <- 'particle'",
            "df_l2$model <- 'continuum'",
            "df <- rbind(df_l2, df_p2)",
            #"len_anim <- length(unique(df$time)) * 0.1"
            "numframes <- as.integer(length(unique(df$time)) / 2)"
        ]
        self.r("\n".join(comg))
        print("2D comparison initialized")
        #
        #
    def draw(self, moviefile, lastframe = None):
        renderer = "ffmpeg_renderer()"
        comg = [
            #"print(ls())",
            "g000 <- ggplot(df, aes(x, y)) + geom_path(",
            "  aes(group = cell)",
            ") + theme_classic(",
            ") + coord_fixed(",
            "  ratio = 1.0",
            ") + transition_manual(",
            "  time_f",
            ") + labs(",
            "  title = 'time: {current_frame}'",
            ")",
            "anim_save(",
            "  filename  = '%s'," % moviefile,
            "  animation = g000,",
            "  renderer  = %s," % renderer,
            #"  duration  = len_anim,",
            "  nframes   = numframes,",
            "  fps       = 20,",
            "  width     = 160, height = 160, unit = 'mm', res = 360",
            ")"
        ]
        print("start rendering...")
        #
        #print(self.r("\n".join(comg)))
        self.r("\n".join(comg))
        #
        if os.path.exists(moviefile):
            print("saved %s" % moviefile)
        else:
            print("failed %s" % moviefile)
        #
        if not lastframe is None:
            comg = [
                "g000 <- ggplot(subset(df, time == max(df$time)), aes(x, y)) + geom_path(",
                "  aes(group = cell)",
                ") + theme_classic(",
                ") + coord_fixed(",
                "  ratio = 1.0",
                ") + labs(",
                "  title = 'difference between two models'",
                ")",
                "ggsave(plot = g000, file = '%s', width = 160, height = 160, unit = 'mm')" % lastframe
            ]
            self.r("\n".join(comg))
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
