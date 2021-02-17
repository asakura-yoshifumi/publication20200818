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

import pandas as pd



################
### overwrite Sim_Part with graph settings
################
#class Sim_Part_Py(Sim_Part):
#    pass ### not needed.


################
###
################
class Particle_model_draw:
    def __init__(self, timerange, xve, paras, draw_skip = 1, draw_timerange = None):
        if draw_timerange is None:
            self.timerange = timerange
            self.xve       = xve
        else:
            stind = np.argmin(np.absolute(timerange - draw_timerange[0]))
            enind = np.argmin(np.absolute(timerange - draw_timerange[1]))
            #self.timerange = timerange[stind:enind, :]
            #self.xve       = xve[stind:enind, :, :, :]
            self.timerange = timerange[stind:, :]; #print(self.timerange)
            self.xve       = xve[stind:, :, :]
        #
        self.paras     = paras
        self.paras_str = "mu0=%04.3f, alpha=%04.3f, beta=%04.3f\nk0=%04.3f, R0=%04.3f, eta=%04.3f\nr=%04.3f, erkspeed=%04.3f, sigma=%04.3f" % tuple([x for x in paras[0:9]])
        self.draw_skip = draw_skip
        #
        self.num_cell  = self.xve.shape[1]
        #
        self.slicer = 1
        lentime = self.timerange.shape[0]
        tmp     = divmod(lentime, 400)
        self.slicer = tmp[0]
        if self.slicer == 0 or tmp[1] > 0.5:
            self.slicer += 1
        #
        #
    def j_cell(self, j):
        out = {
            "time": self.timerange[0::self.slicer, 0],
            "x":    self.xve[0::self.slicer, j, 0],
            "ERK":  self.xve[0::self.slicer, j, 2],
            "cell": "cell%04d" % j
        }
        return(pd.DataFrame(out))
        #
        #
    def draw(self, filename = None, Rdata = None, another_color = None, draw_range = None):
        testing = False
        #
        #if draw_range is None:
        #    draw_range = [0.0, np.max(self.xve[0, :, 0])]
        #
        r = pyper.R(use_pandas = True)
        r("library(tidyverse)")
        #
        df = pd.concat([self.j_cell(j) for j in range(0, self.num_cell, self.draw_skip)], axis = 0).reset_index(drop = True)
        r.assign("df", df)
        #
        comg = [
            "print(str(df))",
            "g000 <- ggplot(df, aes(x, time)) + geom_point(",
            "  aes(color = ERK), size = 4",
            ")",
            "s000 <- g000",
            "g000 <- g000 + theme_classic("
        ]
        if not draw_range is None:
            comg += [
            ") + coord_cartesian(",
            "  xlim = c(%s, %s)" % tuple(draw_range)
            ]
        #
        comg += [
            ") + scale_y_reverse(",
            #") + scale_color_viridis_c(",
            ") + labs(",
            "  title = '%s'" % self.paras_str,
            ")"
        ]
        if testing:
            print(self.xve.shape)
            print(r("\n".join(comg)))
        else:
            r("\n".join(comg))
        #
        if not Rdata is None:
            #r("save.image('%s')" % Rdata)
            comg = [
                "df_spr <- df",
                "s010 <- g000",
                "save(list = c('df_spr', 's000', 's010'), file = '%s')" % Rdata
            ]
            r("\n".join(comg))
            if os.path.exists(Rdata):
                print("saved %s" % Rdata)
            else:
                print("failed %s" % Rdata)
                #
            #
        #
        if not filename is None:
            comg = [
                "g010 <- g000 + scale_color_viridis_c()",
                "ggsave(plot = g010, file = '%s', width = 160, height = 120, unit = 'mm')" % filename
            ]
            if testing:
                print(r("\n".join(comg)))
            else:
                r("\n".join(comg))
            #
            if os.path.exists(filename):
                print("saved %s" % filename)
            else:
                print("failed %s" % filename)
                #
            #
        #
        if not another_color is None:
            comg = [
                "g010 <- g000 + scale_color_gradient2(",
                "  low = 'blue', high = 'red', mid = 'white', midpoint = 0.0",
                ")",
                "ggsave(plot = g010, file = '%s', width = 160, height = 120, unit = 'mm')" % another_color
            ]
            if testing:
                print(r("\n".join(comg)))
            else:
                r("\n".join(comg))
            #
            if os.path.exists(another_color):
                print("saved %s" % another_color)
            else:
                print("failed %s" % another_color)
                #
            #
        #
        #
        #


class Comparison:
    def __init__(self, Rdata, init_range, paras = None):
        self.init_starts = init_range[0]
        #
        ### Rdata includes df and g000 above
        self.r = pyper.R(use_pandas = True, use_numpy = True)
        #
        ### choose initial x values on R
        comg = [
            "load('%s')" % Rdata,
            "df <- df_spr", ### new in 126, save data in Rdata setting change
            "df$model <- 'particle'",
            "mint <- min(df$time)",
            "df2  <- subset(df, time == mint)",
            #"df2  <- subset(df2, %d <= x)"  % init_range[0],
            #"df2  <- subset(df2, x  <  %d)" % init_range[1],
            "xs   <- df2$x"
        ]
        self.r("\n".join(comg)); #print(self.r("print(ls())")); #print(self.r("print(str(df2))"))
        #print(self.r("print(str(df))")); sys.exit()
        xs = self.r.get("xs") ### this returns 1d np.array because of True
        #
        self.init = np.zeros((xs.shape[0], 4))
        self.init[:, 0] = xs
        #
        if paras is None:
            self.paras_str = None
        else:
            self.paras_str = "mu0=%04.3f, alpha=%04.3f, beta=%04.3f\nk0=%04.3f, R0=%04.3f, eta=%04.3f\nr=%04.3f, erkspeed=%04.3f, sigma=%04.3f" % tuple([x for x in paras[0:9]])
        #
        print("Comparison initialized.")
        #
    def get_init(self):
        return(self.init)
        #
    def compare(self, df_lag, Rdata = None, gifname = None, graphname = None, another_color = None, draw_xrange = None):
        print("draw comparison")
        self.r("library(tidyverse)")
        self.r("library(gganimate)")
        #
        ### overwrite cell indice on Lagrange
        #print(df_lag["cell"].tolist()); sys.exit()
        df_lag["cell"] = ["cell%04d" % (int(tmp[4:]) * 10) for tmp in df_lag["cell"]]
        ### assign lagrange
        self.r.assign("lag", df_lag)
        comg = [
            "lag$model <- 'fluid'",
            "lag$color <- 'blue'",  ### fluid    model
            "df$color  <- 'black'", ### particle model
            "comp      <- rbind(lag, df)", ### use this.
            "df_com    <- comp" ### to save
        ]
        self.r("\n".join(comg))
        #
        if not gifname is None:
            comg = [
                "alpha_set <- 0.4",
                "a000 <- ggplot(comp, aes(x, time)) + geom_path(",
                "  data = subset(comp, model == 'particle'), mapping = aes(x, time, group = cell), alpha = alpha_set, color = 'black', size = 3",
                ") + geom_path(",
                "  data = subset(comp, model == 'fluid'),    mapping = aes(x, time, group = cell), alpha = alpha_set, color = 'blue', size = 3",
                ") + geom_point(",
                "  size = 0.2, aes(color = ERK)",
                ") + theme_bw(",
                ") + scale_color_viridis_c(",
                ") + scale_y_reverse(",
                ")"
            ]
            if not draw_xrange is None:
                comg.append("a000 <- a000 + coord_cartesian(xlim = c(%f, %f))" % tuple(draw_xrange))
            if not self.paras_str is None:
                comg.append("a000 <- a000 + labs(title = '%s')" % self.paras_str)
            #
            #print(self.r("\n".join(comg)))
            self.r("\n".join(comg))
            #
            print("rendering starts...")
            comg = [
                "anim_time <- length(unique(lag$cell)) * 0.1",
                "a100 <- a000 + transition_manual(",
                "  cell",
                ") + labs(",
                "  subtitle = '{current_frame}, black is particle, blue is fluid'",
                ")",
                "anim_save(",
                "  filename  = '%s'," % gifname,
                "  animation = a100,",
                "  renderer  = gifski_renderer(),",
                "  duration  = anim_time,",
                "  width     = 160, height = 120, unit = 'mm', res = 270",
                ")"
            ]
            #print(self.r("\n".join(comg)))
            self.r("\n".join(comg))
            if os.path.exists(gifname):
                print("saved %s" % gifname)
            else:
                print("failed %s" % gifname)
        #
        if graphname is None:
            comg_Rdata = "c('df_com')"
        else:
            comg = [
                ### non-merged preparation
                "comp$al    <- 0.4",        ### alpha
                "comp$si    <- 3.0",        ### size
                "comp$fa    <- comp$model", ### facet labels
                ### merged preparation
                "comptmp    <- comp",
                "comptmp$fa <- 'merged'",   ### facet labels
                "comptmp$al <- 0.0",        ### alpha
                "comptmp$si <- 2.0",        ### size
                ### bind merged and each
                "comp3    <- rbind(comptmp, comp)",
                "comp3$fa <- factor(comp3$fa, levels = unique(comp3$fa))",
                "comp3$time <- comp3$time - min(comp3$time)",
                "df_com   <- comp3",
                ### graph
                "a000 <- ggplot(comp3, aes(x, time)) + geom_path(",
                "  data = subset(comp3, model == 'fluid'),    mapping = aes(x, time, group = cell, size = si), alpha = 0.4, color = 'blue'",
                ") + geom_path(",
                "  data = subset(comp3, model == 'particle'), mapping = aes(x, time, group = cell, size = si), alpha = 0.4, color = 'black'",
                ") + geom_point(",
                "  aes(color = ERK, alpha = al), size = 0.1",
                ") + facet_grid(",
                "  . ~ fa",
                ") + scale_color_viridis_c(",
                ") + scale_alpha_identity(",
                "  guide = 'none'",
                ") + scale_size_identity(",
                "  guide = 'none'",
                ")",
                "c000 <- a000", ### to save
                "a000 <- a000 + scale_y_reverse(",
                ") + theme_bw(",
                ") + theme(",
                "  strip.text       = element_text(size = 16)",
                ") + coord_cartesian(",
                "  xlim = c(96, 160)",
                ")",
                ### facet color management
                """
                facet_color <- function(plot, colors, textcol){
                  g          <- ggplot_gtable(ggplot_build(plot))
                  strip_both <- which(grepl('strip-', g$layout$name))
                  fills      <- colors
                  k <- 1
                  for (i in strip_both) {
                    ### box fill
                    j <- which(grepl('rect', g$grobs[[i]]$grobs[[1]]$childrenOrder))
                    g$grobs[[i]]$grobs[[1]]$children[[j]]$gp$fill <- fills[k]
                    ### letters color
                    l <- which(grepl('title', g$grobs[[i]]$grobs[[1]]$childrenOrder))
                    m <- which(grepl('text',  g$grobs[[i]]$grobs[[1]]$children[[l]]$children))
                    g$grobs[[i]]$grobs[[1]]$children[[l]]$children[[m]]$gp$col <- textcol[k]
                    #
                    k <- k + 1
                  }
                  return(g)
                }
                """,
                "a000 <- facet_color(a000, c('white', 'royalblue1', 'grey'), c('grey10', 'white', 'grey10'))",
                "c010 <- a000",
                ### save
                "ggsave(plot = a000, file = '%s', height = 120, width = 160, unit = 'mm')" % graphname
            ]
            comg_Rdata = "c('df_com', 'c000', 'c010')"
            self.r("\n".join(comg))
            if os.path.exists(graphname):
                print("saved %s" % graphname)
            else:
                print("failed %s" % graphname)
            #
            #
        if not Rdata is None:
            self.r("save(list = %s, file = '%s')" % (comg_Rdata, Rdata))
            if os.path.exists(Rdata):
                print("saved %s" % Rdata)
            else:
                print("failed %s" % Rdata)
                #
        #
        #


###
