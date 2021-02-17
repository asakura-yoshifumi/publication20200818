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

#from scipy.stats import multivariate_normal as multinormal
from scipy.integrate import RK45

import math
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib import animation as ani

import pandas as pd

### handwrite
from .fun_dir import di1
from .fun_dir import di2


class Euler_diff:
    def __init__(self, fun, t0, y0, t_bound, first_step, max_step):
        self.fun = fun
        self.t   = t0
        self.y   = y0
        self.t_bound    = t_bound
        self.status     = "running"
        self.first_step = first_step
        self.max_step   = max_step
        #
        self.t_old      = t0
        #
        self.step_size  = first_step
        #
    def set_dt(self, dt):
        self.step_size = dt
        #
    def step(self):
        if self.status == "running":
            self.t_old = self.t
            #
            self.t += self.step_size
            self.y += self.fun(self.t, self.y) * self.step_size
        else:
            self.t = None
            self.y = None
        #
        if self.t >= self.t_bound:
            self.status = "finished"
        #
        #
