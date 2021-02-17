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
### .npy file name
################
def name_extender(name):
    '''
    add "extend" to name, between the name and "."
    '''
    if False: #not ".npy" in name:
        return(name)
        #
    else:
        head = name.split(".")[0]
        for j in range(0, 100):
            newname = "".join([head, "_ext%03d.npy" % j])
            if not os.path.exists(newname):
                break
        return(newname)
        #
        #
    #

def find_max_ext(name):
    if not ".npy" in name:
        return(name)
        #
    else:
        head = name.split(".")[0]
        j = 0
        newname = "".join([head, "_ext%03d.npy" % j])
        if not os.path.exists(newname):
            return(name)
        ### else below
        for j in range(1, 100):
            newname = "".join([head, "_ext%03d.npy" % j])
            if not os.path.exists(newname):
                break
        extmax = "".join([head, "_ext%03d.npy" % (j - 1)])
        return(extmax)
        #
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
