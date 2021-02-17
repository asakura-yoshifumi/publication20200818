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


################
### import
################
import os


################
### function
################
def spaces(x):
    if x.count("\n") >= 5:
        tmp = x.split("\n")
        #for line in tmp:
        #    print(line)
        for j in range(0, len(x)):
            if not tmp[1][j] == " ":
                break
                #
            #
        #
        return("\n".join([line[j:] for line in tmp]))
    else:
        return(x)

def R_manual(comg, joined = False, dir = None, file = "graphs_written.R"):
    if dir is None:
        tmp = __file__.rsplit("/", 2)[1]
        #print(__file__)
        dir_in = "at%s_sets/tmp_data" % tmp[2:5]
    else:
        dir_in = dir
        #
    #
    if os.path.exists(dir_in):
        #print("exisits %s" % dir_in)
        file_write = "/".join([dir_in, file])
        if os.path.exists(file_write):
            with open(file_write, mode = "r") as f:
                lines = f.read()
                #
            #
        else:
            lines = ""
            #
        #
        with open(file_write, mode = 'a') as f:
            if joined:
                joined_comg = comg
                separated   = comg.split("\n")
            else:
                joined_comg = "\n".join([spaces(tmp) for tmp in comg])
                separated   = comg
                #
            #
            #if not joined_comg in lines:
            lens = len(separated)
            if sum([(separated[j] in lines) or ("########" in separated[j]) for j in range(0, min(4, lens))]) < min(4, lens):
                f.write(joined_comg + "\n")
    else:
        #print("not exisits %s" % dir_in)
        pass
    #
    #
    #


###
