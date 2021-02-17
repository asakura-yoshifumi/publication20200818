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
### packages
###############
'''
write basic tools for calculation, and for directory or run here.
'''

################
### packages
################
import numpy as np
#from scipy.optimize import minimize
#from scipy import stats
import time
import sys
import pandas as pd

import os
import shutil

import pyper

#import math
#import matplotlib.pyplot as plt
#import seaborn as sns


import platform   as plf
import subprocess as sbp

import datetime as dtt

import traceback as tb

### handwrite


################
### field diff
################
def field_d1(field, xy, dx = 1, perio = True):
    ### field is an array
    ### xy is a directio x or y to diff
    ### dx should be 1
    ### periodic or not
    if xy == "x":
        pass
    elif xy == "y":
        field = field.T
        #
    else:
        print("error, not defined direction.")
        sys.exit()
    #
    if field.shape[0] == 1:
        return(np.zeros((1, 1)))
    #
    if perio:
        K = np.zeros((field.shape[0], field.shape[0]))\
            + np.diag(np.ones((field.shape[0] - 1)), k = 1)\
            - np.diag(np.ones((field.shape[0] - 1)), k = -1)\
            + np.diag((1.0,), k = (1 - field.shape[0]))\
            - np.diag((1.0,), k = (field.shape[0] - 1))
        #
        #print(K)
        out = K.dot(field)
        #
    else:
        K = np.zeros((field.shape[0], field.shape[0]))\
            + np.diag(np.ones((field.shape[0] - 1)), k = 1)\
            - np.diag(np.ones((field.shape[0] - 1)), k = -1)\
            + np.diag((1.0,), k = (1 - field.shape[0]))\
            - np.diag((1.0,), k = (field.shape[0] - 1))
        K[0, 1] = 0.0
        K[field.shape[0] - 1, field.shape[0] - 2] = 0.0
        out = K.dot(field)
        #
    #
    if xy == "y":
        out = out.T
        #
    #
    return(out)
    #
    #

def field_d2(field, xy, dx = 1, perio = True):
    ### field is an array
    ### xy is a directio x or y to diff
    ### dx should be 1
    ### periodic or not
    if xy == "x":
        pass
    elif xy == "y":
        field = field.T
        #
    else:
        print("error, not defined direction.")
        sys.exit()
    #
    if field.shape[0] == 1:
        return(np.zeros((1, 1)))
    #
    if perio:
        K = np.zeros((field.shape[0], field.shape[0]))\
            - np.diag(np.ones((field.shape[0])), k = 0) * 2.0\
            + np.diag(np.ones((field.shape[0] - 1)), k = 1)\
            + np.diag(np.ones((field.shape[0] - 1)), k = -1)\
            + np.diag((1.0,), k = (1 - field.shape[0]))\
            + np.diag((1.0,), k = (field.shape[0] - 1))
        #
        #print(K)
        out = K.dot(field)
        #
    else:
        K = np.zeros((field.shape[0], field.shape[0]))\
            - np.diag(np.ones((field.shape[0])), k = 0) * 2.0\
            + np.diag(np.ones((field.shape[0] - 1)), k = 1)\
            + np.diag(np.ones((field.shape[0] - 1)), k = -1)\
            + np.diag((1.0,), k = (1 - field.shape[0]))\
            + np.diag((1.0,), k = (field.shape[0] - 1))
        K[0, 1] = 2.0
        K[field.shape[0] - 1, field.shape[0] - 2] = 2.0
        out = K.dot(field)
    #
    if xy == "y":
        out = out.T
        #
    #
    return(out)
    #
    #

def di1(field, xy, dx = 1, perio = True):
    ### this calculates differential of the field, instead of just diff
    return(0.5 * field_d1(field, xy, dx, perio) / dx)
    #

def di2(field, xy, dx = 1, perio = True):
    return(1.0 * field_d2(field, xy, dx, perio) / (dx**2.0))
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
### markdown
################
'''
moved to another file
'''


################
### directory setting
################
def dir_reset(dirname, options = True):
    if os.path.exists(dirname) and options:
        shutil.rmtree(dirname)
        os.mkdir(dirname)
        #
    elif not os.path.exists(dirname):
        os.mkdir(dirname)
        #
    #


################
###
################
class Name_Maker:
    def __init__(self, outdir, outhead, ext, reg_com = None, sim_com = None):
        self.outdir    = outdir
        self.outhead   = outhead
        self.ext       = ext
        self.reg_com   = reg_com
        self.sim_com   = sim_com
        #
        #
        ### versions information
        filename = "%s/versions_info.txt" % self.outdir
        dir_reset(self.outdir, False)
        with open(filename, mode = "w") as f:
            tmp = sbp.run("pip list --format columns".split(" "), stdout = sbp.PIPE)
            mes = [
                "conducted in",
                os.getcwd(),
                "\nenvironments",
                sys.version,
                "\n",
                plf.platform(),
                "\n",
                tmp.stdout.decode()
            ]
            f.write("\n".join(mes))
        #
        reg = "%s/regression" % self.outdir
        sim = "%s/simulation" % self.outdir
        dir_reset(reg, False)
        dir_reset(sim, False)
        self.head_d_f_reg = "%s/%s" % (reg, self.outhead)
        self.head_d_f_sim = "%s/%s" % (sim, self.outhead)
        #
        #
    def get_passed_paras(self):
        out = "%s/passed_parameters.csv" % self.outdir
        return(out)
        #
        #
        #
    def set_sim_loop(self, paras, ind_level):
        self.paras  = paras
        #self.mod_choice = mod_choice
        self.i_paras, self.i_mod_choice, self.ind_angle = ind_level
        #
        ### prepare
        #self.param_num_str = "m%02d_p%02d" %(self.i_mod_choice, self.i_paras)
        self.param_num_str = "angle%03d_param%03d_fmodel%03d" %(self.ind_angle, self.i_paras, self.i_mod_choice)
        self.particle_str  = "angle%03d_param%03d_particle"   %(self.ind_angle, self.i_paras)
        #
        #
    def names_sim(self):
        out = {}
        #
        #
        ################
        ### output names
        ################
        #
        ### tests
        dir_test  = "%s_%s" % (self.head_d_f_sim, "test")
        dir_reset(dir_test, False)
        tmp       = "test_arr_grid_density"
        out[tmp]  = "%s/%s.npy" %(dir_test, tmp)
        tmp       = "test_mov_grid_density"
        out[tmp]  = "%s/%s.mp4" %(dir_test, tmp)
        tmp       = "test_arr_cell_density"
        out[tmp]  = "%s/%s.npy" %(dir_test, tmp)
        tmp       = "test_mov_cell_density"
        out[tmp]  = "%s/%s.mp4" %(dir_test, tmp)
        #
        #
        ### simulation directories
        dir_p   = "%s_%s" %(self.head_d_f_sim, self.particle_str)
        dir_reset(dir_p, False)
        #
        out["df_log"] = "%s_logs.txt" %(self.head_d_f_sim)
        #
        #
        ### particle
        tmp       = "keep_particle_integ"
        out[tmp]  = "%s/%s.npy" %(dir_p, tmp)
        tmp       = "keep_particle_time"
        out[tmp]  = "%s/%s.npy" %(dir_p, tmp)
        tmp       = "keep_particle_erk"
        out[tmp]  = "%s/%s.npy" %(dir_p, tmp)
        #
        tmp       = "keep_particle_R"
        out[tmp]  = "%s/%s.Rdata" %(dir_p, tmp)
        tmp       = "graph_mov_part2d"
        out[tmp]  = "%s/%s.mp4"   %(dir_p, tmp)
        tmp       = "graph_x_t_part2d"
        out[tmp]  = "%s/%s.png"   %(dir_p, tmp)
        #
        tmp       = "graphdir_snapshot_particle"
        out[tmp]  = "%s/%s"       %(dir_p, tmp)
        dir_reset(out[tmp], False)
        #
        ### field reconstruction from particle data
        tmp       = "keep_recP_grid_TimeLoc14val"
        out[tmp]  = "%s/%s.npy" %(dir_p, tmp)
        tmp       = "keep_recP_grid_TimeLocWei6x4"
        out[tmp]  = "%s/%s.npy" %(dir_p, tmp)
        tmp       = "keep_recP_grid_density_KDE"
        out[tmp]  = "%s/%s.npy" %(dir_p, tmp)
        tmp       = "graph_recP_grid_density_KDE"
        out[tmp]  = "%s/%s.mp4" %(dir_p, tmp)
        tmp       = "keep_recP_cell_TimeLoc14val"
        out[tmp]  = "%s/%s.npy" %(dir_p, tmp)
        tmp       = "keep_recP_cell_TimeLocWei6x4"
        out[tmp]  = "%s/%s.npy" %(dir_p, tmp)
        tmp       = "keep_recP_cell_density_KDE"
        out[tmp]  = "%s/%s.npy" %(dir_p, tmp)
        tmp       = "graph_recP_cell_density_KDE"
        out[tmp]  = "%s/%s.mp4" %(dir_p, tmp)
        #
        tmp       = "graphdir_recP_cell_values"
        out[tmp]  = "%s/%s" %(dir_p, tmp)
        dir_reset(out[tmp], False)
        #
        tmp       = "keep_recP_cell_acc_x"
        out[tmp]  = "%s/%s.npy" %(dir_p, tmp)
        tmp       = "graphdir_recP_cell_acc_x"
        out[tmp]  = "%s/%s" %(dir_p, tmp)
        dir_reset(out[tmp], False)
        tmp       = "keep_recP_cell_acc_y"
        out[tmp]  = "%s/%s.npy" %(dir_p, tmp)
        tmp       = "graphdir_recP_cell_acc_y"
        out[tmp]  = "%s/%s" %(dir_p, tmp)
        dir_reset(out[tmp], False)
        #
        tmp       = "graph_recP_grid_mov_VRho_Erk"
        out[tmp]  = "%s/%s.mp4" %(dir_p, tmp)
        tmp       = "keep_recP_cell_Acceleration"
        out[tmp]  = "%s/%s.npy" %(dir_p, tmp)
        tmp       = "graphdir_recP_Acceleration"
        out[tmp]  = "%s/%s"     %(dir_p, tmp)
        dir_reset(out[tmp], False)
        #
        #
        ################
        ### output names
        ################
        ### fluid
        #
        ### simulation directories
        dir_f   = "%s_%s" %(self.head_d_f_sim, self.param_num_str)
        dir_reset(dir_f, False)
        ### layer 2
        dir_l2e = "%s/%s" %(dir_f, "euler")
        dir_reset(dir_l2e, False)
        dir_l2l = "%s/%s" %(dir_f, "lagrange")
        dir_reset(dir_l2l, False)
        dir_l2c = "%s/%s" %(dir_f, "compare")
        dir_reset(dir_l2c, False)
        #
        ################
        ### output names
        ################
        ### fluid euler
        tmp       = "keep_euler_integ"
        out[tmp]  = "%s/%s.npy" %(dir_l2e, tmp)
        tmp       = "keep_euler_time"
        out[tmp]  = "%s/%s.npy" %(dir_l2e, tmp)
        tmp       = "keep_euler_erk"
        out[tmp]  = "%s/%s.npy" %(dir_l2e, tmp)
        tmp       = "keep_euler_coord"
        out[tmp]  = "%s/%s.npy" %(dir_l2e, tmp)
        tmp       = "keep_euler_field"
        out[tmp]  = "%s/%s.npy" %(dir_l2e, tmp)
        #
        gdir      = "graphdir_snapshot_euler"
        out[gdir] = "%s/%s"    %(dir_l2e, gdir)
        dir_reset(out[gdir], False)
        tmp       = "graph_mov_euler"
        out[tmp]  = "%s/%s.mp4" %(dir_l2e, tmp)
        #
        ### fluid lagrange
        tmp       = "keep_lagrange_integ"
        out[tmp]  = "%s/%s.npy" %(dir_l2l, tmp)
        tmp       = "keep_lagrange_time"
        out[tmp]  = "%s/%s.npy" %(dir_l2l, tmp)
        tmp       = "keep_lagrange_erk"
        out[tmp]  = "%s/%s.npy" %(dir_l2l, tmp)
        #
        tmp       = "keep_lagrange_R"
        out[tmp]  = "%s/%s.Rdata" %(dir_l2c, tmp)
        tmp       = "graph_mov_lagrange"
        out[tmp]  = "%s/%s.mp4"   %(dir_l2c, tmp)
        tmp       = "graph_x_t_lagrange"
        out[tmp]  = "%s/%s.png"   %(dir_l2c, tmp)
        #
        ### compare fluid and particle
        tmp       = "graph_mov_compare"
        out[tmp]  = "%s/%s.mp4"   %(dir_l2c, tmp)
        tmp       = "graph_last_compare"
        out[tmp]  = "%s/%s.png"   %(dir_l2c, tmp)
        #
        tmp       = "graphdir_snapshot_lagrange"
        out[tmp]  = "%s/%s"       %(dir_l2c, tmp)
        dir_reset(out[tmp], False)
        #
        #
        ### field reconstruction from lagrange data
        tmp       = "keep_recL_grid_TimeLoc14val"
        out[tmp]  = "%s/%s.npy" %(dir_l2l, tmp)
        tmp       = "keep_recL_grid_TimeLocWei6x4"
        out[tmp]  = "%s/%s.npy" %(dir_l2l, tmp)
        tmp       = "keep_recL_grid_density_KDE"
        out[tmp]  = "%s/%s.npy" %(dir_l2l, tmp)
        tmp       = "graph_recL_grid_density_KDE"
        out[tmp]  = "%s/%s.mp4" %(dir_l2l, tmp)
        tmp       = "keep_recL_cell_TimeLoc14val"
        out[tmp]  = "%s/%s.npy" %(dir_l2l, tmp)
        tmp       = "keep_recL_cell_TimeLocWei6x4"
        out[tmp]  = "%s/%s.npy" %(dir_l2l, tmp)
        tmp       = "keep_recL_cell_density_KDE"
        out[tmp]  = "%s/%s.npy" %(dir_l2l, tmp)
        tmp       = "graph_recL_cell_density_KDE"
        out[tmp]  = "%s/%s.mp4" %(dir_l2l, tmp)
        #
        tmp       = "graphdir_recL_cell_values"
        out[tmp]  = "%s/%s" %(dir_l2l, tmp)
        dir_reset(out[tmp], False)
        #
        tmp       = "keep_recL_cell_acc_x"
        out[tmp]  = "%s/%s.npy" %(dir_l2l, tmp)
        tmp       = "graphdir_recL_cell_acc_x"
        out[tmp]  = "%s/%s" %(dir_l2l, tmp)
        dir_reset(out[tmp], False)
        tmp       = "keep_recL_cell_acc_y"
        out[tmp]  = "%s/%s.npy" %(dir_l2l, tmp)
        tmp       = "graphdir_recL_cell_acc_y"
        out[tmp]  = "%s/%s" %(dir_l2l, tmp)
        dir_reset(out[tmp], False)
        #
        tmp       = "graph_recL_grid_mov_VRho_Erk"
        out[tmp]  = "%s/%s.mp4" %(dir_l2l, tmp)
        tmp       = "keep_recL_cell_Acceleration"
        out[tmp]  = "%s/%s.npy" %(dir_l2l, tmp)
        tmp       = "graphdir_recL_Acceleration"
        out[tmp]  = "%s/%s"     %(dir_l2l, tmp)
        dir_reset(out[tmp], False)
        #
        return(out)
        #
        #
    def save_fields(self, fields):
        dir_f = "%s_%s_fields" %(self.head_d_f_sim, self.param_num_str)
        files = ["%s/array%02d.npy" % (dir_f, j) for j in range(0, len(fields))]
        dir_reset(dir_f, False)
        #
        for j, field_j in enumerate(fields):
            np.save(file = files[j], arr = field_j)
            #
        #
    def load_fields(self):
        dir_f = "%s_%s_fields" %(self.head_d_f_sim, self.param_num_str)
        '''
        files  = os.listdir(dir_f)
        #for file in files:
        #    print(file)
        files2 = []
        for file in files:
            if ".npy" in file:
                print("    loading %s/%s" % (dir_f, file))
                files2.append("%s/%s" % (dir_f, file))
        out = [np.load(file) for file in files2]
        '''
        fields = os.listdir(dir_f) ### only length is needed
        files = ["%s/array%02d.npy" % (dir_f, j) for j in range(0, len(fields))]
        out = [np.load(file) for file in files]
        return(out)
        #
        #


class Data_Arrange:
    def __init__(self, resultdir):
        self.table = {"compare": [], "euler": [], "lagrange": [], "spring": []}
        self.label = []
        #
        self.dir       = resultdir
        self.tablename = "%s/paths_to_Rdata.csv" % resultdir
        #
    def add_row(self, names_sim):
        self.label.append(names_sim["param_num"])
        self.table["compare" ].append(names_sim["comp_R"])
        self.table["euler"   ].append(names_sim["Rdata"])
        self.table["lagrange"].append(names_sim["Lag_Rdata"])
        self.table["spring"  ].append(names_sim["pmodel_Rdata"])
        #
    def get_Rdata_paths(self):
        out0 = pd.DataFrame({"label": self.label})
        out1 = pd.DataFrame(self.table)
        out  = pd.concat([out0, out1], axis = 1)
        return(out)
        #
    def save_Rdata_paths(self, file = None, one_Rdata = True):
        if file is None:
            out = self.tablename
        else:
            out = file
        df = self.get_Rdata_paths()
        df.to_csv(out, header = True, index = False)
        #
        ### make all Rdata onto one / setting
        if one_Rdata:
            dir = "%s/graphs_Rdata" % self.dir
            dir_reset(dir)
            for j, label in enumerate(self.label):
                filej = "%s/%s.Rdata" % (dir, label)
                rowj  = df.iloc[j, 1:].tolist()
                #
                r = pyper.R()
                for k, path in enumerate(rowj):
                    r("load('%s')" % path)
                    #
                #
                r("save.image('%s')" % filej)
                if os.path.exists(filej):
                    print("saved %s" % filej)
                else:
                    print("failed %s" % filej)
                    #
                #
            #
        #


################
### path
################
def find_up(path):
    if "/" in path:
        path0 = path.split("/")[0]
    else:
        path0 = path
    #print(path0)
    path1 = "../"
    pathout = ""
    while len(path1) < 30:
        if any([path0 in j for j in os.listdir(path1)]):
            pathout = path1 + path
            break
        else:
            path1 = "../" + path1
        #
    #
    return(pathout)
    #


################
### time counter
################
class Time_keeper:
    def __init__(self):
        self.start_t = []
        self.start_t.append(time.time())
        #
        self.yet_init_rclone = True
        dn = dtt.datetime.now()
        self.starttime = "%02d%02d%02d_%02d%02d" % (dn.year % 100, dn.month, dn.day, dn.hour, dn.minute)
        #
    def start_count(self, print_i = True):
        self.start_t.append(time.time())
        if print_i:
            print(len(self.start_t))
        #
    def get_indice(self):
        return(len(self.start_t))
        #
    def get_elapsed(self, index = 0, seconds = False):
        took_time = int(time.time() - self.start_t[index])
        if seconds:
            out = took_time
            #
        else:
            el_hr = divmod(took_time, 3600)
            el_mi = divmod(el_hr[1],  60)
            out   = "%d hr %02d min %02d sec" % (el_hr[0], el_mi[0], el_mi[1])
            #
            #
        #
        return(out)
        #
        #


################
###
################
def main():
    pass
    #

################
###
################
if __name__ == '__main__':
    main()


###
