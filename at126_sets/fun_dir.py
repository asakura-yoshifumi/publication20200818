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
import re
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
        self.outdir  = outdir
        self.outhead = outhead
        self.ext     = ext
        self.reg_com = reg_com
        self.sim_com = sim_com
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
        ### markdown using list
        self.mdlist = [
            filename, ### txt file
            "%s/summary_reg.csv" % self.outdir,
            "%s/summary_sim.csv" % self.outdir
        ]
        self.mdlist2 = [
            filename, ### txt file
            "%s/summary_reg.csv" % self.outdir,
            "%s/summary_sim.csv" % self.outdir
        ]
        self.comp_only_head = [
            filename, ### txt file
            "%s/summary_reg.csv" % self.outdir,
            "%s/summary_sim.csv" % self.outdir
        ]
        self.comp_label = []
        self.comp_only  = []
        #
        self.prev = "yet"
        #
        #
    def get_passed_paras(self):
        out = "%s/passed_parameters.csv" % self.outdir
        return(out)
        #
    def get_sim_sum(self):
        return(self.mdlist[2])
        #
        #
    def set_sim(self, paras, mod_choice, ind_level):
        self.paras  = paras
        self.mod_choice = mod_choice
        self.i_paras, self.i_mod_choice = ind_level
        #
        ### prepare
        self.param_num_str = "m%02d_p%02d" %(self.i_mod_choice, self.i_paras)
        #
        #
    def get_in_reg(self):
        return(self.tablepaths, self.methods_df, self.i_meth, self.i_input)
        #
    def get_in_sim(self):
        return(self.paras, self.mod_choice, self.i_paras, self.i_mod_choice)
        #
        #
    def names_sim(self, another_color = []):
        out = {}
        out["heat_name"]     = "%s_%s_field_Euler.png"          %(self.head_d_f_sim, self.param_num_str)
        out["Vxname"]        = "%s_%s_field_Vx.png"             %(self.head_d_f_sim, self.param_num_str)
        out["Vyname"]        = "%s_%s_field_Vy.png"             %(self.head_d_f_sim, self.param_num_str)
        out["Rhoname"]       = "%s_%s_field_Rho.png"            %(self.head_d_f_sim, self.param_num_str)
        out["ERKname"]       = "%s_%s_field_ERK.png"            %(self.head_d_f_sim, self.param_num_str)
        out["part_name"]     = "%s_%s_particles_track.png"      %(self.head_d_f_sim, self.param_num_str)
        out["part_table"]    = "%s_%s_particles_track.csv"      %(self.head_d_f_sim, self.param_num_str)
        out["Lag_Rdata"]     = "%s_%s_sim/Lagrange.Rdata"       %(self.head_d_f_sim, self.param_num_str)
        out["integ_place"]   = "%s_%s_sim"                      %(self.head_d_f_sim, self.param_num_str)
        out["integ_value"]   = "%s_%s_sim/integrated.npy"       %(self.head_d_f_sim, self.param_num_str)
        out["integ_time"]    = "%s_%s_sim/timerange.npy"        %(self.head_d_f_sim, self.param_num_str)
        out["integ_coord"]   = "%s_%s_sim/coord.npy"            %(self.head_d_f_sim, self.param_num_str)
        out["animation"]     = "%s_%s_animation"                %(self.head_d_f_sim, self.param_num_str)
        out["ani_capture"]   = "%s_%s_ani_cap_last.png"         %(self.head_d_f_sim, self.param_num_str)
        out["source"]        = "%s_%s_sim/source.npy"           %(self.head_d_f_sim, self.param_num_str)
        out["erks_name"]     = "%s_%s_ERK_last.png"             %(self.head_d_f_sim, self.param_num_str)
        out["Rdata"]         = "%s_%s_sim/fields.Rdata"         %(self.head_d_f_sim, self.param_num_str)
        out["another_color"] = {
            "all":             "%s_%s_field_color2_Euler.png"   %(self.head_d_f_sim, self.param_num_str),
            "Vx":              "%s_%s_field_color2_Vx.png"      %(self.head_d_f_sim, self.param_num_str),
            "Vy":              "%s_%s_field_color2_Vy.png"      %(self.head_d_f_sim, self.param_num_str),
            "Rho":             "%s_%s_field_color2_Rho.png"     %(self.head_d_f_sim, self.param_num_str),
            "ERK":             "%s_%s_field_color2_ERK.png"     %(self.head_d_f_sim, self.param_num_str),
            "part":            "%s_%s_particles_color2.png"     %(self.head_d_f_sim, self.param_num_str)
        }
        out["pmodel_track"]  = "%s_%s_Pmodel_track.png"         %(self.head_d_f_sim, self.param_num_str)
        out["pmodel_color"]  = "%s_%s_Pmodel_track_color2.png"  %(self.head_d_f_sim, self.param_num_str)
        out["pmodel_Rdata"]  = "%s_%s_sim/Pmodel.Rdata"         %(self.head_d_f_sim, self.param_num_str)
        #
        out["comp_R"]        = "%s_%s_sim/compare.Rdata"        %(self.head_d_f_sim, self.param_num_str)
        out["comp_gif"]      = "%s_%s_particles_compare.gif"    %(self.head_d_f_sim, self.param_num_str)
        out["comp_all"]      = "%s_%s_p_track_compare.png"      %(self.head_d_f_sim, self.param_num_str)
        #
        out["param_num"]     = self.param_num_str
        #
        self.mdlist.append(out["heat_name"])
        self.mdlist.append(out["part_name"])
        #
        line  = os.getcwd()
        ### select save fig
        tmp_h = "%s/%s" % (line, out["heat_name"])
        if "heat" in another_color:
            tmp_h = "%s/%s" % (line, out["another_color"].get("all"))
        tmp_p = "%s/%s" % (line, out["part_name"])
        if "part" in another_color:
            tmp_p = "%s/%s" % (line, out["another_color"].get("part"))
        #
        table_md = [
            "|%s_field_Euler|%s_particles_track|" % (self.param_num_str, self.param_num_str),
            "|---|---|",
            "|![](%s)|![](%s)|\n"                   % (tmp_h, tmp_p)
        ]; #print("\n".join(table_md))
        if not self.sim_com is None:
            if self.i_mod_choice < len(self.sim_com):
                table_md = [self.sim_com[self.i_mod_choice]] + table_md
                #self.mdlist2.append("\n".join([self.sim_com[self.i_mod_choice] + "\n"]))
        #print("\n".join(table_md))
        ### select save fig 2nd row
        tmp_v = "%s/%s" % (line, out["Vxname"])
        if "Vx" in another_color:
            tmp_v = "%s/%s" % (line, out["another_color"].get("Vx"))
        tmp_a = "%s/%s" % (line, out["ani_capture"])
        #
        table_md2 = [
            "|%s_field_Vx|%s_ani_capture|" % (self.param_num_str, self.param_num_str),
            "|---|---|",
            "|![](%s)|![](%s)|\n"          % (tmp_v, tmp_a)
        ]
        #
        ### select save fig 3rd row
        #tmp_r = "%s/%s" % (line, out["Rhoname"]) ### later, overwrite instead
        tmp_r = "%s/%s" % (line, out["comp_all"])
        tmp_p = "%s/%s" % (line, out["pmodel_track"])
        #
        table_md3 = [
            "|%s_compare|%s_Pmodel_track|" % (self.param_num_str, self.param_num_str),
            "|---|---|",
            "|![](%s)|![](%s)|\n"          % (tmp_r, tmp_p)
        ]
        #
        if (not self.prev == "sim") and (not self.prev == "yet"):
            #table_md = ["<div style='page-break-before:always'></div>\n"] + table_md
            self.mdlist2.append("\n<div style='page-break-before:always'></div>\n")
        self.mdlist2.append("\n".join(table_md))
        self.mdlist2.append("\n".join(table_md2))
        self.mdlist2.append("\n".join(table_md3))
        self.mdlist2.append("\n<div style='page-break-before:always'></div>\n")
        self.prev = "sim"
        #
        self.comp_label.append("%s_compare" % self.param_num_str)
        self.comp_only.append(tmp_r)
        #
        return(out)
        #
    def get_mdlist(self):
        '''
        returns a list to include in markdown summary
        '''
        time_stmp = dtt.datetime.today().strftime("%Y%m%d_%H%M")[2:]
        out = "%s/%s_summary.md" % (self.outdir, time_stmp)
        return([out, self.mdlist])
        #
    def get_mdlist2(self, insertion = []):
        '''
        returns a list to include in markdown summary with figures in tables
        '''
        time_stmp = dtt.datetime.today().strftime("%Y%m%d_%H%M")[2:]
        out = "%s/%s_summary.md" % (self.outdir, time_stmp)
        if len(insertion) > 0:
            for j in range(0, len(insertion)):
                self.mdlist2.insert((3 + j), insertion[j])
        return([out, self.mdlist2])
        #
    def get_comp_only(self, insertion = []):
        #comp_only = self.comp_only
        ### put all figures into tables
        #
        if len(self.comp_label) <= 1:
            out = []
            if len(insertion) > 0:
                out = self.comp_only_head
                for j in range(0, len(insertion)):
                    out.insert((3 + j), insertion[j])
            return(out)
        #
        comp_only = []
        #
        pages = divmod(len(self.comp_only), 6)
        if pages[1] == 1:
            pages = [pages[0] - 1, 7]
            listk = [4, 3]
        else:
            listk = [pages[1]]
        #
        def table_command(j):
            indice = [6 * j + k for k in range(0, 6)]
            for k, ind in enumerate(indice):
                if ind >= len(self.comp_only):
                    indice[k] = 0
            return([
                "\n<div style='page-break-before:always'></div>\n",                       #1
                "|%s|%s|"           % (self.comp_label[indice[0]], self.comp_label[indice[1]]),     #2
                "|---|---|",                                                              #3
                "|![](%s)|![](%s)|" % (self.comp_only[ indice[0]], self.comp_only[ indice[1]]),     #4
                "|%s|%s|"           % (self.comp_label[indice[2]], self.comp_label[indice[3]]),     #5
                "|![](%s)|![](%s)|" % (self.comp_only[ indice[2]], self.comp_only[ indice[3]]),     #6
                "|%s|%s|"           % (self.comp_label[indice[4]], self.comp_label[indice[5]]),     #7
                "|![](%s)|![](%s)|" % (self.comp_only[ indice[4]], self.comp_only[ indice[5]]),     #8
                "\n"
            ])
        def odd_table(j, remainder):
            if remainder % 2 == 1:
                return([
                    "|%s|end|" % self.comp_label[(6 * j + remainder - 1)],
                    "|![](%s)|end|" % self.comp_only[ (6 * j + remainder - 1)],
                    "\n"
                ])
            else:
                return(["\n"])
                #
            #
        #
        for j in range(0, pages[0]):
            comp_only.append("\n".join(table_command(j)))
        try:
            j += 1; #print(j)
        except:
            j = 0
        try:
            for k in range(0, len(listk)):
                if listk[k] == 5:
                    comp_only.append("\n".join(table_command(j)[:6] + odd_table(j, listk[k])))
                elif listk[k] == 4:
                    comp_only.append("\n".join(table_command(j)[:6] + odd_table(j, listk[k])))
                elif listk[k] == 3:
                    comp_only.append("\n".join(table_command(j)[:4] + odd_table(j, listk[k])))
                elif listk[k] == 2:
                    comp_only.append("\n".join(table_command(j)[:4] + odd_table(j, listk[k])))
                    #
                #
            #
        except:
            print("ERROR occured in fun_dir.Name_Maker.get_comp_only")
            print(pages)
            print(listk)
            print(len(self.comp_label))
            print(len(self.comp_only))
            tb.print_exc()
        #
        #
        if len(insertion) > 0:
            out = self.comp_only_head + comp_only
            for j in range(0, len(insertion)):
                out.insert((3 + j), insertion[j])
            #
        else:
            out = comp_only
            #
        return(out)
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
        dir_f  = "%s_%s_fields" %(self.head_d_f_sim, self.param_num_str)
        files  = os.listdir(dir_f)
        #for file in files:
        #    print(file)
        files2 = []
        for file in files:
            if ".npy" in file:
                print("    loading %s/%s" % (dir_f, file))
                files2.append("%s/%s" % (dir_f, file))
        out = [np.load(file) for file in files2]
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
    def draw_pub_fig(self):
        path = "%s/sub_pub_graph.R" % os.path.dirname(__file__)
        with open(path, mode = "r") as f:
            cmd = f.read()
        cmd = re.sub("__result__", self.dir, cmd)
        r   = pyper.R()
        r(cmd)
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


################
###
################
if __name__ == '__main__':
    main()


###
