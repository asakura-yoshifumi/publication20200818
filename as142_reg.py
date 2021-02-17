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
################
import numpy as np
import math
import sys
import os
import shutil

import pandas as pd

import traceback as tb
import getpass

### handwrite
from at142_sets import *


################
### almost the same
################
def main_args(
    file_paras,
    forj = None,
    fork = None,
    forl = None,
    fort = None,
    forp = None,
    form = None,
    proc = 0
):
    ################
    ### timer
    ################
    tmkpr = Time_keeper()
    to_rclone = []
    #
    #
    ################
    ### general settings
    ################
    gset      = sub_global()
    outdirs   = ["result142"]
    outheads  = ["result142"]
    ext       = "png"
    #
    option_TF = gset["force_erace"] or (gset["training_now"] and gset["test_data_now"])
    dir_reset(outdirs[0], option_TF and (proc == 0)); #sys.exit()
    #
    #
    ################
    ### for regression
    ################
    #
    #
    ################
    ### for simulation, some may passed from regression
    ################
    ### to store parameters passed from regression
    paras_store_reg = None
    #
    #
    ### for simulation
    #file_paras  = "./at142_sets/setting_tables/passed_test_191213.csv"
    #
    force_paras = None
    if gset["read_parameters"]:
        force_paras = pd.read_csv(file_paras)
        if force_paras.get("amp_dt") is None:
            force_paras["amp_dt"] = gset["amp_list"][1]
        ### prepare force_num
        #
    #
    ### theta settings
    index_theta = 0
    thetas_pi = [j * 0.25 * 0.5 for j in range(5)]
    #
    #
    ################
    ### names maker
    ################
    nm = Name_Maker(outdirs[index_theta], outheads[0], ext, sim_com = gset["model_notes"])
    #
    da = Data_Arrange(outdirs[index_theta])
    #
    #
    #
    '''
    ----------------------------------------------------------------
    setting part is over
    start running the commands
    ----------------------------------------------------------------
    '''
    #
    #
    ### parameter preparation for simulation
    if paras_store_reg is None:
        paras_in = force_paras
    else:
        paras_in = pd.concat(
            [paras_store_reg, force_paras],
            axis = 0
        ).reset_index(drop = True)
        ### above can get None and work properly
    #
    ### save parameter list
    paras_in.to_csv(nm.get_passed_paras(), header = True, index = False)
    print("parameters below are passed to simulation")
    print(paras_in)
    #
    #
    ################
    ### loop numbers
    ################
    if fort is None:
        fort = range(0, len(thetas_pi))
    #
    if forp is None:
        forp = range(0, paras_in.shape[0])
    #
    if form is None:
        form = range(0, len(gset["model_indice"]))
    #
    #
    ################
    ### simulation
    ################
    break_flag = False
    for theta in fort:
        for l in forp:
            if any([tmp in [6] for tmp in gset["kill_num"]]):
                break
            #
            ### print a log
            mes = [
                "",
                "########====####",
                "loop for simulation parameter",
                "########====####",
                "%02d" % l
            ]
            print("\n".join(mes))
            #
            simm = Sim_manager(gset, tmkpr)
            #
            ### fluid model
            for m in form:
                ### print a log
                mes = [
                    "",
                    "################",
                    "loop for model indice",
                    "################",
                    "%02d" % m
                ]
                print("\n".join(mes))
                #
                ### name maker
                nm.set_sim_loop(paras_in.iloc[l, :].to_list(), [l, m, theta])
                #
                simm.main_sim(
                    sim_paras    = paras_in.iloc[l, :].to_list(),
                    model_choice = gset["model_indice"][m],
                    name_made    = nm.names_sim(),
                    i_theta      = theta,
                    focus        = gset.get("focus_draw_part")
                )
                #
                if 1 in gset["kill_num"]:
                    break_flag = True
                    break
                #
            if 2 in gset["kill_num"] or 1 in gset["kill_num"]:
                break_flag = True
                break
            #
        if break_flag:
        #if 1 <= theta:
            break
        #
    #
    #
    ################
    ### end all analyses
    ################
    print(tmkpr.get_elapsed())
    #
    #
    #


################
###
################
def main():
    v = sys.argv
    if "-m" in v:
        PARAS  = "./at142_sets/setting_tables/passed_test_200402.csv"
        num_in = [int(j) for j in v[(v.index('-m')+1):]]
        #
        ### MPI settings
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        #
        ### multi processes below
        forx = [num_in[rank]]
        #print(forx, end = ""); print(" in %d" % size) ### printed many at once
        #
    else:
        rank = 0
        if len(v) == 1:
            PARAS = "./at142_sets/setting_tables/passed_test_191213.csv"
            forx  = None
        else:
            ### for qsub
            PARAS = "./at142_sets/setting_tables/passed_test_200402.csv"
            forx  = [int(v[1])]
            #
    ########
    # indice j, k, l, t, p, m
    ########
    main_args(
        file_paras = PARAS,
        fort = forx,
        forp = [0],
        proc = rank
    )
    if not "-m" in v:
        import subprocess as sbp
        sbp.run(["bash", "tmp_ff_bash.txt"])
    #


################
###
################
if __name__ == '__main__':
    main()


###
