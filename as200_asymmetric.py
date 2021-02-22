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

### handwrite
from at200 import *


################
###
################
def main_sim(gset, nm, tmkpr, data_arrange):
    ### inherit
    paras, model_ind, l, m = nm.get_in_sim()
    #
    name_made        = nm.names_sim(gset["redblue"])
    data_arrange.add_row(name_made)
    dir_reset(name_made["integ_place"], False)
    '''
    l   simulation parameter
    m   indice in model
    '''
    sim_paras        = paras.iloc[l, :].tolist()
    sim_mod_ind      = model_ind[m]
    #
    ### output
    out_sim          = {}
    #
    ################
    ### Euler descibed, tissue scale
    ################
    #if gset["run_ode_now"]:
    ### initial condition, added is to add to ordinary state that can be all 0
    added = np.zeros((3, gset["pixls_IMGx"], gset["pixls_IMG"]))
    if gset["add_init"]:
        choice = 0
        if   choice == 0:
            tmp   = 2.0 * math.pi * np.arange(0, gset["pixls_IMGx"], dtype = np.float64).reshape((gset["pixls_IMGx"], gset["pixls_IMG"])) / float(gset["pixls_IMGx"])
            added[0, :, :] = np.sin(tmp) * 0.8
        elif choice == 1:
            added[0, 64:128, :] += 1.0
        elif choice == 2:
            added[0, :, :] += 0.2
        #
    #
    ### amplitude, used in Euler (fluid model), Particle (P model)
    # new 190824_2000
    amp_dt = math.exp(0.5) * sim_paras[8] * sim_paras[10]
    if not gset["amp_list"][0]:
        ### 0 th in the list is T/F of dependency on ERK wave speed
        amp_dt = amp_dt / sim_paras[7]
        #
    #
    if gset["run_ode_now"] and not all([os.path.exists(name_made[l]) for l in ["integ_value", "integ_time", "integ_coord", "source"]]):
        #
        ### prepare simulation
        sim_eu = Diff_vrho(gset["dt"], gset["dx"], gset["pixls_IMGx"], gset["pixls_IMG"], sim_paras, gset["tend"], gset["courant"], gset["noisy"], add_init = added, model_indice = sim_mod_ind, amp_dt = amp_dt)
        #
        ### run simulation
        sim_eu.run_sim()
        #
        #
        ### draw field values as graphs
        if gset["draw_now"]:
            fie_eu = Time_values(sim_eu.get_times(), sim_eu.get_fields(), sim_paras, gset["dx"], sim_eu.get_coord())
            fie_eu.draw_heat(name_made["heat_name"], Vxname = name_made["Vxname"], Vyname = name_made["Vyname"], Rhoname = name_made["Rhoname"], ERKname = name_made["ERKname"], choice_var = gset["choice_var"], Rdata = name_made["Rdata"], another_color = name_made["another_color"])
            #
            ### erk
            time_last = np.max(sim_eu.get_times())
            draw_erks(name_made["erks_name"], sim_eu.get_ERK_3models(time_last), sim_eu.get_coord(), name_made["erk_shape"])
            #
        #
        ### save the integrated
        #dir_reset(name_made["integ_place"], False)
        np.save(file = name_made["integ_value"], arr = sim_eu.get_integrated())
        np.save(file = name_made["integ_time"],  arr = sim_eu.get_times())
        np.save(file = name_made["integ_coord"], arr = sim_eu.get_coord())
        np.save(file = name_made["source"],      arr = sim_eu.get_source())
        ### fields
        nm.save_fields(sim_eu.get_fields())
        #
        ### to pass
        tmp_integrated = sim_eu.get_integrated(); #print(tmp_integrated.shape)
        tmp_coord      = sim_eu.get_coord()
        tmp_timerange  = sim_eu.get_times()
        tmp_sim_shape  = sim_eu.get_integrated().shape[0]
        #
        ### to pass to Lagrange simulation
        pass_field     = sim_eu.get_fields()
        pass_times     = sim_eu.get_times()
        pass_source    = sim_eu.get_source()
        #
    else:
        ### load the integted
        tmp_integrated = np.load(name_made["integ_value"])
        tmp_coord      = np.load(name_made["integ_coord"])
        tmp_timerange  = np.load(name_made["integ_time"])
        tmp_sim_shape  = tmp_integrated.shape[0]
        #
        ### to pass to Lagrange simulation
        print("just initialize Diff_vrho, but not to run simulation...")
        pass_field  = nm.load_fields()
        pass_times  = tmp_timerange
        pass_source = np.load(name_made["source"])
        print("loaded.")
        #
    #
    #
    ################
    ### run animation ro not
    ################
    if not any([os.path.exists(name_made[l]) for l in ["animation", "ani_capture"]]) and (gset["force_animation"] or gset["lastframe_only"] or (gset["run_ode_now"] and gset["run_animation"])):
        tmp = tmp_sim_shape
        slice_step = gset["slice_step"]
        ar = Ar(tmp_integrated[::slice_step, :], tmp_coord, tmp_timerange[::slice_step, :], tmp, gset["interval"], gset["repeat_ani"])
        moviename = name_made["animation"]
        if gset["lastframe_only"]:
            moviename = None
        ar.drawR(filename = moviename, res_choice = gset["resolution"], format = gset["format"], cap_name = name_made["ani_capture"])
        print("animation ended at\n    %s" % tmkpr.get_elapsed())
        #
    #
    #
    ################
    ### Lagrange described, cell scale, material time derivertive
    ################
    ### preparation for Lagrange, Pmodel and comparison
    if gset["Lag_late"]:
        ### start late or not
        stind = np.argmin(np.abs(pass_times[:, 0] - gset["ylim_draw"][0]))
        stend = np.argmin(np.abs(pass_times[:, 0] - gset["ylim_draw"][1]))
        if pass_times.shape[0] * 0.75 < stind:
            stind = 0
        pass_times_short = pass_times[stind:, :]
        pass_field_short = [tmp[stind:, :] for tmp in pass_field]
        #print(pass_times.reshape(-1))
        #
    #
    if (gset["run_ode_now"] or gset["force_lag_now"]) and gset["permit_lagrange"] and not gset["Lag_in_comp"]:
        ode_startx = gset["sightx"][0]
        ode_gapx   = gset["sightx"][1] - ode_startx
        iniy       = gset["pixls_IMG"] * 0
        init = np.array([[ode_startx + ode_gapx * gaps , iniy, 0.0, 0.0] for gaps in np.arange(0.1, 1.0, gset["x_x_span"])])
        #
        pe = Pon_Euler(init, fields = pass_field_short, ERK = pass_field_short[3], timerange = pass_times_short, dt = gset["dt"], dx = gset["dx"])
        pe.run_sim()
        pe.draw_track(name_made["part_name"], csvname = None, draw_range = gset["xlim_draw"], draw_time_range = gset["ylim_draw"], Rdata = name_made["Lag_Rdata"])
        #
        #
    #
    #
    ################
    ### particlemodel
    ################
    if gset["particle_model"] and not all([os.path.exists(name_made[l]) for l in ["pmodel_Rdata"]]):
        coord_max = np.max(tmp_coord[0, :, :])
        cell_init = np.zeros((int(coord_max), 2))
        cell_init[:, 0] = np.arange(0, coord_max, dtype = np.float64)
        sim_sp = Sim_Part(pass_times, cell_init = cell_init, paras = sim_paras, model_choice = sim_mod_ind, amp_dt = amp_dt)
        #
        ### run
        sim_sp.run_sim()
        #
        ### draw
        pmd = Particle_model_draw(sim_sp.get_timerange(), sim_sp.get_xve(), paras = sim_paras, draw_skip = gset["draw_skip"], draw_timerange = gset["ylim_draw"])
        pmd.draw(name_made["pmodel_track"], Rdata = name_made["pmodel_Rdata"], another_color = name_made["pmodel_color"], draw_range = gset["xlim_draw"])
        #pmd.draw(name_made["pmodel_track"], Rdata = name_made["pmodel_Rdata"], another_color = name_made["pmodel_color"], draw_range = [140.0, 150.0])
        #pmd.draw(name_made["pmodel_track"], Rdata = name_made["pmodel_Rdata"], another_color = name_made["pmodel_color"])
        #
        #
    #
    #
    ################
    ### comparison, particle model and fluid model Lagrange particles
    ################
    if gset["compare"] and not all([os.path.exists(name_made[l]) for l in ["Lag_Rdata"]]):
        '''
        depends on
            pass_field (Euler simulation)
            pass_times (Euler simulation)
            Rdata saved via... Rdata = name_made["pmodel_Rdata"] (Pmodel)
        '''
        if os.path.exists(name_made["pmodel_Rdata"]):
            comp = Comparison(name_made["pmodel_Rdata"], gset["init_range"])
            init = comp.get_init()
            #
            pe = Pon_Euler(init, fields = pass_field_short, ERK = pass_field_short[3], timerange = pass_times_short, dt = gset["dt"], dx = gset["dx"])
            pe.run_sim()
            #
            if gset["Lag_in_comp"]:
                df = pe.draw_track(name_made["part_name"], csvname = None, draw_range = gset["xlim_draw"], draw_time_range = gset["ylim_draw"], Rdata = name_made["Lag_Rdata"])
            else:
                df = pe.get_df(draw_time_range = gset["ylim_draw"])
            #
            if gset["comp_only_new"]:
                comp.compare(df, graphname = name_made["comp_all"])
            else:
                comp.compare(df, Rdata = name_made["comp_R"], gifname = name_made["comp_gif"], graphname = name_made["comp_all"], draw_xrange = gset["xlim_draw"])
            #
            #
        else:
            print("no Rdata named %s" % name_made["pmodel_Rdata"])
        #
    #
    #
    return(out_sim)
    #
    #


################
###
################
def main():
    ################
    ### timer
    ################
    tmkpr = Time_keeper()
    #
    #
    ################
    ### general settings
    ################
    gset      = sub_global()
    outdirs   = ["result200"]
    outheads  = ["result200"]
    ext       = "png"
    #
    option_TF = gset["force_erace"] or (gset["training_now"] and gset["test_data_now"])
    dir_reset(outdirs[0], option_TF); #sys.exit()
    #
    #
    ################
    ### for regression
    ################
    #
    #
    ################
    ### parameters for simulation, some may passed from regression
    ################
    ### to store parameters passed from regression
    paras_store_reg = None
    #
    #
    ### for simulation
    file_paras  = "./at200/param.csv"
    force_paras = None
    if gset["read_parameters"]:
        force_paras = pd.read_csv(file_paras)
        if force_paras.get("amp_dt") is None:
            force_paras["amp_dt"] = gset["amp_list"][1]
        ### prepare force_num
        #
    #
    #
    ################
    ### names maker
    ################
    nm = Name_Maker(outdirs[0], outheads[0], ext, sim_com = gset["model_notes"])
    #
    da = Data_Arrange(outdirs[0])
    #
    ################
    ### loops
    ################
    #
    #
    ### parameter preparation for simulation
    #if gset["run_ode_now"]:
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
    forl = range(0, paras_in.shape[0])
    #forl = [6, 10]
    #forl = [0]
    #
    form = range(0, len(gset["model_indice"]))
    #form = range(0, 3)
    #form = [0]
    #
    sim_sum = [["yet" for j in range(0, len(gset["model_indice"]))] for k in range(0, paras_in.shape[0])]
    ind_sim = ["para%02d" % tmp for tmp in range(0, paras_in.shape[0])]
    col_sim = ["mode%02d" % tmp for tmp in range(0, len(gset["model_indice"]))]
    #
    #
    ################
    ### simulation
    ################
    for l in forl:
        if any([tmp in [6] for tmp in gset["kill_num"]]):
            break
        #
        mes = [
            "",
            "########====####",
            "loop for simulation parameter",
            "########====####",
            "%02d" % l
        ]
        print("\n".join(mes))
        #
        for m in form:
            mes = [
                "",
                "################",
                "loop for model indice",
                "################",
                "%02d" % m
            ]
            print("\n".join(mes))
            #
            ### names maker
            nm.set_sim(paras_in, gset["model_indice"], [l, m])
            #
            ### run
            try:
                out_sim = main_sim(gset, nm, tmkpr, da)
                sim_sum[l][m] = "success"
                #
            except:
                print(tb.print_exc())
                sim_sum[l][m] = "failure"
                #
            #
            pd.DataFrame(sim_sum, index = ind_sim, columns = col_sim).to_csv(nm.get_sim_sum())
            #
            if any([tmp in [1] for tmp in gset["kill_num"]]):
                break
            #
            #
        #
        if any([tmp in [1, 2] for tmp in gset["kill_num"]]):
            break
            #
        #
    #
    #
    da.save_Rdata_paths()
    da.draw_pub_fig()
    ################
    ### all parts finished
    ################
    print("all done")
    #
    ### time elapsed
    print(tmkpr.get_elapsed())
    #
    #
    #


################
###
################
def test():
    pass


################
###
################
if __name__ == '__main__':
    main()

###
