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
import numpy     as np
import math
import sys
import os
import shutil

import pandas    as pd
import traceback as tb

import subprocess as sbp

### handwrite
from .fun_dir      import dir_reset, find_up, Time_keeper, Name_Maker, Data_Arrange, di1
from .global_set   import sub_global

from .fun_sim      import Pon_Euler

from .cfun_weno2   import Diff_vrho_2D as Diff_vrho

from .cfun_part2   import Two_Part
from .fun_part2    import init_hex_close, init_hex_circle, Draw_Part2

from .fun_fluid2d  import draw_2D_fluid, build_tuples, draw_one_2D, draw_den_p2D
from .fun_comp2    import Compare2D

from .fun_ext_sim  import name_extender, find_max_ext


################
### simulation
################
class Sim_manager:
    def __init__(
        self,
        gset,
        tmkpr
    ):
        self.gset         = gset
        self.tmkpr        = tmkpr
        #
        self.yet_particle = True
        #
        #
    def main_sim(self, sim_paras, model_choice, name_made, i_theta = 1.0, focus = None):
        out = {}
        gset         = self.gset
        tmkpr        = self.tmkpr
        #
        # amplitude
        amp_dt = math.exp(0.5) * sim_paras[8] * sim_paras[10]
        if not gset["amp_list"][0]:
            ### 0 th in the list is T/F of dependency on ERK wave speed
            amp_dt = amp_dt / sim_paras[7]
            #
        #
        man_snap_time = gset.get("man_snap_time")
        if focus is None:
            focus_area = gset.get("lim_2d_draw")
        else:
            if i_theta == 1:
                focus_area = [64, 86, 42, 64]
            elif i_theta == 2:
                focus_area = [53, 75]
            elif i_theta == 3:
                focus_area = [42, 64, 64, 86]
            else:
                focus_area = gset.get("lim_2d_draw")
        #
        ################
        ### fluid model
        ################
        added  = np.zeros((3, gset["pixls_IMGx"], gset["pixls_IMG"]))
        t_init = 0.0
        #
        ### euler
        sim_eu = Diff_vrho(
            gset["dt"],
            gset["dx"],
            gset["pixls_IMGx"],
            gset["pixls_IMG"],
            sim_paras,
            gset["tend"],
            gset["courant"],
            gset["noisy"],
            add_init     = added,
            t_init       = t_init,
            model_indice = model_choice,
            amp_dt       = amp_dt,
            startposi    = gset["startposi"],
            theta_on_pi  = gset["theta_on_pi"] * i_theta,
            speed_grow   = gset["speed_grow"]
        )
        erk_c_list = sim_eu.get_in_erk_share()
        #
        if gset["Eul_ode_now"]:
            sim_eu.run_sim()
            print("simulation ended at %s" % tmkpr.get_elapsed())
            #
            np.save(file = name_made["keep_euler_integ"], arr = sim_eu.get_integrated())
            np.save(file = name_made["keep_euler_time"],  arr = sim_eu.get_times())
            np.save(file = name_made["keep_euler_coord"], arr = sim_eu.get_coord())
            np.save(file = name_made["keep_euler_erk"],   arr = sim_eu.get_ERK_blocked())
            np.save(file = name_made["keep_euler_field"], arr = sim_eu.get_fields())
            #
            keep_euler_integ = sim_eu.get_integrated()
            keep_euler_time  = sim_eu.get_times()
            keep_euler_coord = sim_eu.get_coord()
            keep_euler_erk   = sim_eu.get_ERK_blocked()
            #
            keep_euler_field = sim_eu.get_fields()
            keep_euler_shape = sim_eu.get_integrated().shape[0]
            #
        else:
            keep_euler_integ = np.load(name_made["keep_euler_integ"])
            keep_euler_time  = np.load(name_made["keep_euler_time"])
            keep_euler_coord = np.load(name_made["keep_euler_coord"])
            keep_euler_erk   = np.load(name_made["keep_euler_erk"])
            #
            keep_euler_field = np.load(name_made["keep_euler_field"])
            keep_euler_shape = keep_euler_integ.shape[0]
        #
        if gset["draw_euler_now"]:
            draw_2D_fluid(
                dir     = name_made["graphdir_snapshot_euler"],
                arr     = keep_euler_integ,
                coord   = keep_euler_coord,
                times   = keep_euler_time,
                points  = 0,
                man_points = gset.get("man_snap_time"),
                movfile = name_made["graph_mov_euler"],
                ERK     = keep_euler_field[:, 3, :, :],
                lim2d   = focus_area,
                rect_time = gset.get("square_draw_p"),
                rect_tups = build_tuples(gset["new_O_x_t"], gset.get("th_draw"), focus_area)
            )
        #
        del sim_eu
        #
        ##################
        ### particle model
        ##################
        cell_init = init_hex_circle(gset["center_particle"], gset["radius"])
        coord_max = float(gset["pixls_IMG"]) * gset["dx"]
        #
        #
        #
        ### run simulation
        '''
        particle simulation is only needed to run one time,
        that is less than fluid model simulations
        '''
        if self.yet_particle:
            self.yet_particle = False
            if gset["particle_model"]:
                sim_sp = Two_Part(
                    keep_euler_time,
                    cell_init    = cell_init,
                    paras        = sim_paras,
                    model_choice = model_choice,
                    amp_dt       = amp_dt,
                    dt_inner     = gset["dt_inner"],
                    startposi    = gset["startposi"],
                    theta_on_pi  = gset["theta_on_pi"] * i_theta,
                    speed_grow   = gset["speed_grow"],
                    vor_span     = 8
                )
                sim_sp.run_sim_c()
                #
                np.save(file = name_made["keep_particle_integ"], arr = sim_sp.get_integrated())
                np.save(file = name_made["keep_particle_time"],  arr = sim_sp.get_timerange())
                np.save(file = name_made["keep_particle_erk"],   arr = sim_sp.get_ERK_on())
                print("saved keep files")
                print(tmkpr.get_elapsed())
                #return(0)
                #
                keep_particle_integ = sim_sp.get_integrated()
                keep_particle_time  = sim_sp.get_timerange()
                keep_particle_erk   = sim_sp.get_ERK_on()
                #
                del sim_sp
                #
            else:
                keep_particle_integ = np.load(name_made["keep_particle_integ"])
                keep_particle_time  = np.load(name_made["keep_particle_time"])
                keep_particle_erk   = np.load(name_made["keep_particle_erk"])
                #
            #
        else:
            ### self.yet_particle is False, i.e. done,
            ### therefore there are keep_particle... objects
            pass
            #
        out["keep_particle_integ"] = keep_particle_integ
        out["keep_particle_time"]  = keep_particle_time
        out["keep_particle_erk"]   = keep_particle_erk
        #
        ### drawing
        if gset["particle_model"] or gset["draw_particle"]:
            ### draw
            print("draw 2D particles")
            pmd = Draw_Part2(
                keep_particle_integ,
                keep_particle_time,
                keep_particle_erk
            )
            #print(" >> %f" % np.mean(keep_particle_integ[:, :, 1])); #sys.exit()
            if gset["part_track"]:
                e_st = np.array(gset["startposi"]).reshape((2, 1))
                e_th = gset["theta_on_pi"] * i_theta * np.pi
            else:
                e_st = None
                e_th = None
            pmd.draw(
                filename = name_made["graph_mov_part2d"],
                Rdata    = name_made["keep_particle_R"],
                xylims   = focus_area,
                snap_dir = name_made["graphdir_snapshot_particle"],
                points   = gset["snap_points"],
                erk_start = e_st,
                erk_theta = e_th,
                track_color   = "red",
                man_snap_time = gset.get("man_snap_time"),
                make_traj_mov = gset.get("make_traj_mov")
            )
            pmd.x_t(
                filename     = name_made["graph_x_t_part2d"],
                list_for_erk = erk_c_list,
                xlim_draw    = gset["lim_2d_draw"],
                theta        = gset.get("th_draw"),
                new_O        = gset["new_O_x_t"]
            )
            #
        #
        #
        ################
        ### compare
        ################
        ### use in elif
        in_elif = [
            name_made["keep_lagrange_integ"],
            name_made["keep_lagrange_time"],
            name_made["keep_lagrange_erk"]
        ]
        if gset["compare"] or gset["Lag_ode_now"]:
            ### run Lagrange described simulation
            ### based on Euler-described results
            pe = Pon_Euler(
                cell_init, ### see particle model part
                keep_euler_field,
                keep_euler_time,
                gset["dx"]
            )
            pe.run_sim()
            print(tmkpr.get_elapsed())
            #
            keep_lagrange_integ = pe.get_integ()
            keep_lagrange_time  = pe.get_time()
            keep_lagrange_erk   = pe.get_erk()
            np.save(
                file = name_made["keep_lagrange_integ"],
                arr  =            keep_lagrange_integ
            )
            np.save(
                file = name_made["keep_lagrange_time"],
                arr  =            keep_lagrange_time
            )
            np.save(
                file = name_made["keep_lagrange_erk"],
                arr  =            keep_lagrange_erk
            )
            del pe
        elif all(os.path.exists(tmp) for tmp in in_elif):
            keep_lagrange_integ = np.load(name_made["keep_lagrange_integ"])
            keep_lagrange_time  = np.load(name_made["keep_lagrange_time"])
            keep_lagrange_erk   = np.load(name_made["keep_lagrange_erk"])
        if gset["draw_lagrange"]:
            ################
            ### drawing
            ################
            fld = Draw_Part2(
                keep_lagrange_integ,
                keep_lagrange_time,
                keep_lagrange_erk,
                lagrange = True
            )
            ### again for a case if particle_draw above is False
            if gset["part_track"]:
                e_st = np.array(gset["startposi"]).reshape((2, 1))
                e_th = gset["theta_on_pi"] * i_theta * np.pi
            else:
                e_st = None
                e_th = None
            fld.draw(
                filename = name_made["graph_mov_lagrange"],
                Rdata    = name_made["keep_lagrange_R"],
                xylims   = focus_area,
                snap_dir = name_made["graphdir_snapshot_lagrange"],
                points   = gset["snap_points"],
                erk_start = e_st,
                erk_theta = e_th,
                track_color   = "red",
                man_snap_time = gset.get("man_snap_time"),
                make_traj_mov = gset.get("make_traj_mov")
            )
            fld.x_t(
                filename     = name_made["graph_x_t_lagrange"],
                list_for_erk = erk_c_list,
                xlim_draw    = gset["lim_2d_draw"],
                theta        = gset.get("th_draw"),
                new_O        = gset["new_O_x_t"]
            )
        #
        return(out)
        #
        #


###
