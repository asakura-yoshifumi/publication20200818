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


""" shared filenames over bash
tmp_ff_bash.txt
"""

def sub_global():
    d = {}
    ################
    ### choose which block to run now
    ################
    ### regression
    d["training_now"]    = False
    d["test_data_now"]   = False
    #
    #
    ### simulation
    d["extension"]       = False
    #
    d["Eul_ode_now"]     = True
    d["draw_euler_now"]  = True
    #
    d["particle_model"]  = True
    d["draw_particle"]   = True
    d["part_track"]      = True
    d["focus_draw_part"] = True
    #
    #
    d["read_parameters"] = True
    #
    ### during test, erace the all result regardless above
    d["force_erace"]     = False
    #
    d["markdown"]        = False
    #
    ### compare
    d["compare"]         = True
    d["Lag_ode_now"]     = False
    d["draw_lagrange"]   = True
    #
    #
    d["make_traj_mov"]      = False
    #
    #
    ### if not kill the run, set 0 here.
    ### otherwize, put flag number to kill the run there
    #
    # 1 simulation, both of parameters and models
    # 2 simulation, only parameter sets
    # 3 regression, all
    # 4 regression, breaks loop of j, methods to minimize
    # 5 regression, breaks loop of k, regression term sets
    # 6 simulation, all
    #
    d["kill_num"]        = [1, 4, 5] #[1, 4, 5] #[2, 3]
    #
    d["markdown_figs"]   = 3
    #
    #
    ################
    ### regression
    ################
    ###
    d["l_ridge"]    = 10.0
    d["conv"]       = 1.0e-4
    #
    ################
    ### simulation
    ################
    d["model_indice"] = [[0, 1, 2, 3, 4], [6, 7, 2, 3, 4]]
    #
    ### note in markdown, regarding model
    d["model_notes"]  = [
        "fliction(ERK) and pressure(dERK/dx)",
        #"fliction(dERK/dt) and pressure(dERK/dx)",
        "fliction(dERK/dt) and pressure(d2ERK/dt/dx)"
    ]
    #
    ### amplitude alteration
    # default [], or [dep. on speed (T/F, dERK/dt only), value(double)]
    d["amp_list"]  = [False, 0.32]
    ### meaninig of [1] above changed, set as default values if not given
    #
    ### initial value noise on density rho
    d["noisy"]     = False
    #
    ### indice of ["Vx", "Vy", "Rho", "ERK"] in this order
    d["choice_var"] = [0, 2, 3]
    ### animation resolution, "high", "mid", or "low"
    d["resolution"]  = "low"
    ### [".gif", ".mp4"], give its index in list
    d["format"]      = [0] #[0, 1]
    #
    ### parameters inside
    d["speed_grow"] = 0.01
    #
    # time
    d["dt"]       = 5.0 * (10.0 ** -1.0)
    d["tend"]     = 2000.0
    #
    d["dt_inner"] = 1.0 * (10.0 ** -1.0)
    #
    d["euler_points"] = 20 ### may not used?
    d["snap_points"]  = 20
    #
    ### paralell
    d["threads"]  = 2
    #
    d["slice_step"] = 2
    d["interval"]   = 5
    #
    ### courant from 0.0 to 1.0
    d["courant"] = 0.1
    #
    ### coord
    '''
    optimization is no longer needed below
    '''
    d["reci_dx"] = 1
    d["dx"]      = 1.0 / float(d["reci_dx"])
    if d["reci_dx"] < 1.0:
        d["xlim_draw"] = [int(d["dx"] * j) for j in d["xlim_draw"]]
        d["pixls_dx"]  =  int(d["dx"])
    else:
        d["pixls_dx"]  = 1 ### not to make the array too small
    '''
    optimization is no longer needed above
    '''
    #
    d["pixls_IMG"]  = 128#64
    d["pixls_IMGx"] = d["pixls_IMG"] * 1 * d["pixls_dx"]
    #
    d["theta_on_pi"] = 0.125
    d["startposi"]   = [32.0, 32.0]
    #
    d["center_particle"] = [64.0, 64.0]#[32.0, 32.0]
    d["radius"]          = 62.0#30.0
    #
    #
    ### reconstruction
    d["threshold_nei"]   = 8.0 * d["dx"]
    d["lam_lr"]          = 1.0e-3
    #
    #
    #
    #
    # modification
    d["true_x"]     = float(d["pixls_IMGx"])
    d["pixls_IMGx"] = int(d["pixls_IMGx"] * d["reci_dx"])
    #
    ### sight to truck as particles ### only 1D
    qu_x = int(d["true_x"] / 4)
    d["sightx"] = [qu_x, qu_x * 3]
    d["sighty"] = [0, d["pixls_IMG"]]
    #
    ### draw lagrange
    d["xlim_draw"] = None
    d["xlim_draw"] = [32, 224]
    d["ylim_draw"] = [40.0, 80.0]
    #d["ylim_draw"] = [400.0, 800.0]
    d["Lag_late"]  = True ### only 1D
    #
    d["x_x_span"]  = 0.08 ### only 1D
    #
    ### color selection for markdown, put [] if no need of redblue
    # "Vx", "heat", "part"
    d["redblue"]   = ["Vx"]
    #
    ### particle model
    d["draw_skip"] = 10   ### only 1D
    #
    #
    ### compare
    d["init_range"] = [100, 164] ### only 1D
    #
    #
    ### limits to draw in 2D
    d["lim_2d_draw"] = [42, 86] ### cut 42 from both side
    d["new_O_x_t"]   = [0.0, 64.0]
    #
    #
    ### making sure that ylim is in simulation timerange
    if all([tmp <= d["tend"] for tmp in d["ylim_draw"]]):
        pass
    else:
        d["ylim_draw"] = [0.0, d["tend"]]
    #
    ### model decription
    df = d["model_indice"]
    df2 = [", ".join(["% 2d" % j for j in tmp]) for tmp in df]
    df2 = [
        "material term indice in Dv/Dt\n",
        "  #: description",
        "  0: fliction(ERK)",
        "  1: pressure(ERK)",
        "  2: pressure(rho)",
        "  3: viscosity",
        "  4: viscosity",
        "  5: fliction(dERK/dx)",
        "  6: fliction(dERK/dt)",
        "  7: pressure(dERK/dt)",
        "selected models are;"
    ] + ["m%02d: %s" % (j, df2[j]) for j in range(0, len(df2))]
    df2 = "\n    ".join(df2)
    #
    d["model_desc"] = df2 ### this is a sting
    #
    #
    d["th_draw"]       = 0.0 ### draw x_t graph in this theta
    d["man_snap_time"] = list(range(400, 1600, 16))
    d["square_draw_p"] = [720]
    #
    #
    ### auto adjustment
    d["man_snap_time"] = [l for l in d["man_snap_time"] if l < d["tend"]]
    d["square_draw_p"] = [min(l, int(d["tend"] / d["dt"]) - 1) for l in d["square_draw_p"]]
    #
    #
    return(d)
    #
    #



def main():
    import sys
    import os
    argv = sys.argv
    gset = sub_global()
    #
    targ = []
    for j in argv[1:]:
        if os.path.isdir(j):
            targ += [ "%s/%s" % (j, l) for l in os.listdir(j) if ".py" in l ]
        elif os.path.isfile(j) and ".py" in j:
            targ.append(j)
    #
    for k in gset.keys():
        print(">>> %s" % k)
        for j in targ:
            if os.path.isdir(j):
                continue
            with open(j, mode = "r") as f:
                lines = f.read()
            if k in lines:
                print(j)
        print("")

if __name__ == '__main__':
    main()


###
