
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

from distutils.core import setup, Extension
from Cython.Build import cythonize

import numpy
import scipy

from time import sleep
import os

import platform

if 'Darwin' in platform.platform():
    extra_comp = ["-Xpreprocessor", "-fopenmp"]
    extra_link = ["-Xpreprocessor", "-fopenmp", "-lomp"]
else:
    extra_comp = ["-fopenmp"]
    extra_link = ["-fopenmp"]
here = os.path.dirname(os.path.abspath(__file__))


''' ----------------------------
qhull for simulation of particle in 2D
---------------------------- '''
### add qhull .c files
pathdir = "c_qhull/libqhull_r/"
extent  = ".c"

inpath  = os.listdir(pathdir)
sous    = []
for j in inpath:
    if extent in j:
        toadd = pathdir + j
        sous.append(toadd)

ext = Extension(
    "cfun_part2",
    sources = ["cfun_part2.pyx", "c_qhull/c_voronoi.c", "c_erk_share.c"] + sous,
    extra_compile_args = extra_comp + ["-I %s/c_qhull" % here],
    extra_link_args    = extra_link
)
setup(
    ext_modules = cythonize([ext],  compiler_directives={'language_level' : "3"}),
    include_dirs = [numpy.get_include()]
)

sleep(0.5)

''' ----------------------------
simulation fluid 2D
---------------------------- '''
ext = Extension(
    "cfun_weno2",
    sources = ["cfun_weno2.pyx", "c_weno1d.c", "c_erk_share.c"],
    extra_compile_args = extra_comp,
    extra_link_args    = extra_link
)
setup(
    ext_modules = cythonize([ext],  compiler_directives={'language_level' : "3"}),
    include_dirs = [numpy.get_include()]
)
