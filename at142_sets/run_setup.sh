#!/bin/sh
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


### automatically below
here=$PWD

ddir="${here}/../download"
qhull_tar_gz="${ddir}/v7.3.2.tar.gz"
qdir="qhull-7.3.2"

if ! [ -f $qhull_tar_gz ]; then
  echo ">>> ERROR <<<"
  echo "prepare the tar gz file below in the 'download' directory here;"
  echo  "https://github.com/qhull/qhull/archive/v7.3.2.tar.gz"
fi


if ! [ -d ./c_qhull/libqhull_r ]; then
  ### copy already downloaded or download above url
  cd c_qhull
  tar zxf $qhull_tar_gz ${qdir}/src/libqhull_r
  mv ${qdir}/src/libqhull_r ./
  rm -rf $qdir
  cd $here
fi



### setup.py running
mv ./__init__.py ./tmp__init__.py

python setup.py build_ext --inplace

mv ./tmp__init__.py ./__init__.py


###
