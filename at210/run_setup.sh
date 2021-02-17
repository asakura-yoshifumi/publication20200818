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
ddir="../download"

### for boost
bdir="boost_1_74_0"
boost_tar_gz_path="${ddir}/${bdir}.tar.gz"

if ! [ -f $boost_tar_gz_path ]; then
  echo ">>> ERROR <<<"
  echo "prepare the tar gz file below in the 'download' directory here;"
  echo  "https://dl.bintray.com/boostorg/release/1.74.0/source/boost_1_74_0.tar.gz"
fi

pre="${here}/cpp_boost"
if ! [ -d ${pre}/boost ]; then
  echo "compile boost frst"
  tar zxf $boost_tar_gz_path
  cd $bdir

  ### only math header
  ./bootstrap.sh --with-libraries=math --prefix=$pre
  ./b2 headers
  rsync -au boost $pre/

  cd $here
fi

###

mv ./__init__.py ./tmp__init__.py

python setup.py build_ext --inplace

mv ./tmp__init__.py ./__init__.py


###
