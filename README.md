# Codes for Asakura et al. 2021
The full code will be uploaded after the publication.


## 1. Citation
The citation information will be updated after the publication.

## 2. License
Copyright (c) 2021 Yoshifumi Asakura

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


## 3. Tested environment

### OS and C compiler
The programs here were tested under the systems below;
- **MacOS Catalina 10.15.7** and **clang 12.0.0**
- **CentOS Linux 7.6.1810** and **gcc 4.8.5**

Note that codes here depend on a multithreading library OpenMP (https://www.openmp.org/) for speed.
The default C compiler in MacOS, clang, do not contain OpenMP runtime library when the computer was shipped.
The easiest way to install it is by homebrew (https://brew.sh/). After instllation of homebrew,
type below in your shell for OpenMP instllation;
```bash
brew install libomp
```
GCC (https://gcc.gnu.org/) does not need further installation for OpenMP, since it has the OpenMP runtime library within it.

### python 3.7.2

    cycler==0.10.0
    Cython==0.29.13
    h5py==2.9.0
    imagecodecs==2020.2.18
    joblib==0.13.2
    kiwisolver==1.1.0
    llvmlite==0.31.0
    matplotlib==3.1.0
    numba==0.48.0
    numpy==1.16.3
    opencv-python==4.1.0.25
    pandas==0.24.2
    Pillow==6.1.0
    pyparsing==2.4.1.1
    PypeR==1.1.2
    python-dateutil==2.8.0
    pytz==2019.1
    scikit-learn==0.21.2
    scipy==1.2.1
    seaborn==0.9.0
    six==1.12.0
    tifffile==2020.2.16

to install them at once, type below on your shell in a directory you cloned this repository;

``` bash
pip install -r ./list_python_packages.txt
```

The file **list_python_packages.txt** located here is the same list as above.

If you would not like to overwrite your python environment,

- pyenv (https://github.com/pyenv/pyenv)
- pyenv-virtualenv (https://github.com/pyenv/pyenv-virtualenv)

will help you to separate the environments without sudo permission.


### R 3.6.3

    tidyverse 1.3.0
    gganimate 1.0.5
    ggthemes  4.2.0

to install them, type below on your R;

``` R
install.packages("tidyverse", version = "1.3.0")
install.packages("gganimate", version = "1.0.5")
install.packages("ggthemes",  version = "4.2.0")
```

### C++ packages dependency
The programs here depends on the C++ packages below.
Please download the files on the URLs below to "download" directory here.
- "Qhull"
https://github.com/qhull/qhull/archive/v7.3.2.tar.gz
- "BOOST"
https://dl.bintray.com/boostorg/release/1.74.0/source/boost_1_74_0.tar.gz


## 4. Contents and elapsed time
As scripts for simulation, "asXXX" python files are located on the top directory here.

At a directory "atXXX" with corresponding number, you can find assistive sub functions.

- **run_setup.sh**
  - run c, c++, and cython setup at once
  - it searches in download directory here for necessary c++ packages
- **as126.py**
  - one dimensional simulation
- **as142.py**
  - two dimensional simulation
  - it took 92 hours
  - about 80 hours were for the particle-based model simulation
  - it depends on a c++ package "Qhull" (http://www.qhull.org/)
- **as200.py**
  - asymmetric ERK waves simulation
  - it depends on a c++ package "boost" (https://www.boost.org/)
- **as210.py**
  - simulations with some grid sizes
  - it depends on a c++ package "boost" (https://www.boost.org/)

The other scripts than two dimensional "as142", each script took less than 1 hour with an environment variable on shell OMP_NUM_THREADS as 8.

Only two dimensional "as142" was tested only on CentOS.

The other codes were tested both on MacOS and CentOS.
