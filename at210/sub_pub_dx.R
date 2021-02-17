#!/usr/local/bin/Rscript
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



library(tidyverse)
library(ggthemes)

################
### load data
################
l_full <- __rdata__
l      <- sapply(l_full, function(x){str_replace(x, ".*/", "")})
legend <- c(T, F, F, F)


fls <- data.frame(
  file  = l,
  path  = l_full,
  label = sapply(l, function(x){return(str_split(x, "\\.")[[1]][1])})
)
rownames(fls) <- NULL


outdir <- "__result__"
#print(outdir %in% list.files("./"))
if(! outdir %in% list.files("./")){
  dir.create(outdir); #q("no")
}
exts <- c(
  "png",
  #"eps",
  #"tex",
  #"svg",
  "pdf"
)

for(j in 1:length(fls[[1]])){
  header <- str_c(as.character(fls$label[j]), "_")
  load(as.character(fls$path[j]))
  for(ext in exts){
    dir <- str_c(outdir, "/", ext)
    if(! ext %in% list.files(outdir)){
      dir.create(dir)
    }
    header2 <- str_c(dir, "/", header)
    if(legend[j]){
      g010    <- t000 + theme(
        title      = element_text(size = 16),
        legend.text= element_text(size = 14),
        legend.position      = c(0.05, 0.95),
        legend.justification = c(0, 1),
        axis.text  = element_text(size = 11, color = "black"),
        axis.title = element_text(size = 16)
      )
    } else {
      g010    <- t000 + theme(
        title      = element_text(size = 16),
        legend.text= element_text(size = 14),
        legend.position      = "none",
        axis.text  = element_text(size = 11, color = "black"),
        axis.title = element_text(size = 16)
      )
    }
    g010 <- g010 + guides(
      color = guide_legend(title = "wave length (sd)")
    )
    ggsave(
      plot = g010,
      file = str_c(header2, "sup04_dx.", ext),
      height =  80,
      width  = 160,
      unit   = "mm"
    )
  }
}
