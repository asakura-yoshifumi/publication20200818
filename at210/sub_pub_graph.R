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



################################################################
#
# this file is to run an adjustment of figures for publication
# regarding below
#     font size
#     font color
#     figure size
#     file format
#
################################################################

library(tidyverse)

################
### load data
################
path <- "__result__/graphs_Rdata/"
l <- list.files(path)
#print(l); q("no")

fls <- data.frame(
  file  = l,
  path  = sapply(l, function(x){return(str_c(path, x))}),
  label = sapply(l, function(x){return(str_split(x, "\\.")[[1]][1])})
)
rownames(fls) <- NULL
#print(sapply(l, function(x){return(str_split(x, "\\.")[[1]][1])}))
#print(fls); q("no")


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


################
### euler
################
euler <- function(e000){
  out <- e000 + theme_classic(
  ) + scale_fill_viridis_c(
  ) + theme(
    axis.text  = element_text(size = 14, color = "black"),
    axis.title = element_text(size = 16)
  ) + labs(
    x = "space"
  )
  return(out)
}

base_tile <- function(df_eul, key, labeller = Null){
  df <- subset(df_eul, factors == key)
  df <- subset(df, time >= 1000)
  df$time <- df$time - 1000
  out <- ggplot(df, aes(x, time)) + geom_tile(
    aes(fill = value)
  ) + theme_classic(
  #) + scale_fill_viridis_c(
  #) + scale_fill_gradient2(
  #  low      = "blue",
  #  high     = "red",
  #  mid      = "white",
  #  midpoint = 0.0
  ) + scale_y_reverse(
  ) + theme(
    strip.text = element_text(size = 20),
    axis.text  = element_text(size = 16, color = "black"),
    axis.title = element_text(size = 20),
    legend.title = element_text(size = 16),
    legend.text  = element_text(size = 16)
  )
  if(is.null(labeller)){
    out <- out + facet_grid(
      . ~ factors
    )
  } else {
    out <- out + facet_grid(
      . ~ factors,
      labeller = as_labeller(labeller)
    )
  }
  out <- out + labs(
    x = "space"
  #) + coord_cartesian(
  #  ylim = c(1000, 2000)
)
  return(out)
}

#velorho <- function(x){
#  if(x == "Vx"){
#    return("velocity")
#  } else if(x == "Rho"){
#    return("density")
#  } else {
#    return(x)
#  }
#}

labels = c("Vx" = "velocity", "Rho" = "denisty", "ERK" = "ERK")

Vx <- function(df_eul){
  out <- base_tile(df_eul, 'Vx', labeller = labels) + scale_fill_gradient2(
    low      = "blue",
    high     = "red",
    mid      = "white",
    midpoint = 0.0
  )
  return(out)
}

Rho <- function(df_eul){
  out <- base_tile(df_eul, 'Rho', labeller = labels) + scale_fill_viridis_c(
  )
  return(out)
}

ERK <- function(df_eul){
  out <- base_tile(df_eul, 'ERK', labeller = labels) + scale_fill_viridis_c(
  )
  return(out)
}

################
### lagrange
################
lagrange <- function(l000){
  out <- l000 + theme_classic(
  ) + coord_cartesian(
    xlim = c(32, 224)
  ) + scale_y_reverse(
  ) + scale_color_viridis_c(
    limits = c(0, 1)
  ) + theme(
    axis.text  = element_text(size = 16, color = "black"),
    axis.title = element_text(size = 20),
    legend.title = element_text(size = 16),
    legend.text  = element_text(size = 16)
  ) + labs(
    x = "space"
  )
  return(out)
}
lag_df <- function(df_lag){
  df <- df_lag
  df$time <- df$time - 1000
  l000 <- ggplot(df, aes(x, time)) + geom_point(
    aes(color = ERK), size = 4
  )
  return(lagrange(l000))
}


################
### spring
################
spring <- function(s000){
  out <- s000 + theme_classic(
  ) + coord_cartesian(
    xlim = c(32, 224)
  ) + scale_y_reverse(
  ) + theme(
    axis.text  = element_text(size = 16, color = "black"),
    axis.title = element_text(size = 20)
  ) + scale_color_viridis_c(
  ) + labs(
    x = "space"
  )
  return(out)
}
spr_df <- function(df_spr){
  df <- df_spr
  df$time <- df$time - 1000
  s000 <- ggplot(df, aes(x, time)) + geom_point(
    aes(color = ERK), size = 4
  )
  return(spring(s000))
}


################
### compare
################
facet_color <- function(plot, colors, textcol){
  g          <- ggplot_gtable(ggplot_build(plot))
  strip_both <- which(grepl('strip-', g$layout$name))
  fills      <- colors
  k <- 1
  for (i in strip_both) {
    ### box fill
    j <- which(grepl('rect', g$grobs[[i]]$grobs[[1]]$childrenOrder))
    g$grobs[[i]]$grobs[[1]]$children[[j]]$gp$fill <- fills[k]
    ### letters color
    l <- which(grepl('title', g$grobs[[i]]$grobs[[1]]$childrenOrder))
    m <- which(grepl('text',  g$grobs[[i]]$grobs[[1]]$children[[l]]$children))
    g$grobs[[i]]$grobs[[1]]$children[[l]]$children[[m]]$gp$col <- textcol[k]
    #
    k <- k + 1
  }
  return(g)
}

ftoc <- function(x){
  if(x == "fluid"){
    return("continuum")
  } else {
    return(x)
  }
}

### colomns order in comparison
colorder <- c(3, 2, 1) ### 1 merge, 2 continuum, 3 particle

compare <- function(c000){
  out <- c000 + scale_y_reverse(
  ) + theme_bw(
  ) + theme(
    strip.text = element_text(size = 20),
    legend.title = element_text(size = 16),
    legend.text  = element_text(size = 16),
    axis.text  = element_text(size = 16, color = "black"),
    axis.title = element_text(size = 20)
  ) + coord_cartesian(
    xlim = c(96, 160)
  ) + labs(
    x = "space"
  )
  out <- facet_color(out, c('white', 'royalblue1', 'grey')[colorder], c('grey10', 'white', 'grey10')[colorder])
  return(out)
}

compare_df <- function(df_com){
  df_com$model <- sapply(as.character(df_com$model), ftoc); #print(str(df_com))
  df_com$fa    <- sapply(as.character(df_com$fa   ), ftoc)
  df_com$fa <- factor(df_com$fa, levels = unique(df_com$fa)[colorder])
  c000 <- ggplot(df_com, aes(x, time)) + geom_path(
    data = subset(df_com, model == 'continuum'), mapping = aes(x, time, group = cell, size = si), alpha = 0.4, color = 'blue'
  ) + geom_path(
    data = subset(df_com, model == 'particle'),  mapping = aes(x, time, group = cell, size = si), alpha = 0.4, color = 'black'
  ) + geom_point(
    aes(color = ERK, alpha = al), size = 0.1
  ) + facet_grid(
    . ~ fa
  ) + scale_color_viridis_c(
    limits = c(0, 1)
  ) + scale_alpha_identity(
    guide = 'none'
  ) + scale_size_identity(
    guide = 'none'
  ); #print("1")
  return(compare(c000))
}

grid_erk <- function(erkshape){
  e100 <- erkshape + theme(
    axis.text  = element_text(size = 14, color = "black"),
    axis.title = element_text(size = 16)
  ) + labs(
    x = "space"
  )
  return(e100)
}

################
### graphs make and save
################
inmain <- function(rdata, outdir, header, exts = 'png'){
  print(str_c("loading ", rdata))
  load(rdata)
  ###
  for(ext in exts){
    print(ext)
    dir <- str_c(outdir, "/", ext)
    if(! ext %in% list.files(outdir)){
      dir.create(dir)
    }
    header2 <- str_c(dir, "/", header)
    #
    #e100 <- euler(e000)
    #ggsave(plot = e100, file = str_c(header2, 'fig01_euler.',    ext), height = 120, width = 160, unit = 'mm')
    e101 <- Vx(df_eul)
    ggsave(plot = e101, file = str_c(header2, 'sup03_Vx.',       ext), height = 120, width = 160, unit = 'mm')
    e102 <- Rho(df_eul)
    ggsave(plot = e102, file = str_c(header2, 'sup03_Rho.',      ext), height = 120, width = 160, unit = 'mm')

    #l100 <- lagrange(l000)
    #l100 <- lag_df(df_lag)
    #ggsave(plot = l100, file = str_c(header2, 'fig02_lagrange.', ext), height = 120, width = 160, unit = 'mm')

    #s100 <- spring(s000)
    #s100 <- spr_df(df_spr)
    #ggsave(plot = s100, file = str_c(header2, 'fig03_spring.',   ext), height = 120, width = 160, unit = 'mm')

    #c100 <- compare(c000)
    c100 <- compare_df(df_com)
    ggsave(plot = c100, file = str_c(header2, 'sup03_compare.',  ext), height = 120, width = 160, unit = 'mm')

    e100 <- grid_erk(erkshape)
    ggsave(plot = e100, file = str_c(header2, 'sup03_erkshape.', ext), height =  40, width = 160, unit = 'mm')
  }
}

################
### run
################
if(F){
  j <- 1
  load(as.character(fls$path[j]))
  e103 <- ERK(df_eul)
  for(ext in exts){
    ggsave(plot = e103, file = str_c(outdir, "/", "ERK.", ext), height = 120, width = 160, unit = 'mm')
  }
  if(F){
    q("no")
  }
}

for(j in 1:length(fls[[1]])){
  headname <- str_c(as.character(fls$label[j]), "_")
  inmain(as.character(fls$path[j]), outdir, headname, exts)
}


################
###
################


###
