# Adaptive LASSO solver higher order oscillatory systems 
This code includes the linear regression solver (Adaptive LASSO, also LASSO and OLS) and simulation for the pairwise and higher order oscillatory systems. 

## Outline
This program is accompanyed with ... as the supplementary code material, and is able to generate all graph and table data presented in the context. 

## Requirement
Python3 (3.11.x, also tested on 3.9.x and 3.10.x)
modules:

  essentials: 
  
  scipy
  
  matplotlib
  
  numpy
  
  random

  statsmodels (0.14.0)

  skglm (optional, not necessary to be implemented for all examples in this code)
  
  scikit-learn (sklearn)

  visualizations: 
  
  typing
  
  tabulate
  
  datetime
  
  time

  others:
  
  csv
  
  os
  
## Getting started
This is free to clone through SSH: 

$ git clone "git@github.com:Weiwei-Su112/Adaptive-LASSO-for-higher-order-oscillatory-systems.git"

or HTTPS, etc..

## Data 
By courtesy of [Škoch, A. et al.](https://doi.org/10.1038/s41597-022-01596-90), in the subsection "Application to a real-world network" of ... we tested on the structural connectivity matrices of human brain.

The data is mainly in the folder 

* structural_connectivity_matrices

yet 4 data used in the main text of ... are

* S038.csv
  
* S050.csv
  
* S062.csv
  
* S085.csv
  
and are put in the root directory. Any other data from the folder "structural_connectivity_matrices" can be easily implemented, and is explained in the next section.

Also, 

* fig4_seeds.csv

stores the seeds for simulations in Fig_4.py
  
## Usage

GeneralInteraction.py:

This script includes nearly all the necessary toolkits (except for some functional) including data simulation, solvers, visualization tools, etc.. 

DO NOT run this script directly. 

The executive scripts: 

* Fig_1.py
  corresponds to the Fig. 1 in ...
  
* Fig_2.py
  corresponds to the Fig. 2 in ...
  
* Fig_3.py
  corresponds to the Fig. 3 and Supp. 1 in ...
  
* Fig_4.py
  corresponds to the Fig. 4, Table 3, and Table 4 in ...
  
* binomial_test.py
  corresponds to Table 1 and Table 2 in ...
  
* brain.py & fig4_fig5.py
  corresponds to Fig. 5 and Table 5 in ....
  To change the data usage, at Line 171 it is free to change S0XX.csv into any data you would like to test in the folder "structural_connectivity_matrices"

All these scripts are able to run via "python3 xxx.py" without further parsing parameters. 

All parameters are controlled directly inside the executive scripts directly. 

## Licence
MIT

Please contact [Ryota Kobayashi](http://www.hk.k.u-tokyo.ac.jp/r-koba/en/contact.html) if you want to use the code for commercial purposes.
