#!/usr/bin/python
# -*- coding: utf-8 -*
import matplotlib.pyplot as plt
## bosch colors for plotting ;-)
import seaborn as sns

from base64 import b16encode
toHEX = lambda rgb: '#%02x%02x%02x' % rgb

# grays as defined by style guide
bosch_gray_palette = ['#525F6B', '#BFC0C2']
colors_gry = sns.color_palette(bosch_gray_palette)

# standard colors as defined by style guide
bosch_standard_palette = [toHEX((168,1,99)), toHEX((208,103,173)), toHEX((63,19,108)), toHEX((150,124,177)),
                                 toHEX((8,66,126)), toHEX((109,154,188)), toHEX((14,120,197)), toHEX((111,185,226)),
                                 toHEX((19,153,160)), toHEX((111,201,204)), toHEX((103,180,25)), toHEX((174,219,125)),
                                 toHEX((10,81,57)), toHEX((110,162,147))]
colors_std = sns.color_palette('#E20015	#B90276	#50237F	#005691	#008ECF	#00A8B0	#78BE20	#006249'.split())
colors_stdr = sns.color_palette(bosch_standard_palette)

# optimized standard palette: remove red, start with light blue (i.e. plots with two colors: ligth blue + turquoise)
colors_opt = colors_stdr[:-1]
colors_opt = colors_opt[2:] + colors_opt[:2]

sns.set_palette(colors_stdr)
sns.set_style("whitegrid")

params_savefig = {'figure.figsize': (8, 4),
                  'figure.dpi': 125,
                  
                  'font.sans-serif': ['Arial', 'Bitstream Vera Sans', 'Arial'],
                  
                  'legend.fontsize': 'x-large', 
                  'legend.columnspacing': 1,
                  'legend.labelspacing': 0.25,
                  'legend.fancybox': True,
                  
                  'xtick.labelsize': 'x-large', 
                  'ytick.labelsize': 'x-large', 
                  
                  'axes.labelsize': 'x-large',
                  'axes.labelweight': 'normal',
                  #'axes.color_cycle': cols,
                  
                  'lines.linewidth'     : 2.0,
                  'axes.linewidth'      : 2.0,     # edge linewidth
                  'axes.grid'           : True,   # display grid or not

                  'grid.linewidth'      : 1.0,     # in points
                  'grid.alpha'          : 0.5,     # transparency, between 0.0 and 1.0                  
                  
                  'savefig.dpi':        300,
                  'figure.facecolor' :    'white'
                  }

bosch_params = {'figure.figsize': (8, 4),
          'figure.dpi': 125,
          
          'font.sans-serif': ['Arial', 'Bitstream Vera Sans', 'Arial'],
                  
          'legend.fontsize': 'x-large', 
          'legend.columnspacing': 1,
          'legend.labelspacing': 0.25,
          'legend.fancybox': True,
          
          'xtick.labelsize': 'x-large', 
          'ytick.labelsize': 'x-large', 
          
          'axes.labelsize': 'x-large',
          'axes.labelweight': 'normal',
          #'axes.color_cycle': cols,
          
          'lines.linewidth'     : 2.0,
          'axes.linewidth'      : 2.0,     # edge linewidth
          'axes.grid'           : True,   # display grid or not

          'grid.linewidth'      : 2.0,     # in points
          'grid.alpha'          : 0.5,     # transparency, between 0.0 and 1.0                  
          

          'savefig.dpi': 300,
          'figure.facecolor' :    'white',
          
          'mathtext.cal' : 'cursive',
          'mathtext.rm' : 'serif',
          'mathtext.tt' : 'monospace',
          'mathtext.it' : 'serif:italic',
          'mathtext.bf' : 'serif:bold',
          'mathtext.sf' : 'sans',
          'mathtext.fontset' : 'stixsans', # Should be 'cm' (Computer Modern), 'stix',
                                   # 'stixsans' or 'custom'
          'mathtext.fallback_to_cm' : True,  # When True, use symbols from the Computer Modern
                                             # fonts when a symbol can not be found in one of
                                             # the custom math fonts.

          'mathtext.default' : 'regular' # The default font to use for math.
                                   # Can be any of the LaTeX font names, including
                                   # the special name "regular" for the same font
                                   # used in regular text.
          }

plt.rcParams.update(bosch_params)
