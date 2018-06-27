#!/usr/bin/env python
import pandas as pd
import os
import numpy as np
import matplotlib
from matplotlib.colors import LogNorm
from matplotlib import rc
matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
matplotlib.rcParams['text.latex.preamble'].append(r'\usepackage{amsmath}')
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import argparse as a
import warnings

#Get Options

parser = a.ArgumentParser(description='2D Fat Jet Mass plot')
parser.add_argument('-f', '--files', nargs='*', required=True, help='Path to dataframe file(s) from ROOTCuts')
parser.add_argument('--HT', type=float, default=0., help='Apply minimum HT cut, default = 0')
parser.add_argument('--discrim', type=float, default=0.3, help='Minimum double-b-tag discriminator output, default is 0.3')
parser.add_argument('--type', default='Signal', help='Type of sample: e.g. Signal, TTJets, QCD etc')
parser.add_argument('-x', '--NoX', action='store_true', help='This argument suppresses showing plots via X-forwarding')
parser.add_argument('-o', '--NoOutput', action='store_true', help='This argument suppresses the output of PDF plots')
parser.add_argument('-v', '--verbose', action='store_true', help='Increased verbosity level')
parser.add_argument('--cmap', default='BuPu', help='Optional colour map, default is BuPu')
args=parser.parse_args()

df_list = []
for file in args.files:
    df = pd.read_csv(file, delimiter=r'\s+')
    df_list.append(df)
df = pd.concat(df_list)

#Make the output directories
directory = '2DFatJetMass_{0}'.format(args.type)
temp_dir = directory
suffix = 1
while os.path.exists(temp_dir):
    suffix += 1
    temp_dir = directory + '_{0}'.format(suffix)
if not args.NoOutput:
    print('Files will be written to: {0}'.format(temp_dir))
    os.makedirs(temp_dir)

sns.set_style("white")
discrim = args.discrim

df = df.loc[((df['FatDoubleBJetA_mass'] < 200.) & (df['FatDoubleBJetB_mass'] < 200.) & (df['FatDoubleBJetA_mass'] > 0.) & (df['FatDoubleBJetB_mass'] > 0.) & (df['FatDoubleBJetA_discrim'] > discrim) & (df['FatDoubleBJetB_discrim'] > discrim) & (df['HT'] > args.HT))]

g = sns.JointGrid(x=df['FatDoubleBJetA_mass'], y=df['FatDoubleBJetB_mass'], space=0.)
g.plot_joint(plt.hexbin, norm=LogNorm(), cmap=args.cmap, gridsize=90, C=df['crosssec'], reduce_C_function=np.sum)

cm = plt.cm.get_cmap(args.cmap)

nx, binsx, patchesx = g.ax_marg_x.hist(df['FatDoubleBJetA_mass'], log=True, bins=30, weights=df['crosssec'])
colx = 1.5*(nx-nx.min())/(nx.max()-nx.min())
for c, p in zip(colx, patchesx):
    plt.setp(p, 'facecolor', cm(c))

ny, binsy, patchesy = g.ax_marg_y.hist(df['FatDoubleBJetB_mass'], bins=30, log=True, orientation='horizontal', weights=df['crosssec'])
coly = 1.5*(ny-ny.min())/(ny.max()-ny.min())
for c, p in zip(coly, patchesy):
    plt.setp(p, 'facecolor', cm(c))

g.set_axis_labels('AK8 Double-$b$-tagged Jet A SoftDrop Mass', 'AK8 Double-$b$-tagged Jet B SoftDrop Mass', fontsize=14)

sns.plt.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.15)
cax = g.fig.add_axes([.91, .3, .02, .3])  # x, y, width, height
sns.plt.colorbar(cax=cax)

if not args.NoOutput:
    plt.savefig(os.path.join(temp_dir, '2DFatJetMass.pdf'))
    print('Saved 2DFatJetMass.pdf output file')
if not args.NoX:
    plt.show()
