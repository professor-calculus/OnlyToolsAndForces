#!/usr/bin/env python
import pandas as pd
import dask.dataframe as dd
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
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

parser = a.ArgumentParser(description='2D Fat Jet Score plot')
parser.add_argument('-f', '--files', nargs='*', required=True, help='Path to dataframe file(s) from ROOTCuts')
parser.add_argument('--HT', type=float, default=0., help='Apply minimum HT cut, default = 0')
parser.add_argument('--minMass', type=float, default=0., help='Minimum fat jet mass, default is 0.')
parser.add_argument('--maxMass', type=float, default=999999., help='Maximum fat jet mass, default is 999999.')
parser.add_argument('--type', default='Signal', help='Type of sample: e.g. Signal, TTJets, QCD etc')
parser.add_argument('--region', default='Signal', help='Signal, 2b1mu, 0b1mu, 2mu, 0b2mu')
parser.add_argument('--nHiggs2bb', action='store_true', help='Force all Higgs-->bb, Signal only')
parser.add_argument('-x', '--NoX', action='store_true', help='This argument suppresses showing plots via X-forwarding')
parser.add_argument('-o', '--NoOutput', action='store_true', help='This argument suppresses the output of PDF plots')
parser.add_argument('-v', '--verbose', action='store_true', help='Increased verbosity level')
parser.add_argument('--cmap', default='BuPu', help='Optional colour map, default is BuPu')
args=parser.parse_args()

df = dd.read_csv(args.files, delimiter=r'\s+')

#Make the output directories
filepath = '2DFatJetScore_{0}_{1}Region_HT{2}'.format(args.type, args.region, int(args.HT))
if args.nHiggs2bb:
    filepath = filepath + 'Higgs2bb'
temp_dir = filepath
suffix = 1
while os.path.exists(temp_dir):
    suffix += 1
    temp_dir = filepath + '_{0}'.format(suffix)
if not args.NoOutput:
    print('File will be written as: {0}.pdf'.format(temp_dir))

sns.set_style("white")

# Only events with 2 AK8 jets, otherwise doesn't make sense!
df = df.loc[(df['NFatJet'] > 1)]

df = df.loc[((df['FatDoubleBJetA_mass'] < args.maxMass) & (df['FatDoubleBJetB_mass'] < args.maxMass) & (df['FatDoubleBJetA_mass'] > args.minMass) & (df['FatDoubleBJetB_mass'] > args.minMass) & (df['HT'] > args.HT))]
if args.nHiggs2bb:
    df = df.loc[(df['nHiggs2bb'] == 2)]

if args.region == 'Signal':
    df = df.loc[(df['nLooseMuons'] == 0)]
elif args.region == '1mu':
    df = df.loc[((df['nTightMuons'] == 1) & (df['Muon_MHT_TransMass'] < 100.))]
elif args.region == '1b1mu':
    df = df.loc[((df['NBJet'] == 1) & (df['nTightMuons'] == 1) & (df['Muon_MHT_TransMass'] < 100.))]
elif args.region == '2b1mu':
    df = df.loc[((df['NBJet'] == 2) & (df['nTightMuons'] == 1) & (df['Muon_MHT_TransMass'] < 100.))]
elif args.region == '0b1mu':
    df = df.loc[((df['NBJet'] == 0) & (df['nTightMuons'] == 1) & (df['Muon_MHT_TransMass'] < 100.))]
elif args.region == '2mu':
    df = df.loc[((df['nTightMuons'] == 2) & (df['Muons_InvMass'] > 80.) & (df['Muons_InvMass'] < 100.))]
elif args.region == '2b2mu':
    df = df.loc[((df['NBJet'] == 2) & (df['nTightMuons'] == 2) & (df['Muons_InvMass'] > 80.) & (df['Muons_InvMass'] < 100.))]
elif args.region == '1b2mu':
    df = df.loc[((df['NBJet'] == 1) & (df['nTightMuons'] == 2) & (df['Muons_InvMass'] > 80.) & (df['Muons_InvMass'] < 100.))]
elif args.region == '0b2mu':
    df = df.loc[((df['NBJet'] == 0) & (df['nTightMuons'] == 2) & (df['Muons_InvMass'] > 80.) & (df['Muons_InvMass'] < 100.))]

df = df[['FatDoubleBJetA_discrim', 'FatDoubleBJetB_discrim', 'crosssec']]
df = df.compute()

# Switch fat jet A and B half of the time (randomly) in order to generate symmetric 2D plot
df['Sample'] = 'A'
df.loc[df.sample(frac=0.5, replace=False).index,'Sample'] = 'B'
df.loc[(df['Sample'] == 'B'), ['FatDoubleBJetA_discrim','FatDoubleBJetB_discrim']] = df.loc[(df['Sample'] == 'B'), ['FatDoubleBJetB_discrim','FatDoubleBJetA_discrim']].values

if df.shape[0] == 0:
    sys.exit('Error: No events left after cuts! Cannot plot.')

plt.figure()

g = sns.JointGrid(x=df['FatDoubleBJetA_discrim'], y=df['FatDoubleBJetB_discrim'], space=0.)

g.plot_joint(plt.hexbin, norm=LogNorm(), cmap=args.cmap, gridsize=150, C=df['crosssec'], reduce_C_function=np.sum)
#g.plot_joint(plt.hexbin, norm=LogNorm(), cmap=args.cmap, gridsize=150)

cm = plt.cm.get_cmap(args.cmap)

nx, binsx, patchesx = g.ax_marg_x.hist(df['FatDoubleBJetA_discrim'], log=True, bins=30, weights=df['crosssec'])
colx = 1.5*(nx-nx.min())/(nx.max()-nx.min())
for c, p in zip(colx, patchesx):
    plt.setp(p, 'facecolor', cm(c))

ny, binsy, patchesy = g.ax_marg_y.hist(df['FatDoubleBJetB_discrim'], bins=30, log=True, orientation='horizontal', weights=df['crosssec'])
coly = 1.5*(ny-ny.min())/(ny.max()-ny.min())
for c, p in zip(coly, patchesy):
    plt.setp(p, 'facecolor', cm(c))

g.set_axis_labels('AK8 Fat Jet A Double-$b$-tag Score', 'AK8 Fat Jet A Double-$b$-tag Score', fontsize=14)

plt.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.15)
cax = g.fig.add_axes([.91, .3, .02, .3])  # x, y, width, height
plt.colorbar(cax=cax)

if not args.NoOutput:
    plt.savefig('{0}.pdf'.format(temp_dir))
    print('Saved {0}.pdf output file'.format(temp_dir))
if not args.NoX:
    plt.show()
