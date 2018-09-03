#!/usr/bin/env python
import pandas as pd
import os
import numpy as np
import matplotlib
from matplotlib import rc
matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
matplotlib.rcParams['text.latex.preamble'].append(r'\usepackage{amsmath}')
import matplotlib.pyplot as plt
import sys
import argparse as a
import warnings

#Get Options

parser = a.ArgumentParser(description='Signal vs Background plot')
parser.add_argument('-s', '--signal', nargs='*', required=True, help='Path to signal dataframe file(s) from ROOTCuts')
parser.add_argument('-q', '--QCD', default=None, nargs='*', help='Path to QCD dataframe file(s) from ROOTCuts')
parser.add_argument('-m', '--MSSM', default=None, nargs='*', help='Path to MSSM dataframe file(s) from ROOTCuts')
parser.add_argument('-t', '--TTJets', default=None, nargs='*', help='Path to TTJets dataframe file(s) from ROOTCuts')
parser.add_argument('-w', '--WJets', default=None, nargs='*', help='Path to W+Jets dataframe file(s) from ROOTCuts')
parser.add_argument('-z', '--ZJets', default=None, nargs='*', help='Path to Z+Jets dataframe file(s) from ROOTCuts')
parser.add_argument('--DiBoson', default=None, nargs='*', help='Path to DiBoson dataframe file(s) from ROOTCuts')
parser.add_argument('--SingleTop', default=None, nargs='*', help='Path to SingleTop dataframe file(s) from ROOTCuts')
parser.add_argument('--TTW', default=None, nargs='*', help='Path to TTW dataframe file(s) from ROOTCuts')
parser.add_argument('--TTZ', default=None, nargs='*', help='Path to TTZ dataframe file(s) from ROOTCuts')
parser.add_argument('-x', '--NoX', action='store_true', help='This argument suppresses showing plots via X-forwarding')
parser.add_argument('-o', '--NoOutput', action='store_true', help='This argument suppresses the output of PDF plots')
parser.add_argument('-v', '--verbose', action='store_true', help='Increased verbosity level')
parser.add_argument('--style', default=None, help='Optional drawing style, e.g. \"ggplot\" in Matplotlib or \"dark\" in Seaborn')
args=parser.parse_args()

if args.verbose:
    parser.print_help()
else:
    warnings.filterwarnings("ignore")

if args.style:
    plt.style.use(args.style)

print '\nPython Mountain Range Plotter\n'
print(args.signal)

x = range(27)

df_sig_list = []
for file in args.signal:
    df = pd.read_csv(file, delimiter=r'\s+')
    df['Bin'] = x
    df_sig_list.append(df)
df_sig = pd.concat(df_sig_list)
df_sig = df_sig.groupby(by=['Type', 'M_sq', 'M_lsp', 'HT_bin', 'MHT_bin', 'n_Jet_bin', 'n_DoubleBJet_bin', 'n_Muons_bin']).sum()

if args.verbose:
    print('Signal:')
    print(df_sig)

print '\nSuccessfully read dataframe\n'

#Make the output directories
directory = 'Mountain_Range'
temp_dir = directory
suffix = 1
while os.path.exists(temp_dir):
    suffix += 1
    temp_dir = directory + '_{0}'.format(suffix)
if not args.NoOutput:
    print('Files will be written to: {0}'.format(temp_dir))
    os.makedirs(temp_dir)

df_sig_masses = df_sig[['M_sq', 'M_lsp']].drop_duplicates()
df_sig_masses = df_sig_masses.sort_values(by=['M_sq', 'M_lsp'])
print(df_sig_masses.head())

theBkgs = []
bkgLabels = []
bkgWeights = []
if args.QCD:
    df_list = []
    for file in args.QCD:
        df = pd.read_csv(file, delimiter=r'\s+')
        df_list.append(df)
    df = pd.concat(df_list)
    df = df.groupby(by=['Type', 'M_sq', 'M_lsp', 'HT_bin', 'MHT_bin', 'n_Jet_bin', 'n_DoubleBJet_bin', 'n_Muons_bin']).sum()
    theBkgs.append(x)
    bkgWeights.append(df['Yield'])
    bkgLabels.append('QCD Multijet background')
if args.TTJets:
    df_list = []
    for file in args.TTJets:
        df = pd.read_csv(file, delimiter=r'\s+')
        df_list.append(df)
    df = pd.concat(df_list)
    df = df.groupby(by=['Type', 'M_sq', 'M_lsp', 'HT_bin', 'MHT_bin', 'n_Jet_bin', 'n_DoubleBJet_bin', 'n_Muons_bin']).sum()
    theBkgs.append(x)
    bkgWeights.append(df['Yield'])
    bkgLabels.append('$t\overline{t}$ + jets background')
if args.WJets:
    df_list = []
    for file in args.WJets:
        df = pd.read_csv(file, delimiter=r'\s+')
        df_list.append(df)
    df = pd.concat(df_list)
    df = df.groupby(by=['Type', 'M_sq', 'M_lsp', 'HT_bin', 'MHT_bin', 'n_Jet_bin', 'n_DoubleBJet_bin', 'n_Muons_bin']).sum()
    theBkgs.append(x)
    bkgWeights.append(df['Yield'])
    bkgLabels.append('$W$ + jets background')
if args.ZJets:
    df_list = []
    for file in args.ZJets:
        df = pd.read_csv(file, delimiter=r'\s+')
        df_list.append(df)
    df = pd.concat(df_list)
    df = df.groupby(by=['Type', 'M_sq', 'M_lsp', 'HT_bin', 'MHT_bin', 'n_Jet_bin', 'n_DoubleBJet_bin', 'n_Muons_bin']).sum()
    theBkgs.append(x)
    bkgWeights.append(df['Yield'])
    bkgLabels.append('$Z$ + jets background')
if args.DiBoson:
    df_list = []
    for file in args.DiBoson:
        df = pd.read_csv(file, delimiter=r'\s+')
        df_list.append(df)
    df = pd.concat(df_list)
    df = df.groupby(by=['Type', 'M_sq', 'M_lsp', 'HT_bin', 'MHT_bin', 'n_Jet_bin', 'n_DoubleBJet_bin', 'n_Muons_bin']).sum()
    theBkgs.append(x)
    bkgWeights.append(df['Yield'])
    bkgLabels.append('Di-Boson background')
if args.SingleTop:
    df_list = []
    for file in args.SingleTop:
        df = pd.read_csv(file, delimiter=r'\s+')
        df_list.append(df)
    df = pd.concat(df_list)
    df = df.groupby(by=['Type', 'M_sq', 'M_lsp', 'HT_bin', 'MHT_bin', 'n_Jet_bin', 'n_DoubleBJet_bin', 'n_Muons_bin']).sum()
    theBkgs.append(x)
    bkgWeights.append(df['Yield'])
    bkgLabels.append('Single-$t$ background')
if args.TTW:
    df_list = []
    for file in args.TTW:
        df = pd.read_csv(file, delimiter=r'\s+')
        df_list.append(df)
    df = pd.concat(df_list)
    df = df.groupby(by=['Type', 'M_sq', 'M_lsp', 'HT_bin', 'MHT_bin', 'n_Jet_bin', 'n_DoubleBJet_bin', 'n_Muons_bin']).sum()
    theBkgs.append(x)
    bkgWeights.append(df['Yield'])
    bkgLabels.append('$t\overline{t}W$ background')
if args.TTZ:
    df_list = []
    for file in args.TTZ:
        df = pd.read_csv(file, delimiter=r'\s+')
        df_list.append(df)
    df = pd.concat(df_list)
    df = df.groupby(by=['Type', 'M_sq', 'M_lsp', 'HT_bin', 'MHT_bin', 'n_Jet_bin', 'n_DoubleBJet_bin', 'n_Muons_bin']).sum()
    theBkgs.append(x)
    bkgWeights.append(df['Yield'])
    bkgLabels.append('$t\overline{t}Z$ background')

n_signal = len(args.signal)
linewidth = 4.

if args.verbose:
    print(df_QCD)
    print(df_TTJets)

plt.figure()
temp_i = 0
for index, row in df_sig_masses.iterrows():
    temp_i += 5
    label='$M_{\mathrm{Squark}}$ = ' + str(row["M_sq"]) + ', $M_{\mathrm{LSP}}$ = ' + str(row["M_lsp"])
    df_temp = df_sig.loc[(df_sig['M_sq'] == row['M_sq']) & (df_sig['M_lsp'] == row['M_lsp'])]
    df_temp = df_temp.replace(0., 1e-5) 
    plt.hist(df_temp['Bin'], bins=x, label=label, weights=df_temp['Yield'], log=True, histtype="step", linewidth=linewidth, zorder=35-temp_i)

if (args.QCD) or (args.TTJets) or (args.WJets) or (args.ZJets) or (args.DiBoson) or (args.SingleTop) or (args.TTW) or (args.TTZ):
    plt.hist(theBkgs, bins=x, weights=bkgWeights, label=bkgLabels, stacked=True, log=True, histtype="step", linewidth=linewidth, hatch="////", zorder=0)

plt.xlabel('Bin Number', size=14)
plt.xticks(range(37))
plt.ylim(0.005, None)
leg = plt.legend(loc='upper right', fontsize='medium')
leg.set_zorder(100)
if not args.NoOutput:
    plt.savefig(os.path.join(temp_dir, 'MountainRange.pdf'))
    print('Saved MountainRange.pdf output file')
if not args.NoX:
    plt.show()
