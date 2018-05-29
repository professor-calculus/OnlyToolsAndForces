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
parser.add_argument('-q', '--QCD', nargs='*', default=None, help='Path to QCD dataframe file from ROOTCuts')
parser.add_argument('-t', '--TTJets', nargs='*', default=None, help='Path to TTJets dataframe file from ROOTCuts')
parser.add_argument('-x', '--NoX', action='store_true', help='This argument suppresses showing plots via X-forwarding')
parser.add_argument('-o', '--NoOutput', action='store_true', help='This argument suppresses the output of PDF plots')
parser.add_argument('-v', '--verbose', action='store_true', help='Increased verbosity level')
parser.add_argument('--style', default=None, help='Optional drawing style, e.g. \"ggplot\" in Matplotlib or \"dark\" in Seaborn')
args=parser.parse_args()

if args.verbose:
    parser.print_help()
else:
    warnings.filterwarnings("ignore")

if args.kdeplot_fill:
    args.kdeplot = True

if args.style:
    plt.style.use(args.style)

print '\nPython Mountain Range Plotter\n'
print(args.signal)

df_sig_list = []
for file in args.signal:
    df = pd.read_csv(file, delimiter=r'\s+')
    df_sig_list.append(df)
df_sig = pd.concat(df_sig_list)

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

if args.QCD:
    df_list = []
    for file in args.QCD:
        df = pd.read_csv(file, delimiter=r'\s+')
        df_list.append(df)
    df_QCD = pd.concat(df_list)
if args.TTJets:
    df_list = []
    for file in args.TTJets:
        df = pd.read_csv(file, delimiter=r'\s+')
        df_list.append(df)
    df_TTJets = pd.concat(df_list)

n_signal = len(args.signal)
linewidth = 3.

for var in variables:
    plt.figure()
    temp_i = 0
    for index, row in df_sig_masses.iterrows():
        temp_i += 5
        label='$M_{\mathrm{Squark}}$ = ' + str(row["M_sq"]) + ', $M_{\mathrm{LSP}}$ = ' + str(row["M_lsp"])
        df_temp = df_sig.loc[(df_sig['M_sq'] == row['M_sq']) & (df_sig['M_lsp'] == row['M_lsp'])]
        plt.hist(df_temp['Bin'], label=label, weights=df['Yield'], log=True, histtype="step", linewidth=linewidth, zorder=35-temp_i)

    if (args.QCD):
        plt.hist(df_QCD['Bin'], weights=df_QCD['Yield'], label='QCD background', log=True, histtype="step", linewidth=linewidth, hatch="//", zorder=0)
    if (args.TTJets):
        plt.hist(df_TTJets['Bin'], weights=df_TTJets['Yield'], label='$t \overline{t}$ + $jets$ background', log=True, histtype="step", linewidth=linewidth, hatch="\\\\", zorder=5)



    plt.xlabel('Bin Number', size=14)
    leg = plt.legend(loc='upper right', fontsize='medium')
    leg.set_zorder(100)
    if not args.NoOutput:
        plt.savefig(os.path.join(temp_dir, 'MountainRange.pdf'))
        print('Saved MountainRange.pdf output file')
    if not args.NoX:
        plt.show()
