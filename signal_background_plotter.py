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
import seaborn as sns
import sys
import argparse as a
import warnings

#Get Options

parser = a.ArgumentParser(description='Signal vs Background plot')
parser.add_argument('-s', '--signal', nargs='*', required=True, help='Path to signal .root file(s) from ROOTCuts')
parser.add_argument('-q', '--QCD', default=None, help='Path to QCD .root file from ROOTCuts')
parser.add_argument('-t', '--TTJets', default=None, help='Path to TTJets .root file from ROOTCuts')
parser.add_argument('--HT_cut', type=float, default=None, help='Apply minimum HT cut')
parser.add_argument('-x', '--NoX', action='store_true', help='This argument suppresses showing plots via X-forwarding')
parser.add_argument('-o', '--NoOutput', action='store_true', help='This argument suppresses the output of PDF plots')
parser.add_argument('-v', '--verbose', action='store_true', help='Increased verbosity level')
parser.add_argument('--style', default=None, help='Optional drawing style, e.g. \"ggplot\" in Matplotlib or \"dark\" in Seaborn')
parser.add_argument('--kdeplot', action='store_true', help='Use kdeplot in Seaborn instead of matplotlib histogram')
parser.add_argument('--kdeplot_fill', action='store_true', help='Same as --kdeplot but area under each line is filled')
parser.add_argument('--signal_crosssec', default=1., help='Signal cross-section')
parser.add_argument('--qcd_crosssec', default=1., help='QCD cross-section')
parser.add_argument('--tt_crosssec', default=1., help='TTJets cross-section')
args=parser.parse_args()

if args.verbose:
    parser.print_help()
else:
    warnings.filterwarnings("ignore")

if args.kdeplot_fill:
    args.kdeplot = True

if args.style:
    if args.kdeplot:
        sns.set_style(args.style)
    else:
        plt.style.use(args.style)

print '\nPython Signal vs Background Plotter\n'
print(args.signal)

df_sig_list = []
for file in args.signal:
    df = pd.read_csv(file, delimiter=r'\s+')
    list.append(df)
df_sig = pd.concat(df_sig_list)

print '\nSuccessfully read root files into dataframe\n'

#Make the output directories
directory = 'Signal_vs_Background'
suffix = 1
while os.path.exists(directory):
    suffix += 1
    directory = args.OutDir + '_{0}'.format(suffix)
print('Files will be written to: {0}'.format(directory))
os.makedirs(directory)

df_sig_masses = df_sig[['Truth_Msq', 'Truth_Mlsp']].drop_duplicates()
df_sig_masses = df_sig_masses.sort_values(by=['Truth_Msq', 'Truth_Mlsp'])
print(df_sig_masses.head())

if args.HT_cut:
    df_sig = df_sig.loc[(df_sig['Uncut_HT'] > args.HT_cut)]

if args.QCD:
    df_QCD = pd.read_csv(args.QCD, delimiter=r'\s+')
    if args.HT_cut:
        df_QCD = df_QCD.loc[(df_QCD['HT'] > args.HT_cut)]
if args.TTJets:
    df_TTJets = pd.read_csv(args.TTJets, delimiter=r'\s+')
    if args.HT_cut:
        df_TTJets = df_TTJets.loc[(df_TTJets['HT'] > args.HT_cut)]

bins_HT = np.linspace(0.,8000.,160)
bins_MHT = np.linspace(0.,2000.,200)
bins_DelR = np.linspace(0.,5.,100)
bins_BMass = np.linspace(0.,500.,100)
bins_njet = np.arange(0, 20, 1)
bins_nbjet = np.arange(0, 14, 1)
bins_BDP = np.linspace(0., 3., 60)

dict = {'MET': {'branch': 'MET', 'bins': bins_MHT, 'title': 'Missing $E_{T}$ / GeV'},
        'MHT': {'branch': 'MHT', 'bins': bins_MHT, 'title': 'Missing $H_{T}$ / GeV'},
        'HiggsPT': {'branch': 'Higgs_PT', 'bins': bins_MHT, 'title': 'Higgs $p_{T}$ / GeV'},
        'HT': {'branch': 'HT', 'bins': bins_HT, 'title': 'Total $H_{T}$ / GeV'},
        'DelR': {'branch': 'bJetsDelR', 'bins': bins_DelR, 'title': 'b-Jets $\Delta R$'},
        'NJet': {'branch': 'NJet', 'bins': bins_njet, 'title': 'Number of Jets'},
        'NBJet': {'branch': 'NBJet', 'bins': bins_nbjet, 'title': 'Number of Bottom Quark Jets'},
        'BDP': {'branch': 'bDPhi', 'bins': bins_BDP, 'title': '$\Delta\Phi^{\*}$'},
        }

variables = ['MET', 'MHT', 'HT', 'DelR', 'NJet', 'NBJet', 'BMass', '2BMass']
signal_only_vars = ['DelR', 'BMass', '2BMass']

n_signal = len(args.signal)

for var in variables:
    plt.figure()
    for index, row in df_sig_masses.iterrows():
        label='$M_{\mathrm{Squark}}$ = ' + str(row["Truth_Msq"]) + ', $M_{\mathrm{LSP}}$ = ' + str(row["Truth_Mlsp"])
        df_temp = df_sig.loc[(df_sig['Truth_Msq'] == row['Truth_Msq']) & (df_sig['Truth_Mlsp'] == row['Truth_Mlsp'])]
        if args.kdeplot:
            sns.kdeplot(df_temp[dict[var]['branch']], label=label, shade=args.kdeplot_fill)
        else:
            plt.hist(df_temp[dict[var]['branch']], bins=dict[var]['bins'], alpha=0.8, label=label, density=True, log=True)

    if args.QCD and var not in signal_only_vars:
        if args.kdeplot:
            sns.kdeplot(df_QCD[dict[var]['branch']], label='QCD background', shade=args.kdeplot_fill)
        else:
            plt.hist(df_QCD[dict[var]['branch']], bins=dict[var]['bins'], alpha=0.8, label='QCD background', density=True, log=True)
    if args.TTJets and var not in signal_only_vars:
        if args.kdeplot:
            sns.kdeplot(df_TTJets[dict[var]['branch']], label='$t \overline{t}$ + $jets$ background', shade=args.kdeplot_fill)
        else:
            plt.hist(df_TTJets[dict[var]['branch']], bins=dict[var]['bins'], alpha=0.8, label='$t \overline{t}$ + $jets$ background', density=True, log=True)
            

    plt.xlabel(dict[var]['title'])
    plt.legend(loc='best', fontsize='small')
    if not args.NoOutput:
        plt.savefig(os.path.join(directory, var + '.pdf'))
        print('Saved ' + var + '.pdf output file')
    if not args.NoX:
        plt.show()
