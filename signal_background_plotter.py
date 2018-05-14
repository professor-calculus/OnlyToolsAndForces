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
parser.add_argument('-s', '--signal', nargs='*', required=True, help='Path to signal dataframe file(s) from ROOTCuts')
parser.add_argument('-q', '--QCD', default=None, help='Path to QCD dataframe file from ROOTCuts')
parser.add_argument('-m', '--MSSM', default=None, help='Path to MSSM dataframe file from ROOTCuts')
parser.add_argument('-t', '--TTJets', default=None, help='Path to TTJets dataframe file from ROOTCuts')
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
    df_sig_list.append(df)
df_sig = pd.concat(df_sig_list)

if args.verbose:
    print('Signal:')
    print(df_sig)

print '\nSuccessfully read dataframe\n'

#Make the output directories
directory = 'Signal_vs_Background'
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

if args.HT_cut:
    df_sig = df_sig.loc[(df_sig['HT'] > args.HT_cut)]

if args.MSSM:
    df_MSSM = pd.read_csv(args.MSSM, delimiter=r'\s+')
    if args.HT_cut:
        df_MSSM = df_MSSM.loc[(df_MSSM['HT'] > args.HT_cut)]
    if args.verbose:
        print('MSSM:')
        print(df_MSSM)

if args.QCD:
    df_QCD = pd.read_csv(args.QCD, delimiter=r'\s+')
    if args.HT_cut:
        df_QCD = df_QCD.loc[(df_QCD['HT'] > args.HT_cut)]
    if args.verbose:
        print('QCD:')
        print(df_QCD)
if args.TTJets:
    df_TTJets = pd.read_csv(args.TTJets, delimiter=r'\s+')
    if args.HT_cut:
        df_TTJets = df_TTJets.loc[(df_TTJets['HT'] > args.HT_cut)]
    if args.verbose:
        print('TTJets:')
        print(df_TTJets)

bins_HT = np.linspace(0.,5000.,50)
bins_MHT = np.linspace(0.,2500.,25)
bins_DelR = np.linspace(0.,5.,50)
bins_BMass = np.linspace(50.,200.,50)
bins_njet = np.arange(0, 20, 1)
bins_nbjet = np.arange(0, 14, 1)
bins_BDP = np.linspace(0., 3., 30)

dict = {'MET': {'branch': 'MET', 'bins': bins_MHT, 'title': 'Missing $E_{T}$ / GeV'},
        'MHT': {'branch': 'MHT', 'bins': bins_MHT, 'title': 'Missing $H_{T}$ / GeV'},
        'HiggsPT': {'branch': 'Higgs_PT', 'bins': bins_MHT, 'title': 'Higgs $p_{T}$ / GeV'},
        'HT': {'branch': 'HT', 'bins': bins_HT, 'title': 'Total $H_{T}$ / GeV'},
        'DelR': {'branch': 'bJetsDelR', 'bins': bins_DelR, 'title': 'b-Jets $\Delta R$'},
        'NJet': {'branch': 'NJet', 'bins': bins_njet, 'title': 'Number of Jets'},
        'NBJet': {'branch': 'NBJet', 'bins': bins_nbjet, 'title': 'Number of Bottom Quark Jets'},
        'BDP': {'branch': 'bDPhi', 'bins': bins_BDP, 'title': '$\Delta\Phi^{*}$'},
        }

dict_upper = {'MET': 2000.,
              'MHT': 2000.,
       	      'HiggsPT': 2000.,
       	      'HT': 5000.,
       	      'DelR': 5.,
       	      'NJet': 20.,
       	      'NBJet': 14.,
       	      'BDP': 3.,
             }

variables = ['MET', 'MHT', 'HT', 'DelR', 'NJet', 'NBJet', 'BDP']
signal_only_vars = ['DelR', 'BMass', '2BMass']

n_signal = len(args.signal)
linewidth = 3.

for var in variables:
    plt.figure()
    f, ax = plt.subplots()
    if var not in ['NJet', 'NBJet']:
        ax.set(yscale="log")

    if (args.QCD and not args.TTJets) and var not in signal_only_vars:
        if args.kdeplot and var not in ['NJet', 'NBJet']:
            sns.kdeplot(df_QCD[dict[var]['branch']], ax=ax, label='QCD background', shade=args.kdeplot_fill, clip=(0., dict_upper[var]))
        else:
            #plt.hist(df_QCD[dict[var]['branch']], bins=dict[var]['bins'], alpha=0.3, density=True, label='QCD background', log=True, histtype="stepfilled")
            plt.hist(df_QCD[dict[var]['branch']], bins=dict[var]['bins'], density=True, label='QCD background', log=True, histtype="step", linewidth=linewidth, zorder=0)
    if (args.TTJets and not args.QCD) and var not in signal_only_vars:
        if args.kdeplot and var not in ['NJet', 'NBJet']:
            sns.kdeplot(df_TTJets[dict[var]['branch']], ax=ax, label='$t \overline{t}$ + $jets$ background', shade=args.kdeplot_fill, clip=(0., dict_upper[var]))
        else:
            #plt.hist(df_TTJets[dict[var]['branch']], bins=dict[var]['bins'], alpha=0.3, density=True, label='$t \overline{t}$ + $jets$ background', log=True, histtype="stepfilled")
            plt.hist(df_TTJets[dict[var]['branch']], bins=dict[var]['bins'], density=True, label='$t \overline{t}$ + $jets$ background', log=True, histtype="step", linewidth=linewidth, zorder=5)
    if (args.QCD and args.TTJets) and var not in signal_only_vars:
        if args.kdeplot and var not in ['NJet', 'NBJet']:
            sns.kdeplot(df_QCD[dict[var]['branch']], ax=ax, label='QCD background', shade=args.kdeplot_fill, clip=(0., dict_upper[var]))
        else:
            #plt.hist(df_QCD[dict[var]['branch']], bins=dict[var]['bins'], alpha=0.3, density=True, label='QCD + $t \overline{t}$ background', log=True, histtype="stepfilled")
            plt.hist(df_QCD[dict[var]['branch']], bins=dict[var]['bins'], density=True, label='QCD + $t \overline{t}$ background', log=True, histtype="step", linewidth=linewidth, hatch="\\\\", zorder=5)

    if args.MSSM and var not in signal_only_vars:
        label='MSSM-like: $M_{\mathrm{Squark}}$ = ' + str(df_MSSM["M_sq"][0]) + ', $M_{\mathrm{LSP}}$ = ' + str(df_MSSM["M_lsp"][0])
        if args.kdeplot and var not in ['NJet', 'NBJet']:
            sns.kdeplot(df_MSSM[dict[var]['branch']], ax=ax, label=label, shade=args.kdeplot_fill, clip=(0., dict_upper[var]))
        else:
            #plt.hist(df_MSSM[dict[var]['branch']], bins=dict[var]['bins'], alpha=0.6, density=True, label=label, log=True, histtype="stepfilled")
            plt.hist(df_MSSM[dict[var]['branch']], bins=dict[var]['bins'], density=True, label=label, log=True, histtype="step", linewidth=linewidth, hatch="//", zorder=10)


    temp_i = 0
    for index, row in df_sig_masses.iterrows():
        temp_i += 5
        label='$M_{\mathrm{Squark}}$ = ' + str(row["M_sq"]) + ', $M_{\mathrm{LSP}}$ = ' + str(row["M_lsp"])
        df_temp = df_sig.loc[(df_sig['M_sq'] == row['M_sq']) & (df_sig['M_lsp'] == row['M_lsp'])]
        if args.kdeplot and var not in ['NJet', 'NBJet']:
            sns.kdeplot(df_temp[dict[var]['branch']], ax=ax, label=label, shade=args.kdeplot_fill, clip=(0., dict_upper[var]))
        else:
            #plt.hist(df_temp[dict[var]['branch']], bins=dict[var]['bins'], alpha=0.8, density=True, label=label, log=True, histtype="stepfilled")
            plt.hist(df_temp[dict[var]['branch']], bins=dict[var]['bins'], density=True, label=label, log=True, histtype="step", linewidth=linewidth, hatch="+++", zorder=35-temp_i)

    plt.xlabel(dict[var]['title'])
    handles, labels = ax.get_legend_handles_labels()
    plt.legend(handles[::-1], labels[::-1], loc='best', fontsize='medium')
    if var not in ['NJet', 'NBJet']:
        plt.ylim(0.0001, None)
        plt.xlim(0., None)
    if not args.NoOutput:
        plt.savefig(os.path.join(temp_dir, var + '.pdf'))
        print('Saved ' + var + '.pdf output file')
    if not args.NoX:
        plt.show()