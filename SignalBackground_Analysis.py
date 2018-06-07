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

parser = a.ArgumentParser(description='CMS Analysis Signal vs Background plot')
parser.add_argument('-s', '--signal', nargs='*', required=True, help='Path to signal dataframe file(s) from ROOTCuts')
parser.add_argument('-q', '--QCD', default=None, nargs='*', help='Path to QCD dataframe file from ROOTCuts')
parser.add_argument('-m', '--MSSM', default=None, help='Path to MSSM dataframe file from ROOTCuts')
parser.add_argument('-t', '--TTJets', default=None, help='Path to TTJets dataframe file from ROOTCuts')
parser.add_argument('--HT_cut', type=float, default=None, help='Apply minimum HT cut')
parser.add_argument('--DBT', type=float, default=None, help='Apply minimum DBT score cut')
parser.add_argument('--norm', action='store_true', help='Normalise each histogram')
parser.add_argument('-x', '--NoX', action='store_true', help='This argument suppresses showing plots via X-forwarding')
parser.add_argument('-o', '--NoOutput', action='store_true', help='This argument suppresses the output of PDF plots')
parser.add_argument('-v', '--verbose', action='store_true', help='Increased verbosity level')
parser.add_argument('--style', default=None, help='Optional drawing style, e.g. \"ggplot\" in Matplotlib or \"dark\" in Seaborn')
parser.add_argument('--kdeplot', action='store_true', help='Use kdeplot in Seaborn instead of matplotlib histogram')
parser.add_argument('--kdeplot_fill', action='store_true', help='Same as --kdeplot but area under each line is filled')
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
directory = 'Signal_vs_Background_Analysis'
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
if args.DBT:
    df_sig = df_sig.loc[(df_sig['FatDoubleBJetA_discrim'] > args.DBT)]

if args.MSSM:
    df_MSSM = pd.read_csv(args.MSSM, delimiter=r'\s+')
    if args.HT_cut:
        df_MSSM = df_MSSM.loc[(df_MSSM['HT'] > args.HT_cut)]
    if args.DBT:
        df_MSSM = df_MSSM.loc[(df_MSSM['FatDoubleBJetA_discrim'] > args.DBT)]
    if args.verbose:
        print('MSSM:')
        print(df_MSSM)

if args.QCD:
    df_list = []
    for file in args.QCD:
        df = pd.read_csv(file, delimiter=r'\s+')
        df_list.append(df)
    df_QCD = pd.concat(df_list)
    if args.HT_cut:
        df_QCD = df_QCD.loc[(df_QCD['HT'] > args.HT_cut)]
    if args.DBT:
        df_QCD = df_QCD.loc[(df_QCD['FatDoubleBJetA_discrim'] > args.DBT)]
    if args.verbose:
        print('QCD:')
        print(df_QCD)
if args.TTJets:
    df_TTJets = pd.read_csv(args.TTJets, delimiter=r'\s+')
    if args.HT_cut:
        df_TTJets = df_TTJets.loc[(df_TTJets['HT'] > args.HT_cut)]
    if args.DBT:
        df_TTJets = df_TTJets.loc[(df_TTJets['FatDoubleBJetA_discrim'] > args.DBT)]
    if args.verbose:
        print('TTJets:')
        print(df_TTJets)

bins_HT = np.linspace(0.,5000.,50)
bins_MHT = np.linspace(0.,2500.,25)
bins_DelR = np.linspace(0.,5.,50)
bins_njet = np.arange(0, 20, 1)
bins_nbjet = np.arange(0, 14, 1)
bins_nDoubleB = np.arange(0, 2, 1)
bins_Mbb = np.linspace(0., 200., 40)
bins_DBT = np.linspace(-1., 1., 50)

dict = {'MHT': {'branch': 'MHT', 'bins': bins_MHT, 'title': 'Missing $H_{T}$ [GeV/$c$]'},
        'HT': {'branch': 'HT', 'bins': bins_HT, 'title': 'Total $H_{T}$ [GeV/$c$]'},
        'FatJetAngularSeparation': {'branch': 'FatJetAngularSeparation', 'bins': bins_DelR, 'title': 'AK8 Jets $\Delta R$'},
        'NJet': {'branch': 'NJet', 'bins': bins_njet, 'title': 'Number of Jets'},
        'NBJet': {'branch': 'NBJet', 'bins': bins_nbjet, 'title': 'Number of $b$-tagged Jets'},
        'NDoubleBJet': {'branch': 'NDoubleBJet', 'bins': bins_BDP, 'title': '$\Delta\Phi^{*}$'},
        'FatDoubleBJetA_discrim': {'branch': 'FatDoubleBJetA_discrim', 'bins': bins_DBT, 'title': 'AK8 Fat Jet Double-$b$-tag score'},
        'FatDoubleBJetA_mass': {'branch': 'FatDoubleBJetA_mass', 'bins': bins_Mbb, 'title': 'AK8 SoftDrop Mass [GeV/$c^{2}$]'},
        }

dict_upper = {'MHT': 2000.,
       	      'HT': 5000.,
       	      'FatJetAngularSeparation': 5.,
       	      'NJet': 20.,
       	      'NBJet': 14.,
       	      'NDoubleBJet': 2,
              'FatDoubleBJetA_discrim': 1.,
              'FatDoubleBJetA_mass': 200.,
             }

variables = ['MHT', 'HT', 'DelR', 'NJet', 'NBJet', 'NDoubleBJet', 'FatDoubleBJetA_discrim', 'FatDoubleBJetA_mass']

signal_only_vars = ['DelR', 'BMass', '2BMass']

n_signal = len(args.signal)
linewidth = 3.

for var in variables:
    plt.figure()
    f, ax = plt.subplots()
    if var not in ['NJet', 'NBJet']:
        ax.set(yscale="log")

    temp_i = 0
    for index, row in df_sig_masses.iterrows():
        temp_i += 5
        label='$M_{\mathrm{Squark}}$ = ' + str(row["M_sq"]) + ', $M_{\mathrm{LSP}}$ = ' + str(row["M_lsp"])
        df_temp = df_sig.loc[(df_sig['M_sq'] == row['M_sq']) & (df_sig['M_lsp'] == row['M_lsp'])]
        if args.kdeplot and var not in ['NJet', 'NBJet']:
            sns.kdeplot(df_temp[dict[var]['branch']], ax=ax, label=label, shade=args.kdeplot_fill, clip=(0., dict_upper[var]))
        else:
            #plt.hist(df_temp[dict[var]['branch']], bins=dict[var]['bins'], alpha=0.8, density=True, label=label, log=True, histtype="stepfilled")
            plt.hist(df_temp[dict[var]['branch']], bins=dict[var]['bins'], weights=df_temp['crosssec'], density=args.norm, label=label, log=True, histtype="step", linewidth=linewidth, zorder=35-temp_i)


    if args.MSSM and var not in signal_only_vars:
        label='MSSM-like: $M_{\mathrm{Squark}}$ = ' + str(df_MSSM["M_sq"][0]) + ', $M_{\mathrm{LSP}}$ = ' + str(df_MSSM["M_lsp"][0])
        if args.kdeplot and var not in ['NJet', 'NBJet']:
            sns.kdeplot(df_MSSM[dict[var]['branch']], ax=ax, label=label, shade=args.kdeplot_fill, clip=(0., dict_upper[var]))
        else:
            #plt.hist(df_MSSM[dict[var]['branch']], bins=dict[var]['bins'], alpha=0.6, density=True, label=label, log=True, histtype="stepfilled")
            plt.hist(df_MSSM[dict[var]['branch']], bins=dict[var]['bins'], weights=df_MSSM['crosssec'], density=args.norm, label=label, log=True, histtype="step", linewidth=linewidth, zorder=10)

    if (args.QCD) and var not in signal_only_vars:
        if args.kdeplot and var not in ['NJet', 'NBJet']:
            sns.kdeplot(df_QCD[dict[var]['branch']], ax=ax, label='QCD background', shade=args.kdeplot_fill, clip=(0., dict_upper[var]))
        else:
            plt.hist(df_QCD[dict[var]['branch']], bins=dict[var]['bins'], alpha=0.7, weights=df_QCD['crosssec'], density=args.norm, label='QCD background', log=True, histtype="stepfilled", zorder=5)
            #plt.hist(df_QCD[dict[var]['branch']], bins=dict[var]['bins'], density=True, label='QCD background', log=True, histtype="step", linewidth=linewidth, hatch="xx", zorder=0)
    if (args.TTJets) and var not in signal_only_vars:
        if args.kdeplot and var not in ['NJet', 'NBJet']:
            sns.kdeplot(df_TTJets[dict[var]['branch']], ax=ax, label='$t \overline{t}$ + $jets$ background', shade=args.kdeplot_fill, clip=(0., dict_upper[var]))
        else:
            plt.hist(df_TTJets[dict[var]['branch']], bins=dict[var]['bins'], alpha=1., weights=df_TTJets['crosssec'], density=args.norm, label='$t \overline{t}$ + $jets$ background', log=True, histtype="stepfilled", zorder=0)
            #plt.hist(df_TTJets[dict[var]['branch']], bins=dict[var]['bins'], density=True, label='$t \overline{t}$ + $jets$ background', log=True, histtype="step", linewidth=linewidth, hatch="xx", zorder=5)


    plt.xlabel(dict[var]['title'], size=14)
    leg = plt.legend(loc='upper right', fontsize='medium')
    leg.set_zorder(100)
    if var not in ['NJet', 'NBJet']:
        plt.ylim(0.0001, None)
        plt.xlim(0., None)
    else:
        plt.ylim(0.000005, None)
    if not args.NoOutput:
        plt.savefig(os.path.join(temp_dir, var + '.pdf'))
        print('Saved ' + var + '.pdf output file')
    if not args.NoX:
        plt.show()
