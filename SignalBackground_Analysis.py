#!/usr/bin/env python
import pandas as pd
import dask.dataframe as dd
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
parser.add_argument('-s', '--signal', default=None, nargs='*', help='Path to signal dataframe file(s) from ROOTCuts')
parser.add_argument('-q', '--QCD', default=None, nargs='*', help='Path to QCD dataframe file(s) from ROOTCuts')
parser.add_argument('-m', '--MSSM', default=None, nargs='*', help='Path to MSSM dataframe file(s) from ROOTCuts')
parser.add_argument('-t', '--TTJets', default=None, nargs='*', help='Path to TTJets dataframe file(s) from ROOTCuts')
parser.add_argument('-d', '--Data', default=None, nargs='*', help='Path to Data dataframe file(s) from ROOTCuts')
parser.add_argument('-l', '--Lumi', type=float, default=35900., help='Luminosity in pb-1')
parser.add_argument('--HT_cut', type=float, default=None, help='Apply minimum HT cut')
parser.add_argument('--DBT', type=float, default=None, help='Apply minimum DBT score cut (when no Data samples)')
parser.add_argument('--norm', action='store_true', help='Normalise each histogram')
parser.add_argument('--stackBKG', action='store_true', help='Stack the BKG histos')
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
print('Luminosity = {0}fb-1'.format(args.Lumi/1000.))

# Memory usage of pandas thing
def mem_usage(pandas_obj):
    if isinstance(pandas_obj,pd.DataFrame):
        usage_b = pandas_obj.memory_usage(deep=True).sum()
    else: # we assume if not a df it's a series
        usage_b = pandas_obj.memory_usage(deep=True)
    usage_mb = usage_b / 1024 ** 2 # convert bytes to megabytes
    return "{:03.2f} MB".format(usage_mb)


if args.Data:
    variables = ['MHT', 'HT', 'NJet', 'nMuons', 'Muon_MHT_TransMass', 'Muons_InvMass', 'LeadSlimJet_Pt']
    types = {'MHT': np.float32,
             'HT': np.float32,
             'NJet': np.int8,
             'nMuons': np.int8,
             'Muon_MHT_TransMass': np.float32,
             'Muons_InvMass': np.float32,
             'LeadSlimJet_Pt': np.float32,
             'crosssec': np.float32,
            }
else:
    variables = ['MHT', 'HT', 'FatJetAngularSeparation', 'NJet', 'NFatJet', 'NBJet', 'NDoubleBJet', 'MaxFatJetDoubleB_discrim', 'FatJet_MaxDoubleB_discrim_mass', 'nMuons', 'Muon_MHT_TransMass', 'Muons_InvMass', 'LeadSlimJet_Pt']
    types = {'MHT': np.float32,
             'HT': np.float32,
             'NJet': np.int8,
             'NFatJet': np.int8,
             'NBJet': np.int8,
             'NDoubleBJet': np.int8,
             'MaxFatJetDoubleB_discrim': np.float32,
             'FatJet_MaxDoubleB_discrim_mass': np.float32,
             'nMuons': np.int8,
             'Muon_MHT_TransMass': np.float32,
             'Muons_InvMass': np.float32,
             'LeadSlimJet_Pt': np.float32,
             'crosssec': np.float32,
            }

columns = variables
columns.append('crosssec')

# Read in the dataframes:
if args.signal:
    df_sig = dd.read_csv(args.signal, delimiter=r'\s+', usecols=columns, dtype=types)
    if args.verbose:
        print('Signal:')
        print(df_sig)
    sigweight = args.Lumi/float(df_sig.shape[0])
    df_sig_masses = df_sig[['M_sq', 'M_lsp']].drop_duplicates().compute()
    df_sig_masses = df_sig_masses.sort_values(by=['M_sq', 'M_lsp'])
    print(df_sig_masses.head())
    if args.HT_cut:
        df_sig = df_sig.loc[(df_sig['HT'] > args.HT_cut)]
    if args.DBT and not args.Data:
        df_sig = df_sig.loc[(df_sig['MaxFatJetDoubleB_discrim'] > args.DBT)]
    print('Signal df read, memory used: {0}'.format(mem_usage(df_sig)))

if args.MSSM:
    df_MSSM = dd.read_csv(args.MSSM, delimiter=r'\s+', usecols=columns, dtype=types)
    MSSMweight = args.Lumi/float(df_MSSM.shape[0])
    if args.HT_cut:
        df_MSSM = df_MSSM.loc[(df_MSSM['HT'] > args.HT_cut)]
    if args.DBT and not args.Data:
        df_MSSM = df_MSSM.loc[(df_MSSM['MaxFatJetDoubleB_discrim'] > args.DBT)]
    if args.verbose:
        print('MSSM:')
        print(df_MSSM)
    print('MSSM df read, memory used: {0}'.format(mem_usage(df_MSSM)))

if args.QCD:
    df_QCD = dd.read_csv(args.QCD, delimiter=r'\s+', usecols=columns, dtype=types)
    QCDweight = args.Lumi/float(df_QCD.shape[0])
    if args.HT_cut:
        df_QCD = df_QCD.loc[(df_QCD['HT'] > args.HT_cut)]
    if args.DBT and not args.Data:
        df_QCD = df_QCD.loc[(df_QCD['MaxFatJetDoubleB_discrim'] > args.DBT)]
    if args.verbose:
        print('QCD:')
        print(df_QCD)
    print('QCD df read, memory used: {0}'.format(mem_usage(df_QCD)))

if args.TTJets:
    df_TTJets = dd.read_csv(args.TTJets, delimiter=r'\s+', usecols=columns, dtype=types)
    TTJetsweight = args.Lumi/float(df_TTJets.shape[0])
    if args.HT_cut:
        df_TTJets = df_TTJets.loc[(df_TTJets['HT'] > args.HT_cut)]
    if args.DBT and not args.Data:
        df_TTJets = df_TTJets.loc[(df_TTJets['FatDoubleBJetA_discrim'] > args.DBT)]
    if args.verbose:
        print('TTJets:')
        print(df_TTJets)
    print('TTJets df read, memory used: {0}'.format(mem_usage(df_TTJets)))

if args.Data:
    df_Data = dd.read_csv(args.Data, delimiter=r'\s+', usecols=columns, dtype=types)
    if args.HT_cut:
        df_Data = df_Data.loc[(df_Data['HT'] > args.HT_cut)]
    if args.verbose:
        print('Data:')
        print(df_Data)
    print('Data df read, memory used: {0}'.format(mem_usage(df_Data)))

#Make the output directories
directory = 'Signal_vs_Background_Analysis'
if args.HT_cut:
    directory = directory + '_HT{0}'.format(args.HT_cut)
if args.DBT and not args.Data:
    directory = directory + '_DBT{0}'.format(args.DBT)
temp_dir = directory
suffix = 1
while os.path.exists(temp_dir):
    suffix += 1
    temp_dir = directory + '_{0}'.format(suffix)
if not args.NoOutput:
    print('Files will be written to: {0}'.format(temp_dir))
    os.makedirs(temp_dir)

# Save original command for later
commandString = ' '.join(sys.argv[0:])
print(commandString)
if not args.NoOutput:
    f = open(os.path.join(directory, 'command.txt'), 'w')
    f.write(commandString)
    f.close()






bins_HT = np.linspace(0.,5000.,50)
bins_MHT = np.linspace(0.,1000.,50)
bins_DelR = np.linspace(0.,5.,50)
bins_njet = np.arange(0, 20, 1)
bins_nfatjet = np.arange(0, 8, 1)
bins_nbjet = np.arange(0, 14, 1)
bins_nDoubleB = np.arange(0, 2, 1)
bins_Mbb = np.linspace(0., 200., 40)
bins_DBT = np.linspace(-1., 1., 50)
bins_nMuons = np.arange(0, 6, 1)
bins_muon_transMass = np.linspace(0., 400., 50)

dict = {'MHT': {'branch': 'MHT', 'bins': bins_MHT, 'title': 'Missing $H_{T}$ [GeV/$c$]'},
        'HT': {'branch': 'HT', 'bins': bins_HT, 'title': 'Total $H_{T}$ [GeV/$c$]'},
        'FatJetAngularSeparation': {'branch': 'FatJetAngularSeparation', 'bins': bins_DelR, 'title': 'AK8 Jets $\Delta R$'},
        'NJet': {'branch': 'NJet', 'bins': bins_njet, 'title': 'Number of Jets'},
        'NFatJet': {'branch': 'NFatJet', 'bins': bins_nfatjet, 'title': 'Number of AK8 FatJets'},
        'NBJet': {'branch': 'NBJet', 'bins': bins_nbjet, 'title': 'Number of $b$-tagged Jets'},
        'NDoubleBJet': {'branch': 'NDoubleBJet', 'bins': bins_nDoubleB, 'title': 'Number of double-$b$-tagged AK8 Jets'},
        'MaxFatJetDoubleB_discrim': {'branch': 'MaxFatJetDoubleB_discrim', 'bins': bins_DBT, 'title': 'AK8 Fat Jet Double-$b$-tag score'},
        'FatJet_MaxDoubleB_discrim_mass': {'branch': 'FatJet_MaxDoubleB_discrim_mass', 'bins': bins_Mbb, 'title': 'AK8 SoftDrop Mass [GeV/$c^{2}$]'},
        'nMuons': {'branch': 'nMuons', 'bins': bins_nMuons, 'title': 'Number of isolated Muons'},
        'Muon_MHT_TransMass': {'branch': 'Muon_MHT_TransMass', 'bins': bins_muon_transMass, 'title': 'Muon-Missing $H_{T}$ Transverse Mass [GeV/$c^{2}$]'},
        'Muons_InvMass': {'branch': 'Muons_InvMass', 'bins': bins_muon_transMass, 'title': "Di-Muon Invariant Mass [GeV/$c^{2}$]"},
        'LeadSlimJet_Pt': {'branch': 'LeadSlimJet_Pt', 'bins': bins_MHT, 'title': "Lead Jet $p_{T}$ [GeV/$c$]"},
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

linewidth = 3.

for var in variables:
    plt.figure()
    f, ax = plt.subplots()
    if var not in ['NJet', 'NBJet']:
        ax.set(yscale="log")

    temp_i = 0
    if args.signal:
        for index, row in df_sig_masses.iterrows():
            temp_i += 5
            label='$M_{\mathrm{Squark}}$ = ' + str(row["M_sq"]) + ', $M_{\mathrm{LSP}}$ = ' + str(row["M_lsp"])
            df_temp = df_sig.loc[(df_sig['M_sq'] == row['M_sq']) & (df_sig['M_lsp'] == row['M_lsp'])]
            if args.norm:
                plt.hist(df_temp[dict[var]['branch']], bins=dict[var]['bins'], weights=sigweight*df_temp['crosssec'], label=label, log=True, normed=1., histtype="step", linewidth=linewidth, zorder=35-temp_i)
            else:
                #plt.hist(df_temp[dict[var]['branch']], bins=dict[var]['bins'], alpha=0.8, density=True, label=label, log=True, histtype="stepfilled")
                plt.hist(df_temp[dict[var]['branch']], bins=dict[var]['bins'], weights=sigweight*df_temp['crosssec'], label=label, log=True, histtype="step", linewidth=linewidth, zorder=35-temp_i)

    if args.MSSM:
        label='MSSM-like: $M_{\mathrm{Squark}}$ = ' + str(df_MSSM["M_sq"][0]) + ', $M_{\mathrm{LSP}}$ = ' + str(df_MSSM["M_lsp"][0])
        if args.norm:
            plt.hist(df_MSSM[dict[var]['branch']], bins=dict[var]['bins'], weights=MSSMweight*df_MSSM['crosssec'], label=label, log=True, normed=1., histtype="step", linewidth=linewidth, zorder=10)
        else:
            #plt.hist(df_MSSM[dict[var]['branch']], bins=dict[var]['bins'], alpha=0.6, density=True, label=label, log=True, histtype="stepfilled")
            plt.hist(df_MSSM[dict[var]['branch']], bins=dict[var]['bins'], weights=MSSMweight*df_MSSM['crosssec'], label=label, log=True, histtype="step", linewidth=linewidth, zorder=10)

    if args.QCD and args.TTJets and args.stackBKG:
        if args.norm:
            plt.hist([df_QCD[dict[var]['branch']], df_TTJets[dict[var]['branch']]], bins=dict[var]['bins'], alpha=1., weights=[QCDweight*df_QCD['crosssec'], TTJetsweight*df_TTJets['crosssec']], label=['QCD background', '$t \overline{t}$ + $jets$ background'], stacked=True, log=True, normed=1., histtype="stepfilled", zorder=5)
        else:
            plt.hist([df_QCD[dict[var]['branch']], df_TTJets[dict[var]['branch']]], bins=dict[var]['bins'], alpha=1., weights=[QCDweight*df_QCD['crosssec'], TTJetsweight*df_TTJets['crosssec']], label=['QCD background', '$t \overline{t}$ + $jets$ background'], stacked=True, log=True, histtype="stepfilled", zorder=5)
            #plt.hist(df_QCD[dict[var]['branch']], bins=dict[var]['bins'], density=True, label='QCD background', log=True, histtype="step", linewidth=linewidth, hatch="xx", zorder=0)
    else:
        if args.QCD:
            if args.norm:
                plt.hist(df_QCD[dict[var]['branch']], bins=dict[var]['bins'], alpha=0.7, weights=QCDweight*df_QCD['crosssec'], label='QCD background', log=True, normed=1., histtype="stepfilled", zorder=5)
            else:
                plt.hist(df_QCD[dict[var]['branch']], bins=dict[var]['bins'], alpha=0.7, weights=QCDweight*df_QCD['crosssec'], label='QCD background', log=True, histtype="stepfilled", zorder=5)
                #plt.hist(df_QCD[dict[var]['branch']], bins=dict[var]['bins'], density=True, label='QCD background', log=True, histtype="step", linewidth=linewidth, hatch="xx", zorder=0)
        if args.TTJets:
            if args.norm:
                plt.hist(df_TTJets[dict[var]['branch']], bins=dict[var]['bins'], alpha=1., weights=TTJetsweight*df_TTJets['crosssec'], normed=1., label='$t \overline{t}$ + $jets$ background', log=True, histtype="stepfilled", zorder=0)
            else:
                plt.hist(df_TTJets[dict[var]['branch']], bins=dict[var]['bins'], alpha=1., weights=TTJetsweight*df_TTJets['crosssec'], label='$t \overline{t}$ + $jets$ background', log=True, histtype="stepfilled", zorder=0)
                #plt.hist(df_TTJets[dict[var]['branch']], bins=dict[var]['bins'], density=True, label='$t \overline{t}$ + $jets$ background', log=True, histtype="step", linewidth=linewidth, hatch="xx", zorder=5)

    if args.Data:
        if args.norm:
            plt.hist(df_Data[dict[var]['branch']], bins=dict[var]['bins'], normed=1., label='Data', log=True, histtype="step", zorder=35)
        else:
            plt.hist(df_Data[dict[var]['branch']], bins=dict[var]['bins'], label='Data', log=True, histtype="step", zorder=35)
            #plt.hist(df_Data[dict[var]['branch']], bins=dict[var]['bins'], density=True, label='$t \overline{t}$ + $jets$ background', log=True, histtype="step", linewidth=linewidth, hatch="xx", zorder=5)


    plt.xlabel(dict[var]['title'], size=14)
    leg = plt.legend(loc='upper right', fontsize='medium')
    leg.set_zorder(100)
    if var in ['NDoubleBJet']:
        continue
    elif var in ['FatDoubleBJetA_discrim']:
        plt.ylim(0.05, None)
    elif var not in ['NJet', 'NBJet']:
        plt.ylim(0.0001, None)
        plt.xlim(0., None)
    else:
        plt.ylim(0.000005, None)
    if not args.NoOutput:
        plt.savefig(os.path.join(temp_dir, var + '.pdf'))
        print('Saved ' + var + '.pdf output file')
    if not args.NoX:
        plt.show()
