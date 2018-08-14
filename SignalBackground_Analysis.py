#!/usr/bin/env python
import pandas as pd
import dask.dataframe as ddd
from histbook import Hist, bin, beside
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
from vega import VegaLite as canvas
import vegascope
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
             'NJet': np.uint8,
             'nMuons': np.uint8,
             'Muon_MHT_TransMass': np.float32,
             'Muons_InvMass': np.float32,
             'LeadSlimJet_Pt': np.float32,
             'crosssec': np.float32,
             'NoEntries': np.uint32,
            }
else:
    variables = ['MHT', 'HT', 'FatJetAngularSeparation', 'NJet', 'NFatJet', 'NBJet', 'NDoubleBJet', 'MaxFatJetDoubleB_discrim', 'FatJet_MaxDoubleB_discrim_mass', 'nMuons', 'Muon_MHT_TransMass', 'Muons_InvMass', 'LeadSlimJet_Pt']
    types = {'MHT': np.float32,
             'HT': np.float32,
             'NJet': np.uint8,
             'NFatJet': np.uint8,
             'NBJet': np.uint8,
             'NDoubleBJet': np.uint8,
             'MaxFatJetDoubleB_discrim': np.float32,
             'FatJet_MaxDoubleB_discrim_mass': np.float32,
             'nMuons': np.uint8,
             'Muon_MHT_TransMass': np.float32,
             'Muons_InvMass': np.float32,
             'LeadSlimJet_Pt': np.float32,
             'crosssec': np.float32,
             'NoEntries': np.float32,
            }

columns = variables
columns.append('crosssec')
columns.append('NoEntries')

# Read in the dataframes:
if args.signal:
    df_sig = dd.read_csv(args.signal, delimiter=r'\s+', usecols=columns, dtype=types)
    df_sig['weight'] = args.Lumi*df_sig['crosssec']/df_sig['NoEntries']
    if args.verbose:
        print('Signal:')
        print(df_sig)
    df_sig_masses = df_sig[['M_sq', 'M_lsp']].drop_duplicates().compute()
    df_sig_masses = df_sig_masses.sort_values(by=['M_sq', 'M_lsp'])
    print(df_sig_masses.head())
    if args.HT_cut:
        df_sig = df_sig.loc[(df_sig['HT'] > args.HT_cut)]
    if args.DBT and not args.Data:
        df_sig = df_sig.loc[(df_sig['MaxFatJetDoubleB_discrim'] > args.DBT)]
    #print('Signal df read, memory used: {0}'.format(mem_usage(df_sig)))

if args.MSSM:
    df_MSSM = dd.read_csv(args.MSSM, delimiter=r'\s+', usecols=columns, dtype=types)
    df_MSSM['weight'] = args.Lumi*df_MSSM['crosssec']/df_MSSM['NoEntries']
    if args.HT_cut:
        df_MSSM = df_MSSM.loc[(df_MSSM['HT'] > args.HT_cut)]
    if args.DBT and not args.Data:
        df_MSSM = df_MSSM.loc[(df_MSSM['MaxFatJetDoubleB_discrim'] > args.DBT)]
    if args.verbose:
        print('MSSM:')
        print(df_MSSM)
    #print('MSSM df read, memory used: {0}'.format(mem_usage(df_MSSM)))

if args.QCD:
    df_QCD = dd.read_csv(args.QCD, delimiter=r'\s+', usecols=columns, dtype=types)
    df_QCD['weight'] = args.Lumi*df_QCD['crosssec']/df_QCD['NoEntries']
    if args.HT_cut:
        df_QCD = df_QCD.loc[(df_QCD['HT'] > args.HT_cut)]
    if args.DBT and not args.Data:
        df_QCD = df_QCD.loc[(df_QCD['MaxFatJetDoubleB_discrim'] > args.DBT)]
    if args.verbose:
        print('QCD:')
        print(df_QCD)
    #print('QCD df read, memory used: {0}'.format(mem_usage(df_QCD)))

if args.TTJets:
    df_TTJets = dd.read_csv(args.TTJets, delimiter=r'\s+', usecols=columns, dtype=types)
    df_TTJets['weight'] = args.Lumi*df_TTJets['crosssec']/df_TTJets['NoEntries']
    if args.HT_cut:
        df_TTJets = df_TTJets.loc[(df_TTJets['HT'] > args.HT_cut)]
    if args.DBT and not args.Data:
        df_TTJets = df_TTJets.loc[(df_TTJets['FatDoubleBJetA_discrim'] > args.DBT)]
    if args.verbose:
        print('TTJets:')
        print(df_TTJets)
    #print('TTJets df read, memory used: {0}'.format(mem_usage(df_TTJets)))

if args.Data:
    df_Data = dd.read_csv(args.Data, delimiter=r'\s+', usecols=columns, dtype=types)
    if args.HT_cut:
        df_Data = df_Data.loc[(df_Data['HT'] > args.HT_cut)]
    if args.verbose:
        print('Data:')
        print(df_Data)
    #print('Data df read, memory used: {0}'.format(mem_usage(df_Data)))

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


dict = {'MHT': {'bins': bin('MHT', 50, 0., 1000.), 'title': 'Missing $H_{T}$ [GeV/$c$]'},
        'HT': {'bins': bin('HT', 50, 0., 5000.), 'title': 'Total $H_{T}$ [GeV/$c$]'},
        'FatJetAngularSeparation': {'bins': bin('FatJetAngularSeparation', 50, 0., 5.), 'title': 'AK8 Jets $\Delta R$'},
        'NJet': {'bins': bin('NJet', 20, 0, 20), 'title': 'Number of Jets'},
        'NFatJet': {'bins': bin('NFatJet', 8, 0, 8), 'title': 'Number of AK8 FatJets'},
        'NBJet': {'bins': bin('NBJet', 14, 0, 14), 'title': 'Number of $b$-tagged Jets'},
        'NDoubleBJet': {'bins': bin('NDoubleBJet', 3, 0, 3), 'title': 'Number of double-$b$-tagged AK8 Jets'},
        'MaxFatJetDoubleB_discrim': {'bins': bin('MaxFatJetDoubleB_discrim', 50, -1., 1,), 'title': 'AK8 Fat Jet Double-$b$-tag score'},
        'FatJet_MaxDoubleB_discrim_mass': {'bins': bin('FatJet_MaxDoubleB_discrim_mass', 50, 0., 500.), 'title': 'AK8 SoftDrop Mass [GeV/$c^{2}$]'},
        'nMuons': {'bins': bin('nMuons', 6, 0, 6), 'title': 'Number of isolated Muons'},
        'Muon_MHT_TransMass': {'bins': bin('Muon_MHT_TransMass', 50, 0., 400.), 'title': 'Muon-Missing $H_{T}$ Transverse Mass [GeV/$c^{2}$]'},
        'Muons_InvMass': {'bins': bin('Muons_InvMass', 50, 0., 400.), 'title': "Di-Muon Invariant Mass [GeV/$c^{2}$]"},
        'LeadSlimJet_Pt': {'bins': bin('LeadSlimJet_Pt', 50, 0., 1000.), 'title': "Lead Jet $p_{T}$ [GeV/$c$]"},
        }

linewidth = 3.

for var in variables:

    canvas = vegascope.LocalCanvas()
    theHists = []
    hists = {}
    if args.signal:
        for index, row in df_sig_masses.iterrows():
            label='$M_{\mathrm{Squark}}$ = ' + str(row["M_sq"]) + ', $M_{\mathrm{LSP}}$ = ' + str(row["M_lsp"])
            df_temp = df_sig.loc[(df_sig['M_sq'] == row['M_sq']) & (df_sig['M_lsp'] == row['M_lsp'])]
            df_plot = df_temp[var, 'weight']
            df_plot = df_plot.compute()
            h = Hist(dict[var]['bin'], weight='weight')
            h.fill(df_plot)
            hists[label] = h
        group = Hist.group(by='source', **hists)
        theHists.append(group.overlay('sample').step(var))
        hists = {}

    if args.MSSM:
        label='MSSM-like: $M_{\mathrm{Squark}}$ = ' + str(df_MSSM["M_sq"][0]) + ', $M_{\mathrm{LSP}}$ = ' + str(df_MSSM["M_lsp"][0])
        df_plot = df_MSSM[var, 'weight']
        df_plot = df_plot.compute()
        h = Hist(dict[var]['bin'], weight='weight')
        h.fill(df_plot)
        hists[label] = h
        group = Hist.group(by='source', **hists)
        theHists.append(group.overlay('sample').step(var))
        hists = {}

    if args.QCD and args.TTJets and args.stackBKG:
        label='QCD background'
        label2='$t \overline{t}$ + $jets$ background'
        df_plot = df_QCD[dict[var]['branch']]
        df_plot = df_plot.compute()
        df_plot2 = df_TTJets[dict[var]['branch']]
        df_plot2 = df_plot2.compute()
        h = Hist(dict[var]['bin'], weight='weight')
        h2 = Hist(dict[var]['bin'], weight='weight')
        h.fill(df_plot)
        h2.fill(df_plot2)
        hists[label] = h
        hists[label2] = h2
        group = Hist.group(by='source', **hists)
        theHists.append(group.stack('sample').area(var))
        hists = {}
    else:
        if args.QCD:
            label='QCD background'
            df_plot = df_QCD[dict[var]['branch']]
            df_plot = df_plot.compute()
            h = Hist(dict[var]['bin'], weight='weight')
            h.fill(df_plot)
            hists[label] = h
            group = Hist.group(by='source', **hists)
            theHists.append(group.stack('sample').area(var))
            hists = {}
        if args.TTJets:
            label='$t \overline{t}$ + $jets$ background'
            df_plot = df_TTJets[dict[var]['branch']]
            df_plot = df_plot.compute()
            h = Hist(dict[var]['bin'], weight='weight')
            h.fill(df_plot)
            hists[label] = h
            group = Hist.group(by='source', **hists)
            theHists.append(group.stack('sample').area(var))
            hists = {}

    if args.Data:
        label='Data'
        df_plot = df_Data[dict[var]['branch']]
        df_plot = df_plot.compute()
        h = Hist(dict[var]['bin'])
        h.fill(df_plot)
        hists[label] = h
        group = Hist.group(by='source', **hists)
        theHists.append(group.overlay('sample').marker(var))
        hists = {}

    overlay((*theHists)).to(canvas)