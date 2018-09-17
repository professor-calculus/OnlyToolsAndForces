#!/usr/bin/env python
import pandas as pd
from multiprocessing.pool import ThreadPool
import dask
import dask.dataframe as dd
import os
import numpy as np
import sys
import argparse as a
import warnings

#Get Options

parser = a.ArgumentParser(description='CMS Analysis Signal vs Background plot')
parser.add_argument('-s', '--signal', default=None, nargs='*', help='Path to signal dataframe file(s) from ROOTCuts')
parser.add_argument('-q', '--QCD', default=None, nargs='*', help='Path to QCD dataframe file(s) from ROOTCuts')
parser.add_argument('-m', '--MSSM', default=None, nargs='*', help='Path to MSSM dataframe file(s) from ROOTCuts')
parser.add_argument('-t', '--TTJets', default=None, nargs='*', help='Path to TTJets dataframe file(s) from ROOTCuts')
parser.add_argument('-w', '--WJets', default=None, nargs='*', help='Path to W+Jets dataframe file(s) from ROOTCuts')
parser.add_argument('-z', '--ZJets', default=None, nargs='*', help='Path to Z+Jets dataframe file(s) from ROOTCuts')
parser.add_argument('--DiBoson', default=None, nargs='*', help='Path to DiBoson dataframe file(s) from ROOTCuts')
parser.add_argument('--SingleTop', default=None, nargs='*', help='Path to SingleTop dataframe file(s) from ROOTCuts')
parser.add_argument('--TTW', default=None, nargs='*', help='Path to TTW dataframe file(s) from ROOTCuts')
parser.add_argument('--TTZ', default=None, nargs='*', help='Path to TTZ dataframe file(s) from ROOTCuts')
parser.add_argument('-d', '--Data', default=None, nargs='*', help='Path to Data dataframe file(s) from ROOTCuts')
parser.add_argument('-o', '--NoOutput', action='store_true', help='This argument suppresses the output of PDF plots')
parser.add_argument('--Threads', type=int, default=None, help='Optional: Set max number of cores for Dask to use')
parser.add_argument('--region', default='Signal', help='Signal, 0b2mu etc region')
parser.add_argument('--latex', action='store_true', help='Save LaTeX table')
arser.add_argument('--Higgs2bb', action='store_true', help='Insist upon 2 Higgs to bb in SIGNAL at MC truth level')
parser.add_argument('-v', '--verbose', action='store_true', help='Increased verbosity level')
args=parser.parse_args()

if args.Threads:
    dask.set_options(pool=ThreadPool(args.Threads))

if args.verbose:
    parser.print_help()
else:
    warnings.filterwarnings("ignore")

print '\nCutflow Table Maker\n'

variables = ['Type', 'MHT', 'HT', 'NJet', 'NBJet', 'NDoubleBJet', 'nLooseMuons', 'nTightMuons', 'Muon_MHT_TransMass', 'Muons_InvMass']
types = {'MHT': np.float32,
         'HT': np.float32,
         'NJet': np.uint8,
         'NBJet': np.uint8,
         'NDoubleBJet': np.uint8,
         'nLooseMuons': np.uint8,
         'nTightMuons': np.uint8,
         'Muon_MHT_TransMass': np.float32,
         'Muons_InvMass': np.float32,
        }

columns = variables
columns.append('crosssec')
columns.append('M_lsp')
columns.append('M_sq')
if args.Higgs2bb:
    columns.append('nHiggs2bb')

MC_types = []
dataframes = {}

# Read in the dataframes:
if args.signal:
    df_sig = dd.read_csv(args.signal, delimiter=r'\s+', usecols=columns, dtype=types)
    if args.verbose:
        print('Signal:')
        print(df_sig)
    df_sig_masses = df_sig[['Type', 'M_sq', 'M_lsp']].drop_duplicates().compute()
    df_sig_masses = df_sig_masses.sort_values(by=['Type', 'M_sq','M_lsp'])
    print(df_sig_masses.head())
    for index, row in df_sig_masses.iterrows():
        MC_types.append(str(row['Type']))
        dataframes[str(row['Type'])] = df_sig.loc[((df_sig['M_sq'] == row['M_sq']) & (df_sig['M_lsp'] == row['M_lsp']))]

if args.QCD:
    df_QCD = dd.read_csv(args.QCD, delimiter=r'\s+', usecols=columns, dtype=types)
    MC_types.append('QCD')
    dataframes['QCD'] = df_QCD
    if args.verbose:
        print('QCD:')
        print(df_QCD)

if args.TTJets:
    df_TTJets = dd.read_csv(args.TTJets, delimiter=r'\s+', usecols=columns, dtype=types)
    MC_types.append('TTJets')
    dataframes['TTJets'] = df_TTJets
    if args.verbose:
        print('TTJets:')
        print(df_TTJets)
    #print('TTJets df read, memory used: {0}'.format(mem_usage(df_TTJets)))

if args.WJets:
    df_WJets = dd.read_csv(args.WJets, delimiter=r'\s+', usecols=columns, dtype=types)
    MC_types.append('WJets')
    dataframes['WJets'] = df_WJets
    if args.verbose:
        print('WJets:')
        print(df_WJets)
    #print('WJets df read, memory used: {0}'.format(mem_usage(df_WJets)))

if args.ZJets:
    df_ZJets = dd.read_csv(args.ZJets, delimiter=r'\s+', usecols=columns, dtype=types)
    MC_types.append('ZJets')
    dataframes['ZJets'] = df_ZJets
    if args.verbose:
        print('ZJets:')
        print(df_ZJets)
    #print('ZJets df read, memory used: {0}'.format(mem_usage(df_ZJets)))

if args.DiBoson:
    df_DiBoson = dd.read_csv(args.DiBoson, delimiter=r'\s+', usecols=columns, dtype=types)
    MC_types.append('DiBoson')
    dataframes['DiBoson'] = df_DiBoson
    if args.verbose:
        print('DiBoson:')
        print(df_DiBoson)
    #print('DiBoson df read, memory used: {0}'.format(mem_usage(df_DiBoson)))

if args.SingleTop:
    df_SingleTop = dd.read_csv(args.SingleTop, delimiter=r'\s+', usecols=columns, dtype=types)
    MC_types.append('SingleTop')
    dataframes['SingleTop'] = df_SingleTop
    if args.verbose:
        print('SingleTop:')
        print(df_SingleTop)
    #print('SingleTop df read, memory used: {0}'.format(mem_usage(df_SingleTop)))

if args.TTW:
    df_TTW = dd.read_csv(args.TTW, delimiter=r'\s+', usecols=columns, dtype=types)
    MC_types.append('TTW')
    dataframes['TTW'] = df_TTW
    if args.verbose:
        print('TTW:')
        print(df_TTW)
    #print('TTW df read, memory used: {0}'.format(mem_usage(df_TTW)))

if args.TTZ:
    df_TTZ = dd.read_csv(args.TTZ, delimiter=r'\s+', usecols=columns, dtype=types)
    MC_types.append('TTZ')
    dataframes['TTZ'] = df_TTZ
    if args.verbose:
        print('TTZ:')
        print(df_TTZ)
    #print('TTZ df read, memory used: {0}'.format(mem_usage(df_TTZ)))

if args.Data:
    df_Data = dd.read_csv(args.Data, delimiter=r'\s+', usecols=columns, dtype=types)
    MC_types.append('Data')
    dataframes['Data'] = df_Data
    if args.verbose:
        print('Data:')
        print(df_Data)
    #print('Data df read, memory used: {0}'.format(mem_usage(df_Data)))

#Make the output directories
directory = 'Cutflow_Table_{0}Region'.format(args.region)
temp_dir = directory
suffix = 1
while os.path.exists(temp_dir):
    suffix += 1
    temp_dir = directory + '_{0}'.format(suffix)
if not args.NoOutput:
    print('Files will be written to: {0}'.format(temp_dir))
    os.makedirs(temp_dir)
directory = temp_dir

# Save original command for later
commandString = ' '.join(sys.argv[0:])
print(commandString)
if not args.NoOutput:
    f = open(os.path.join(directory, 'command.txt'), 'w')
    f.write(commandString)
    f.close()

theDataframe = pd.DataFrame()
if args.region == 'Signal':
    theDataframe['Cuts'] = ['HT > 1500GeV', 'MHT > 200GeV', 'Number of Jets > 5', 'Muon Veto', 'ge3b_0double-b', 'ge2b_1double-b', '2double-b']
else:
    theDataframe['Cuts'] = ['HT > 1500GeV', 'MHT > 200GeV', 'Number of Jets > 5', 'Muon Selection', 'ge3b_0double-b', 'ge2b_1double-b', '2double-b']

for thing in MC_types:
    temp_efficiencies = []
    df_temp = dataframes[thing]

    # If requiring h->bb in Signal sample:
    if args.Higgs2bb:
        df_temp = df_temp.loc[(df_temp['nHiggs2bb'] == 2)]

    # Entries weighted by cross-section
    nentries = float(df_temp['crosssec'].compute().sum())

    # HT
    df_temp = df_temp.loc[(df_temp['HT'] > 1500.)]
    temp_efficiencies.append(float(df_temp['crosssec'].compute().sum())/nentries)

    # MHT
    df_temp = df_temp.loc[(df_temp['MHT'] > 200.)]
    temp_efficiencies.append(float(df_temp['crosssec'].compute().sum())/nentries)

    # NJets
    df_temp = df_temp.loc[(df_temp['NJet'] > 5)]
    temp_efficiencies.append(float(df_temp['crosssec'].compute().sum())/nentries)

    # Muon veto/selection
    if args.region == 'Signal':
        df_temp = df_temp.loc[(df_temp['nLooseMuons'] == 0)]
    elif args.region == '1mu':
        df_temp = df_temp.loc[(df_temp['nTightMuons'] == 1)]
    elif args.region == '0b1mu':
        df_temp = df_temp.loc[((df_temp['nTightMuons'] == 1) & (df_temp['NBJet'] == 0))]
    elif args.region == '1b1mu':
        df_temp = df_temp.loc[((df_temp['nTightMuons'] == 1) & (df_temp['NBJet'] == 0))]
    elif args.region == '2b1mu':
        df_temp = df_temp.loc[((df_temp['nTightMuons'] == 1) & (df_temp['NBJet'] == 0))]
    elif args.region == '2mu':
        df_temp = df_temp.loc[(df_temp['nTightMuons'] == 2)]
    elif args.region == '0b2mu':
        df_temp = df_temp.loc[((df_temp['nTightMuons'] == 2) & (df_temp['NBJet'] == 0))]
    elif args.region == '1b2mu':
        df_temp = df_temp.loc[((df_temp['nTightMuons'] == 2) & (df_temp['NBJet'] == 1))]
    elif args.region == '2b2mu':
        df_temp = df_temp.loc[((df_temp['nTightMuons'] == 2) & (df_temp['NBJet'] == 2))]
    else:
        print('Error: not implemented yet')
        exit()
    temp_efficiencies.append(float(df_temp['crosssec'].compute().sum())/nentries)

    # Double-b jets
    df_temp_temp = df_temp.loc[((df_temp['NDoubleBJet'] == 0) & (df_temp['NBJet'] > 2))]
    temp_efficiencies.append(float(df_temp_temp['crosssec'].compute().sum())/nentries)
    df_temp_temp = df_temp.loc[((df_temp['NDoubleBJet'] == 1) & (df_temp['NBJet'] > 1))]
    temp_efficiencies.append(float(df_temp_temp['crosssec'].compute().sum())/nentries)
    df_temp_temp = df_temp.loc[(df_temp['NDoubleBJet'] == 2)]
    temp_efficiencies.append(float(df_temp_temp['crosssec'].compute().sum())/nentries)

    theDataframe[thing] = temp_efficiencies

theDataframe.to_csv(os.path.join(directory, 'CutFlow.txt'), sep='\t', index=False)
theDataframe.to_latex(os.path.join(directory, 'CutFlow.tex'), index=False, bold_rows=True)
