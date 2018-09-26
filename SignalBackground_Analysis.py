#!/usr/bin/env python
import pandas as pd
from multiprocessing.pool import ThreadPool
import dask
import dask.dataframe as dd
from histbook import Hist, bin
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
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
parser.add_argument('-l', '--Lumi', type=float, default=35900., help='Luminosity in pb-1')
parser.add_argument('--HT_cut', type=float, default=None, help='Apply minimum HT cut')
parser.add_argument('--NMinusOne', action='store_true', help='Make n-1 plots, i.e. all cuts except those on the x-axis')
parser.add_argument('--region', default='All', help='Applies no cut, lepton veto or lepton requirement. Choose: All, Signal, 2b1mu, 0b1mu, 2mu, 0b2mu')
parser.add_argument('--DBT', type=float, default=None, help='Apply minimum DBT score cut (when no Data sources)')
parser.add_argument('--norm', action='store_true', help='Normalise each histogram')
parser.add_argument('--Higgs2bb', action='store_true', help='Require both Higgs-->bb in signal')
parser.add_argument('--XKCD', action='store_true', help='XKCD-like plots')
parser.add_argument('--stackBKG', action='store_true', help='Stack the BKG histos')
parser.add_argument('-x', '--NoX', action='store_true', help='This argument suppresses showing plots via X-forwarding')
parser.add_argument('-o', '--NoOutput', action='store_true', help='This argument suppresses the output of PDF plots')
parser.add_argument('--Threads', type=int, default=None, help='Optional: Set max number of cores for Dask to use')
parser.add_argument('-v', '--verbose', action='store_true', help='Increased verbosity level')
parser.add_argument('--style', default=None, help='Optional drawing style, e.g. \"ggplot\" in Matplotlib or \"dark\" in Seaborn')
parser.add_argument('--kdeplot', action='store_true', help='Use kdeplot in Seaborn instead of matplotlib histogram')
parser.add_argument('--kdeplot_fill', action='store_true', help='Same as --kdeplot but area under each line is filled')
args=parser.parse_args()

if args.Data and args.NMinusOne:
    print('Error: Cannot apply n-1 cuts with data since currently blinded!')
    exit()

if args.Threads:
    dask.set_options(pool=ThreadPool(args.Threads))

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

if args.XKCD:
    plt.xkcd()

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

def df_chop_chop(df=None, region='All', HT=None, DBT=None, isData=None):
    if region == 'Signal':
        df = df.loc[(df['nLooseMuons'] == 0)]
    elif region == '1mu':
        df = df.loc[((df['nTightMuons'] == 1) & (df['Muon_MHT_TransMass'] < 100.))]
    elif region == '1b1mu':
        df = df.loc[((df['NBJet'] == 1) & (df['nTightMuons'] == 1) & (df['Muon_MHT_TransMass'] < 100.))]
    elif region == '2b1mu':
        df = df.loc[((df['NBJet'] == 2) & (df['nTightMuons'] == 1) & (df['Muon_MHT_TransMass'] < 100.))]
    elif region == '0b1mu':
        df = df.loc[((df['NBJet'] == 0) & (df['nTightMuons'] == 1) & (df['Muon_MHT_TransMass'] < 100.))]
    elif region == '2mu':
        df = df.loc[((df['nTightMuons'] == 2) & (df['Muons_InvMass'] > 80.) & (df['Muons_InvMass'] < 100.))]
    elif region == '2b2mu':
        df = df.loc[((df['NBJet'] == 2) & (df['nTightMuons'] == 2) & (df['Muons_InvMass'] > 80.) & (df['Muons_InvMass'] < 100.))]
    elif region == '1b2mu':
        df = df.loc[((df['NBJet'] == 1) & (df['nTightMuons'] == 2) & (df['Muons_InvMass'] > 80.) & (df['Muons_InvMass'] < 100.))]
    elif region == '0b2mu':
        df = df.loc[((df['NBJet'] == 0) & (df['nTightMuons'] == 2) & (df['Muons_InvMass'] > 80.) & (df['Muons_InvMass'] < 100.))]
    if HT:
        df = df.loc[(df['HT'] > HT)]
    if DBT and not isData:
        df = df.loc[(df['MaxFatJetDoubleB_discrim'] > DBT)]
    return df;

def df_NMinusOne(df=None, var=None, region=None):
    d = {'MHT': 200.,
         'HT': 1500.,
         'NJet': 5.,
        }
    d_upper = {'MHT': 2000.,
               'HT': 6000.,
               'NJet': 20}
    if ((var not in ['NBJet', 'NDoubleBJet']) & (region == 'Signal')):
        df = df.loc[(((df['NDoubleBJet'] == 0) & (df['NBJet'] > 2)) | ((df['NDoubleBJet'] == 1) & (df['NBJet'] > 1)) | (df['NDoubleBJet'] > 1))]
    theVars = ['MHT', 'HT', 'NJet']
    for x in theVars:
        if x != var:
            df = df.loc[(df[x] > d[x])]
        elif x == var:
            df = df.loc[(df[x] < d_upper[x])]
    return df;

if args.Data:
    variables = ['HT', 'MHT', 'NJet', 'NBJet', 'nLooseMuons', 'nTightMuons', 'Muon_MHT_TransMass', 'Muons_InvMass', 'LeadSlimJet_Pt']
    types = {'MHT': np.float32,
             'HT': np.float32,
             'NJet': np.uint8,
             'NBJet': np.uint8,
             'nLooseMuons': np.uint8,
             'nTightMuons': np.uint8,
             'Muon_MHT_TransMass': np.float32,
             'Muons_InvMass': np.float32,
             'LeadSlimJet_Pt': np.float32,
             'crosssec': np.float32,
             'NoEntries': np.uint32,
            }
else:
    variables = ['HT', 'MHT', 'FatJetAngularSeparation', 'NJet', 'NFatJet', 'NBJet', 'NDoubleBJet', 'nHiggs2bb', 'MaxFatJetDoubleB_discrim', 'FatJet_MaxDoubleB_discrim_mass', 'nLooseMuons', 'nTightMuons', 'Muon_MHT_TransMass', 'Muons_InvMass', 'LeadSlimJet_Pt']
    types = {'MHT': np.float32,
             'HT': np.float32,
             'NJet': np.uint8,
             'NFatJet': np.uint8,
             'NBJet': np.uint8,
             'NDoubleBJet': np.uint8,
             'MaxFatJetDoubleB_discrim': np.float32,
             'FatJet_MaxDoubleB_discrim_mass': np.float32,
             'nHiggs2bb': np.uint8,
             'nLooseMuons': np.uint8,
             'nTightMuons': np.uint8,
             'Muon_MHT_TransMass': np.float32,
             'Muons_InvMass': np.float32,
             'LeadSlimJet_Pt': np.float32,
             'crosssec': np.float32,
             'NoEntries': np.float32,
            }

columns = variables
columns.append('crosssec')
columns.append('NoEntries')
columns.append('M_lsp')
columns.append('M_sq')

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
    df_sig = df_chop_chop(df=df_sig, region=args.region, HT=args.HT_cut, DBT=args.DBT, isData=args.Data)
    if args.Higgs2bb:
        df_sig = df_sig.loc[(df_sig['nHiggs2bb'] == 2)]
    #print('Signal df read, memory used: {0}'.format(mem_usage(df_sig)))

if args.MSSM:
    df_MSSM = dd.read_csv(args.MSSM, delimiter=r'\s+', usecols=columns, dtype=types)
    df_MSSM['weight'] = args.Lumi*df_MSSM['crosssec']/df_MSSM['NoEntries']
    df_MSSM = df_chop_chop(df=df_MSSM, region=args.region, HT=args.HT_cut, DBT=args.DBT, isData=args.Data)
    if args.verbose:
        print('MSSM:')
        print(df_MSSM)
    #print('MSSM df read, memory used: {0}'.format(mem_usage(df_MSSM)))

if args.QCD:
    df_QCD = dd.read_csv(args.QCD, delimiter=r'\s+', usecols=columns, dtype=types)
    df_QCD_entries = df_QCD[['NoEntries', 'crosssec']].drop_duplicates().compute()
    df_QCD['NoEntries'] = df_QCD_entries['NoEntries'].sum()
    df_QCD['weight'] = args.Lumi*df_QCD['crosssec']/df_QCD['NoEntries']
    df_QCD = df_chop_chop(df=df_QCD, region=args.region, HT=args.HT_cut, DBT=args.DBT, isData=args.Data)
    if args.verbose:
        print('QCD:')
        print(df_QCD)
    #print('QCD df read, memory used: {0}'.format(mem_usage(df_QCD)))

if args.TTJets:
    df_TTJets = dd.read_csv(args.TTJets, delimiter=r'\s+', usecols=columns, dtype=types)
    df_TTJets['weight'] = args.Lumi*df_TTJets['crosssec']/df_TTJets['NoEntries']
    df_TTJets = df_chop_chop(df=df_TTJets, region=args.region, HT=args.HT_cut, DBT=args.DBT, isData=args.Data)
    if args.verbose:
        print('TTJets:')
        print(df_TTJets)
    #print('TTJets df read, memory used: {0}'.format(mem_usage(df_TTJets)))

if args.WJets:
    df_WJets = dd.read_csv(args.WJets, delimiter=r'\s+', usecols=columns, dtype=types)
    df_WJets['weight'] = args.Lumi*df_WJets['crosssec']/df_WJets['NoEntries']
    df_WJets = df_chop_chop(df=df_WJets, region=args.region, HT=args.HT_cut, DBT=args.DBT, isData=args.Data)
    if args.verbose:
        print('WJets:')
        print(df_WJets)
    #print('WJets df read, memory used: {0}'.format(mem_usage(df_WJets)))

if args.ZJets:
    df_ZJets = dd.read_csv(args.ZJets, delimiter=r'\s+', usecols=columns, dtype=types)
    df_ZJets['weight'] = args.Lumi*df_ZJets['crosssec']/df_ZJets['NoEntries']
    df_ZJets = df_chop_chop(df=df_ZJets, region=args.region, HT=args.HT_cut, DBT=args.DBT, isData=args.Data)
    if args.verbose:
        print('ZJets:')
        print(df_ZJets)
    #print('ZJets df read, memory used: {0}'.format(mem_usage(df_ZJets)))

if args.DiBoson:
    df_DiBoson = dd.read_csv(args.DiBoson, delimiter=r'\s+', usecols=columns, dtype=types)
    df_DiBoson['weight'] = args.Lumi*df_DiBoson['crosssec']/df_DiBoson['NoEntries']
    df_DiBoson = df_chop_chop(df=df_DiBoson, region=args.region, HT=args.HT_cut, DBT=args.DBT, isData=args.Data)
    if args.verbose:
        print('DiBoson:')
        print(df_DiBoson)
    #print('DiBoson df read, memory used: {0}'.format(mem_usage(df_DiBoson)))

if args.SingleTop:
    df_SingleTop = dd.read_csv(args.SingleTop, delimiter=r'\s+', usecols=columns, dtype=types)
    df_SingleTop['weight'] = args.Lumi*df_SingleTop['crosssec']/df_SingleTop['NoEntries']
    df_SingleTop = df_chop_chop(df=df_SingleTop, region=args.region, HT=args.HT_cut, DBT=args.DBT, isData=args.Data)
    if args.verbose:
        print('SingleTop:')
        print(df_SingleTop)
    #print('SingleTop df read, memory used: {0}'.format(mem_usage(df_SingleTop)))

if args.TTW:
    df_TTW = dd.read_csv(args.TTW, delimiter=r'\s+', usecols=columns, dtype=types)
    df_TTW['weight'] = args.Lumi*df_TTW['crosssec']/df_TTW['NoEntries']
    df_TTW = df_chop_chop(df=df_TTW, region=args.region, HT=args.HT_cut, DBT=args.DBT, isData=args.Data)
    if args.verbose:
        print('TTW:')
        print(df_TTW)
    #print('TTW df read, memory used: {0}'.format(mem_usage(df_TTW)))

if args.TTZ:
    df_TTZ = dd.read_csv(args.TTZ, delimiter=r'\s+', usecols=columns, dtype=types)
    df_TTZ['weight'] = args.Lumi*df_TTZ['crosssec']/df_TTZ['NoEntries']
    df_TTZ = df_chop_chop(df=df_TTZ, region=args.region, HT=args.HT_cut, DBT=args.DBT, isData=args.Data)
    if args.verbose:
        print('TTZ:')
        print(df_TTZ)
    #print('TTZ df read, memory used: {0}'.format(mem_usage(df_TTZ)))

if args.Data:
    df_Data = dd.read_csv(args.Data, delimiter=r'\s+', usecols=columns, dtype=types)
    df_Data = df_chop_chop(df=df_Data, region=args.region, HT=args.HT_cut, DBT=args.DBT, isData=True)
    if args.verbose:
        print('Data:')
        print(df_Data)
    #print('Data df read, memory used: {0}'.format(mem_usage(df_Data)))

#Make the output directories
directory = 'Signal_vs_Background_Analysis'
if args.NMinusOne:
    directory = directory + '_NMinusOne'
if args.region:
    directory = directory + '_{0}Region'.format(args.region)
if args.HT_cut:
    directory = directory + '_HT{0}'.format(args.HT_cut)
if args.DBT and not args.Data:
    directory = directory + '_DBT{0}'.format(args.DBT)
if args.norm:
    directory = directory + '_normalised'
if args.Higgs2bb:
    directory = directory + '_Higgs2bb'
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
    f = open(os.path.join(temp_dir, 'command.txt'), 'w')
    f.write(commandString)
    f.close()


dict = {'MHT': {'bin': bin('MHT', 50, 0., 1000.), 'title': 'Missing $H_{T}$ [GeV/$c$]'},
        'HT': {'bin': bin('HT', 60, 0., 6000.), 'title': 'Total $H_{T}$ [GeV/$c$]'},
        'FatJetAngularSeparation': {'bin': bin('FatJetAngularSeparation', 50, 0., 5.), 'title': 'AK8 Jets $\Delta R$'},
        'NJet': {'bin': bin('NJet', 20, 0, 20), 'title': 'Number of Jets'},
        'NFatJet': {'bin': bin('NFatJet', 8, 0, 8), 'title': 'Number of AK8 FatJets'},
        'NBJet': {'bin': bin('NBJet', 14, 0, 14), 'title': 'Number of $b$-tagged Jets'},
        'NDoubleBJet': {'bin': bin('NDoubleBJet', 3, 0, 3), 'title': 'Number of double-$b$-tagged AK8 Jets'},
        'MaxFatJetDoubleB_discrim': {'bin': bin('MaxFatJetDoubleB_discrim', 50, -1., 1,), 'title': 'AK8 Fat Jet Double-$b$-tag score'},
        'FatJet_MaxDoubleB_discrim_mass': {'bin': bin('FatJet_MaxDoubleB_discrim_mass', 50, 0., 500.), 'title': 'AK8 SoftDrop Mass [GeV/$c^{2}$]'},
        'nLooseMuons': {'bin': bin('nLooseMuons', 6, 0, 6), 'title': 'Number of Loose ID/isolation Muons'},
        'nTightMuons': {'bin': bin('nTightMuons', 6, 0, 6), 'title': 'Number of Tight ID/isolation Muons'},
        'Muon_MHT_TransMass': {'bin': bin('Muon_MHT_TransMass', 50, 0., 400.), 'title': 'Muon-Missing $H_{T}$ Transverse Mass [GeV/$c^{2}$]'},
        'Muons_InvMass': {'bin': bin('Muons_InvMass', 50, 0., 400.), 'title': "Di-Muon Invariant Mass [GeV/$c^{2}$]"},
        'LeadSlimJet_Pt': {'bin': bin('LeadSlimJet_Pt', 50, 0., 1000.), 'title': "Lead Jet $p_{T}$ [GeV/$c$]"},
        }

linewidth = 2.

for var in variables:
    if var in ['nHiggs2bb', 'NoEntries', 'M_lsp', 'M_sq']:
        continue
    plt.figure()
    temp_i = 0
    if args.signal:
        for index, row in df_sig_masses.iterrows():
            temp_i += 5
            label='$M_{\mathrm{Squark}}$ = ' + str(row["M_sq"]) + ', $M_{\mathrm{LSP}}$ = ' + str(row["M_lsp"])
            print(label, var)
            df_temp = df_sig.loc[(df_sig['M_sq'] == row['M_sq']) & (df_sig['M_lsp'] == row['M_lsp'])]
            if args.NMinusOne:
                df_temp = df_NMinusOne(df_temp, var, args.region)
            if df_temp[var].compute().shape[0] > 0:
                h = Hist(dict[var]['bin'], weight='weight')
                h.fill(df_temp)
                df = h.pandas(normalized=args.norm).reset_index()[1:-1]
                df[var] = df[var].apply(lambda x: x.left)
                plt.hist(df[var], bins=df[var], weights=df['count()'], normed=args.norm, label=label, log=True, histtype="step", linewidth=linewidth, zorder=35-temp_i)

    if args.MSSM:
        label='MSSM-like: $M_{\mathrm{Squark}}$ = ' + str(df_MSSM["M_sq"][0]) + ', $M_{\mathrm{LSP}}$ = ' + str(df_MSSM["M_lsp"][0])
        print(label, var)
        h = Hist(dict[var]['bin'], weight='weight')
        if args.NMinusOne:
            df_temp = df_NMinusOne(df_MSSM, var, args.region)
        if df_temp[var].compute().shape[0] > 0:
            h.fill(df_temp)
            df = h.pandas(normalized=args.norm).reset_index()[1:-1]
            df[var] = df[var].apply(lambda x: x.left)
            plt.hist(df[var], bins=df[var], weights=df['count()'], label=label, normed=args.norm, log=True, histtype="step", linewidth=linewidth, zorder=10)

    theBkgs = []
    bkgLabels = []
    bkgWeights = []
    if args.QCD:
        bkgLabels.append('QCD background')
        print('qcd', var)
        h = Hist(dict[var]['bin'], weight='weight')
        if args.NMinusOne:
       	    df_temp = df_NMinusOne(df_QCD, var, args.region)
        if df_temp[var].compute().shape[0] > 0:
            h.fill(df_temp)
            df = h.pandas().reset_index()[1:-1]
            df[var] = df[var].apply(lambda x: x.left)
            theBkgs.append(df[var])
            bkgWeights.append(df['count()'])
    if args.TTJets:
        bkgLabels.append('$t \overline{t}$ + $jets$ background')
        print('tt', var)
        h = Hist(dict[var]['bin'], weight='weight')
        if args.NMinusOne:
       	    df_temp = df_NMinusOne(df_TTJets, var, args.region)
        if df_temp[var].compute().shape[0] > 0:
            h.fill(df_temp)
            df = h.pandas().reset_index()[1:-1]
            df[var] = df[var].apply(lambda x: x.left)
            theBkgs.append(df[var])
            bkgWeights.append(df['count()'])
    if args.WJets:
        bkgLabels.append('$W$ + $jets$ background')
        print('w', var)
        h = Hist(dict[var]['bin'], weight='weight')
        if args.NMinusOne:
       	    df_temp = df_NMinusOne(df_WJets, var, args.region)
        if df_temp[var].compute().shape[0] > 0:
            h.fill(df_temp)
            df = h.pandas().reset_index()[1:-1]
            df[var] = df[var].apply(lambda x: x.left)
            theBkgs.append(df[var])
            bkgWeights.append(df['count()'])
    if args.ZJets:
        bkgLabels.append('$Z$ + $jets$ background')
        print('z', var)
        h = Hist(dict[var]['bin'], weight='weight')
        if args.NMinusOne:
       	    df_temp = df_NMinusOne(df_ZJets, var, args.region)
        if df_temp[var].compute().shape[0] > 0:
            h.fill(df_temp)
            df = h.pandas().reset_index()[1:-1]
            df[var] = df[var].apply(lambda x: x.left)
            theBkgs.append(df[var])
            bkgWeights.append(df['count()'])
    if args.DiBoson:
        bkgLabels.append('Di-Boson background')
        print('boson', var)
        h = Hist(dict[var]['bin'], weight='weight')
        if args.NMinusOne:
       	    df_temp = df_NMinusOne(df_DiBoson, var, args.region)
        if df_temp[var].compute().shape[0] > 0:
            h.fill(df_temp)
            df = h.pandas().reset_index()[1:-1]
            df[var] = df[var].apply(lambda x: x.left)
            theBkgs.append(df[var])
            bkgWeights.append(df['count()'])
    if args.SingleTop:
        bkgLabels.append('$t$ + $jets$ background')
        print('st', var)
        h = Hist(dict[var]['bin'], weight='weight')
        if args.NMinusOne:
       	    df_temp = df_NMinusOne(df_SingleTop, var, args.region)
        if df_temp[var].compute().shape[0] > 0:
            h.fill(df_temp)
            df = h.pandas().reset_index()[1:-1]
            df[var] = df[var].apply(lambda x: x.left)
            theBkgs.append(df[var])
            bkgWeights.append(df['count()'])
    if args.TTW:
        bkgLabels.append('$t\overline{t}W$ + $jets$ background')
        print('ttw', var)
        h = Hist(dict[var]['bin'], weight='weight')
        if args.NMinusOne:
       	    df_temp = df_NMinusOne(df_TTW, var, args.region)
        if df_temp[var].compute().shape[0] > 0:
            h.fill(df_temp)
            df = h.pandas().reset_index()[1:-1]
            df[var] = df[var].apply(lambda x: x.left)
            theBkgs.append(df[var])
            bkgWeights.append(df['count()'])
    if args.TTZ:
        bkgLabels.append('$t\overline{t}Z$ + $jets$ background')
        print('ttz', var)
        h = Hist(dict[var]['bin'], weight='weight')
        if args.NMinusOne:
       	    df_temp = df_NMinusOne(df_TTZ, var, args.region)
        if df_temp[var].compute().shape[0] > 0:
            h.fill(df_temp)
            df = h.pandas().reset_index()[1:-1]
            df[var] = df[var].apply(lambda x: x.left)
            theBkgs.append(df[var])
            bkgWeights.append(df['count()'])

    if ((args.QCD) or (args.TTJets) or (args.WJets) or (args.ZJets) or (args.DiBoson) or (args.SingleTop) or (args.TTW) or (args.TTZ)):
        plt.hist(theBkgs, bins=df[var], weights=bkgWeights, label=bkgLabels, log=True, normed=args.norm, stacked=True, histtype="stepfilled", linewidth=0., zorder=5)

    if args.Data:
        label='Data'
        h = Hist(dict[var]['bin'])
        h.fill(df_Data)
        df = h.pandas(normalized=args.norm).reset_index()[1:-1]
        df[var] = df[var].apply(lambda x: x.mid)
        plt.errorbar(df[var], df['count()'], yerr=df['err(count())'], fmt='o', markersize=4, label=label, zorder=35)

    plt.xlabel(dict[var]['title'], size=14)
    plt.yscale('log')
    leg = plt.legend(loc='upper right', fontsize='xx-small')
    leg.set_zorder(100)
#    if not args.norm:
#        if var not in ['NJet', 'NBJet']:
#            plt.ylim(0.01, None)
#            plt.xlim(0., None)
#        else:
#            plt.ylim(0.1, None)
    if not args.NoOutput:
        plt.savefig(os.path.join(temp_dir, var + '.pdf'))
        print('Saved ' + var + '.pdf output file')
    if not args.NoX:
        plt.show()
