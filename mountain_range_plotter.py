#!/usr/bin/env python
import pandas as pd
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
import seaborn as sns

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
parser.add_argument('--OutDir', default=None, help='Output folder')
parser.add_argument('--region', default=None, help='Which region, Signal, 1mu, 0b2mu etc')
parser.add_argument('-x', '--NoX', action='store_true', help='This argument suppresses showing plots via X-forwarding')
parser.add_argument('-o', '--NoOutput', action='store_true', help='This argument suppresses the output of PDF plots')
parser.add_argument('-v', '--verbose', action='store_true', help='Increased verbosity level')
parser.add_argument('--style', default='seaborn-colorblind', help='Optional drawing style, e.g. \"ggplot\" in Matplotlib or \"dark\" in Seaborn')
args=parser.parse_args()

sns.set_palette(sns.color_palette("Paired", 20))

def df_chop_chop(df=None, region='All'):
    msq = df['M_sq'][0]
    mlsp = df['M_lsp'][0]
    theType = df['Type'][0]

    binned_msq = []
    binned_mlsp = []
    binned_type = []
    binned_HT_bin = []
    binned_MHT_bin = []
    binned_N_jet_bin = []
    binned_N_bJet_bin = []
    binned_N_bJet_actual = []
    binned_N_doublebjet_bin = []
    binned_N_muons = []
    binned_yield = []

    for mhtBin in [200, 400, 600]:
        for htBin in [1500, 2500, 3500]:
            for nJetBin in [6]:
                for nMuons in [-1, 0, 1, 2]:
                    for nDoubleBJetBin in [0, 1, 2]:
                        if nDoubleBJetBin == 0:
                            for i in [3,4,5]:
                                binned_msq.append(msq)
                                binned_mlsp.append(mlsp)
                                binned_type.append(theType)
                                binned_HT_bin.append(htBin)
                                binned_MHT_bin.append(mhtBin)
                                binned_N_jet_bin.append(nJetBin)
                                binned_N_bJet_bin.append(3)
                                binned_N_bJet_actual.append(i)
                                binned_N_doublebjet_bin.append(nDoubleBJetBin)
                                binned_N_muons.append(nMuons)
                                binned_yield.append(0.)
                        elif nDoubleBJetBin == 1:
                            for nBJetBin in [2]:
                                binned_msq.append(msq)
                                binned_mlsp.append(mlsp)
                                binned_type.append(theType)
                                binned_HT_bin.append(htBin)
                                binned_MHT_bin.append(mhtBin)
                                binned_N_jet_bin.append(nJetBin)
                                binned_N_bJet_bin.append(nBJetBin)
                                binned_N_bJet_actual.append(nBJetBin)
                                binned_N_doublebjet_bin.append(nDoubleBJetBin)
                                binned_N_muons.append(nMuons)
                                binned_yield.append(0.)
                        elif nDoubleBJetBin == 2:
                            for i in [0,1,2,3]:
                                binned_msq.append(msq)
                                binned_mlsp.append(mlsp)
                                binned_type.append(theType)
                                binned_HT_bin.append(htBin)
                                binned_MHT_bin.append(mhtBin)
                                binned_N_jet_bin.append(nJetBin)
                                binned_N_bJet_bin.append(0)
                                binned_N_bJet_actual.append(i)
                                binned_N_doublebjet_bin.append(nDoubleBJetBin)
                                binned_N_muons.append(nMuons)
                                binned_yield.append(0.)

    df_tmp = pd.DataFrame({
        'Type': binned_type,
        'M_sq': binned_msq,
        'M_lsp': binned_mlsp,
        'HT_bin': binned_HT_bin,
        'MHT_bin': binned_MHT_bin,
        'n_Jet_bin': binned_N_jet_bin,
        'n_bJet_actual': binned_N_bJet_actual,
        'n_bJet_bin': binned_N_bJet_bin,
        'n_DoubleBJet_bin': binned_N_doublebjet_bin,
        'n_Muons_bin': binned_N_muons,
        'Yield': binned_yield,
        })

    df = pd.concat([df, df_tmp])

    if region == 'Signal':
        df = df.loc[(df['n_Muons_bin'] == 0)]
    elif region == 'SingleMuon_Control':
        df = df.loc[((df['n_Muons_bin'] == 1))]
    elif region == 'SingleMuon_1b_Control':
        df = df.loc[((df['n_bJet_actual'] == 1) & (df['n_Muons_bin'] == 1))]
    elif region == 'SingleMuon_2b_Control':
        df = df.loc[((df['n_bJet_actual'] == 2) & (df['n_Muons_bin'] == 1))]
    elif region == 'SingleMuon_0b_Control':
        df = df.loc[((df['n_bJet_actual'] == 0) & (df['n_Muons_bin'] == 1))]
    elif region == 'DoubleMuon_Control':
        df = df.loc[((df['n_Muons_bin'] == 2))]
    elif region == 'DoubleMuon_2b_Control':
        df = df.loc[((df['n_bJet_actual'] == 2) & (df['n_Muons_bin'] == 2))]
    elif region == 'DoubleMuon_1b_Control':
        df = df.loc[((df['n_bJet_actual'] == 1) & (df['n_Muons_bin'] == 2))]
    elif region == 'DoubleMuon_0b_Control':
        df = df.loc[((df['n_bJet_actual'] == 0) & (df['n_Muons_bin'] == 2))]
    
    return df;

if args.verbose:
    parser.print_help()
else:
    warnings.filterwarnings("ignore")

if args.style:
    plt.style.use(args.style)

print '\nPython Mountain Range Plotter\n'
print(args.signal)

df_sig_list = []
for file in args.signal:
    df = pd.read_csv(file, delimiter=r'\s+', header='infer')
    df = df_chop_chop(df, args.region)
    df_sig_list.append(df)
df_sig = pd.concat(df_sig_list)
df_sig = df_sig.groupby(by=['Type', 'M_sq', 'M_lsp', 'HT_bin', 'MHT_bin', 'n_Jet_bin', 'n_bJet_bin', 'n_DoubleBJet_bin', 'n_Muons_bin']).sum()
df_sig.reset_index(inplace=True)

print(df_sig)

# Number of bins as read from signal sample, assume bkg is the same else it's all nonsense anyway!
df_bins = df_sig.groupby(by=['HT_bin', 'MHT_bin', 'n_Jet_bin', 'n_bJet_bin', 'n_DoubleBJet_bin']).sum()
print('{0} bins considered'.format(df_bins.shape[0]))
x = np.arange(df_bins.shape[0])

if args.verbose:
    print('Signal:')
    print(df_sig)

print '\nSuccessfully read dataframe\n'

#Make the output directories
directory = 'Mountain_Range'
if args.region:
    directory = directory + '_{0}Region'.format(args.region)
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

df_sig_masses = df_sig[['M_sq', 'M_lsp']].drop_duplicates()
df_sig_masses = df_sig_masses.sort_values(by=['M_sq', 'M_lsp'])
print(df_sig_masses.head())

theBkgs = []
bkgLabels = []
bkgWeights = []
if args.QCD:
    df_list = []
    for file in args.QCD:
        df = pd.read_csv(file, delimiter=r'\s+', header='infer')
        df = df_chop_chop(df, args.region)
        df_list.append(df)
    df = pd.concat(df_list)
    df = df.drop('Type', axis=1)
    df = df.groupby(by=['M_sq', 'M_lsp', 'HT_bin', 'MHT_bin', 'n_Jet_bin', 'n_bJet_bin', 'n_DoubleBJet_bin', 'n_Muons_bin']).sum()
    df.reset_index(inplace=True)
    theBkgs.append(x)
    bkgWeights.append(df['Yield'])
    bkgLabels.append('QCD Multijet background')
if args.TTJets:
    df_list = []
    for file in args.TTJets:
        df = pd.read_csv(file, delimiter=r'\s+', header='infer')
        df = df_chop_chop(df, args.region)
        df_list.append(df)
    df = pd.concat(df_list)
    df = df.drop('Type', axis=1)
    df = df.groupby(by=['M_sq', 'M_lsp', 'HT_bin', 'MHT_bin', 'n_Jet_bin', 'n_bJet_bin', 'n_DoubleBJet_bin', 'n_Muons_bin']).sum()
    df.reset_index(inplace=True)
    theBkgs.append(x)
    bkgWeights.append(df['Yield'])
    bkgLabels.append('$t\overline{t}$ + jets background')
if args.WJets:
    df_list = []
    for file in args.WJets:
        df = pd.read_csv(file, delimiter=r'\s+', header='infer')
        df = df_chop_chop(df, args.region)
        df_list.append(df)
    df = pd.concat(df_list)
    df = df.drop('Type', axis=1)
    df = df.groupby(by=['M_sq', 'M_lsp', 'HT_bin', 'MHT_bin', 'n_Jet_bin', 'n_bJet_bin', 'n_DoubleBJet_bin', 'n_Muons_bin']).sum()
    df.reset_index(inplace=True)
    theBkgs.append(x)
    bkgWeights.append(df['Yield'])
    bkgLabels.append('$W$ + jets background')
if args.ZJets:
    df_list = []
    for file in args.ZJets:
        df = pd.read_csv(file, delimiter=r'\s+', header='infer')
        df = df_chop_chop(df, args.region)
        df_list.append(df)
    df = pd.concat(df_list)
    df = df.drop('Type', axis=1)
    df = df.groupby(by=['M_sq', 'M_lsp', 'HT_bin', 'MHT_bin', 'n_Jet_bin', 'n_bJet_bin', 'n_DoubleBJet_bin', 'n_Muons_bin']).sum()
    df.reset_index(inplace=True)
    theBkgs.append(x)
    bkgWeights.append(df['Yield'])
    bkgLabels.append('$Z$ + jets background')
if args.DiBoson:
    df_list = []
    for file in args.DiBoson:
        df = pd.read_csv(file, delimiter=r'\s+', header='infer')
        df = df_chop_chop(df, args.region)
        df_list.append(df)
    df = pd.concat(df_list)
    df = df.drop('Type', axis=1)
    df = df.groupby(by=['M_sq', 'M_lsp', 'HT_bin', 'MHT_bin', 'n_Jet_bin', 'n_bJet_bin', 'n_DoubleBJet_bin', 'n_Muons_bin']).sum()
    df.reset_index(inplace=True)
    theBkgs.append(x)
    bkgWeights.append(df['Yield'])
    bkgLabels.append('Di-Boson background')
if args.SingleTop:
    df_list = []
    for file in args.SingleTop:
        df = pd.read_csv(file, delimiter=r'\s+', header='infer')
        df = df_chop_chop(df, args.region)
        df_list.append(df)
    df = pd.concat(df_list)
    df = df.drop('Type', axis=1)
    df = df.groupby(by=['M_sq', 'M_lsp', 'HT_bin', 'MHT_bin', 'n_Jet_bin', 'n_bJet_bin', 'n_DoubleBJet_bin', 'n_Muons_bin']).sum()
    df.reset_index(inplace=True)
    theBkgs.append(x)
    bkgWeights.append(df['Yield'])
    bkgLabels.append('Single-$t$ background')
if args.TTW:
    df_list = []
    for file in args.TTW:
        df = pd.read_csv(file, delimiter=r'\s+', header='infer')
        df = df_chop_chop(df, args.region)
        df_list.append(df)
    df = pd.concat(df_list)
    df = df.drop('Type', axis=1)
    df = df.groupby(by=['M_sq', 'M_lsp', 'HT_bin', 'MHT_bin', 'n_Jet_bin', 'n_bJet_bin', 'n_DoubleBJet_bin', 'n_Muons_bin']).sum()
    df.reset_index(inplace=True)
    theBkgs.append(x)
    bkgWeights.append(df['Yield'])
    bkgLabels.append('$t\overline{t}W$ background')
if args.TTZ:
    df_list = []
    for file in args.TTZ:
        df = pd.read_csv(file, delimiter=r'\s+', header='infer')
        df = df_chop_chop(df, args.region)
        df_list.append(df)
    df = pd.concat(df_list)
    df = df.drop('Type', axis=1)
    df = df.groupby(by=['M_sq', 'M_lsp', 'HT_bin', 'MHT_bin', 'n_Jet_bin', 'n_bJet_bin', 'n_DoubleBJet_bin', 'n_Muons_bin']).sum()
    df.reset_index(inplace=True)
    theBkgs.append(x)
    bkgWeights.append(df['Yield'])
    bkgLabels.append('$t\overline{t}Z$ background')

n_signal = len(args.signal)
linewidth = 2.

if args.verbose:
    print(theBkgs)
    print(bkgWeights)
    print(bkgLabels)

plt.figure()
temp_i = 0
for index, row in df_sig_masses.iterrows():
    temp_i += 5
    label='$M_{\mathrm{Squark}}$ = ' + str(row["M_sq"]) + ', $M_{\mathrm{LSP}}$ = ' + str(row["M_lsp"])
    df_temp = df_sig.loc[(df_sig['M_sq'] == row['M_sq']) & (df_sig['M_lsp'] == row['M_lsp'])]
    df_temp = df_temp.replace(0., 1e-5) 
    df_temp['Bin'] = x
    plt.hist(df_temp['Bin'], bins=x, label=label, weights=df_temp['Yield'], log=True, histtype="step", linewidth=linewidth, zorder=35-temp_i)

if (args.QCD) or (args.TTJets) or (args.WJets) or (args.ZJets) or (args.DiBoson) or (args.SingleTop) or (args.TTW) or (args.TTZ):
    plt.hist(theBkgs, bins=x, weights=bkgWeights, label=bkgLabels, stacked=True, log=True, histtype="stepfilled", linewidth=0., zorder=0)

df = df.drop(['M_sq', 'M_lsp', 'n_Jet_bin', 'n_Muons_bin'], axis=1)
df = df.groupby(by=['HT_bin', 'MHT_bin', 'n_bJet_bin', 'n_DoubleBJet_bin']).sum()
df = df.astype('int32')
#print(df.index)

plt.xticks(x+0.5, [''.join(str(t)) for t in df.index], rotation=90)
plt.xlabel("HT, MHT, N_b, N_double-b bin", labelpad=20)
plt.tight_layout()
plt.ylim(0.05, None)
leg = plt.legend(loc='upper right', fontsize='xx-small')
leg.set_zorder(100)
if not args.NoOutput:
    plt.savefig(os.path.join(temp_dir, 'MountainRange.pdf'))
    print('Saved {0}/MountainRange.pdf output file'.format(temp_dir))
if not args.NoX:
    plt.show()

