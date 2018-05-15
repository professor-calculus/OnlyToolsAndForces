#!/usr/bin/env python
from scipy import interpolate
import numpy as np
import matplotlib
from matplotlib import rc
matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
matplotlib.rcParams['text.latex.preamble'].append(r'\usepackage{amsmath}')
#matplotlib.rcParams['text.usetex'] = True
#matplotlib.rcParams['text.latex.unicode'] = True
#matplotlib.rcParams['text.dvipnghack'] = True
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib.colors as mcol

import sys
import argparse as a
import pandas as pd
from snakebite.client import AutoConfigClient
from pandas.compat import StringIO

#Get options
parser=a.ArgumentParser(description='Pheno Limit Plot')
parser.add_argument('-f','--file',required=True)
parser.add_argument('-t','--title',default='Cutflow Efficiency')
parser.add_argument('-x', '--NoX', action='store_true', help='This argument suppresses showing plots via X-forwarding')
parser.add_argument('-o', '--NoOutput', action='store_true', help='This argument suppresses the output of PDF plots')
args=parser.parse_args()

df_list = []
file_list = []

fs = AutoConfigClient()

HT_eff_tot = []
MHT_eff_tot = []
BDP_eff_tot = []
NBJet_eff_tot = []
NJet_eff_tot = []
NVeto_eff_tot = []
M_sq = []
M_lsp = []

for f in fs.text([args.file+'/*/ROOTCuts_output/ROOTCuts.txt']):
    df = pd.read_csv(StringIO(f), delimiter=r'\s+')
    df_HT = df.loc[(df['HT'] > 1200.)]
    df_MHT = df.loc[(df['MHT'] > 200.)]
    df_NBJet = df.loc[(df['NBJet'] > 1)]
    df_BDP = df.loc[(df['bDPhi'] > 0.5)]
    df_NJet = df.loc[(df['NJet'] > 5)]
    df_NVeto = df.loc[(df['NVeto'] == 0)]

    HT_eff = float(len(df_HT['HT']))/float(len(df['HT']))
    MHT_eff = float(len(df_MHT['HT']))/float(len(df['HT']))
    NBJet_eff = float(len(df_NBJet['HT']))/float(len(df['HT']))
    BDP_eff = float(len(df_BDP['HT']))/float(len(df['HT']))
    NJet_eff = float(len(df_NJet['HT']))/float(len(df['HT']))
    NVeto_eff = float(len(df_NVeto['HT']))/float(len(df['HT']))

    M_sq.append(df['M_sq'][0])
    M_lsp.append(df['M_lsp'][0])

    HT_eff_tot.append(HT_eff)
    MHT_eff_tot.append(MHT_eff)
    BDP_eff_tot.append(BDP_eff)
    NBJet_eff_tot.append(NBJet_eff)
    NJet_eff_tot.append(NJet_eff)
    NVeto_eff_tot.append(NVeto_eff)

#    df_list.append(old_df_temp)

#old_df = pd.concat(df_list)

#print(old_df)

#df_sig_masses = old_df[['M_sq', 'M_lsp']].drop_duplicates()
#df_sig_masses = df_sig_masses.sort_values(by=['M_sq', 'M_lsp'])
npoints = 21

#print(df['Efficiencies_MHTeff'])

'''
for index, row in df_sig_masses.iterrows():

    df = old_df.loc[(old_df['M_sq'] == row['M_sq']) & (old_df['M_lsp'] == row['M_lsp'])]

    df_HT = df.loc[(df['HT'] > 1200.)]
    df_MHT = df.loc[(df['MHT'] > 200.)]
    df_NBJet = df.loc[(df['NBJet'] > 1)]
    df_BDP = df.loc[(df['bDPhi'] > 0.5)]
    df_NJet = df.loc[(df['NJet'] > 5)]
    df_NVeto = df.loc[(df['NVeto'] == 0)]

    HT_eff = float(len(df_HT['HT']))/float(len(df['HT']))
    MHT_eff = float(len(df_MHT['HT']))/float(len(df['HT']))
    NBJet_eff = float(len(df_NBJet['HT']))/float(len(df['HT']))
    BDP_eff = float(len(df_BDP['HT']))/float(len(df['HT']))
    NJet_eff = float(len(df_NJet['HT']))/float(len(df['HT']))
    NVeto_eff = float(len(df_NVeto['HT']))/float(len(df['HT']))

    M_sq.append(row['M_sq'])
    M_lsp.append(row['M_lsp'])
    
    HT_eff_tot.append(HT_eff)
    MHT_eff_tot.append(MHT_eff)
    BDP_eff_tot.append(BDP_eff)
    NBJet_eff_tot.append(NBJet_eff)
    NJet_eff_tot.append(NJet_eff)
    NVeto_eff_tot.append(NVeto_eff)
'''

def interp2(m1, m2, eff, vmin, method='linear', n_p=10):

    x = np.asarray(m1)
    y = np.asarray(m2)
    z = np.asarray(eff)

    nx = np.unique(x)
    ny = np.unique(y)

    minn = min(len(nx), len(ny))
    
    xi = np.linspace(x.min(), x.max(), n_p)
    yi = np.linspace(y.min(), y.max(), n_p)
    xi, yi = np.meshgrid(xi,yi)
    zi = interpolate.griddata((x, y), z, (xi, yi), method=method)

    zi = np.clip(zi, vmin, 1.)
    
    return xi, yi, zi


stops = [0.0000, 0.1250, 0.2500, 0.3750, 0.5000, 0.6250, 0.7500, 0.8750, 1.0000]
red   = [0.2082, 0.0592, 0.0780, 0.0232, 0.1802, 0.5301, 0.8186, 0.9956, 0.9764]
green = [0.1664, 0.3599, 0.5041, 0.6419, 0.7178, 0.7492, 0.7328, 0.7862, 0.9832]
blue  = [0.5293, 0.8684, 0.8385, 0.7914, 0.6425, 0.4662, 0.3499, 0.1968, 0.0539]


ered = []
egreen = []
eblue = []
for i, stop in enumerate(stops):
    if i is 0:
        ered.append( (stop, 0., red[i]) )
        egreen.append( (stop, 0., green[i]) )
        eblue.append( (stop, 0., blue[i]) )
    elif i is len(stops)-1:
        ered.append( (stop, red[i], 1.) )
        egreen.append( (stop, green[i], 1.) )
        eblue.append( (stop, blue[i], 1.) )
    else:
        ered.append( (stop, red[i], red[i]) )
        egreen.append( (stop, green[i], green[i]) )
        eblue.append( (stop, blue[i], blue[i]) )
cdict = {'red': ered, 'green': egreen, 'blue': eblue}

bird = mcol.LinearSegmentedColormap('bird', cdict)
bird.set_over("yellow")

vlog = np.logspace(-2, 0, 200)
vlin = np.linspace(0., 1., 200)
vred = np.linspace(0.5, 1., 100)
vsmall = np.linspace(0., 0.5, 100)
vsmallish = np.linspace(0., 0.7, 150)

vars = ['MHT_eff','HT_eff', 'NJet_eff', 'NBJet_eff', 'BDP_eff', 'NVeto_eff']

dict = {'MHT_eff': {'column': MHT_eff_tot, 'title': 'Fraction of Events with Missing-$H_{T} > 200$GeV', 'scale': vred, 'vmin': 0.5},
        'HT_eff': {'column': HT_eff_tot, 'title': 'Fraction of Events with Total $H_{T} > 1200$GeV', 'scale': vlin, 'vmin': 0.},
        'NJet_eff': {'column': NJet_eff_tot, 'title': 'Fraction of Events with # of jets > 5', 'scale': vlin, 'vmin': 0.},
        'NBJet_eff': {'column': NBJet_eff_tot, 'title': 'Fraction of Events with # of $b$-jets $\\geq 2$', 'scale': vlin, 'vmin': 0.},
        'BDP_eff': {'column': BDP_eff_tot, 'title': 'Fraction of Events with $\\Delta\\Phi^{*}$ > 0.5', 'scale': vsmall, 'vmin': 0.},
        'NVeto_eff': {'column': NVeto_eff_tot, 'title': 'Fraction of Events with no veto leptons/photons', 'scale': vlin, 'vmin': 0.},
        }

for var in vars:
    plt.figure()
    xi, yi, zi = interp2(M_sq, M_lsp, dict[var]['column'], dict[var]['vmin'], 'linear', 12*npoints)
    #plt.contourf(xi, yi, zi, levels=v, cmap='PuBu_r')
    plt.contourf(xi, yi, zi, levels=dict[var]['scale'], cmap=bird)
    plt.xlabel('$M_{Squark}$ [GeV]', size=12)
    plt.ylabel('$M_{LSP}$ [GeV]', size=12)
    cbar = plt.colorbar()
    cbar.set_ticks(np.around(0.1*(np.arange(10*dict[var]['vmin'],11,1)),1))
    cbar.set_ticklabels(np.around(0.1*(np.arange(10*dict[var]['vmin'],11,1)),1))
    #cbar.set_ticks([0.01,0.1,1,10,100,1000])
    #cbar.set_ticklabels([0.01,0.1,1])
    cbar.set_label(dict[var]['title'], rotation=90, fontsize=14, labelpad=5)
    if not args.NoX:
        plt.show()
    if not args.NoOutput:
        plt.savefig(var + '.pdf')
        print('Saved ' + var + '.pdf output file')
