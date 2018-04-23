#!/usr/bin/env python
import pandas as pd
import numpy as np
import matplotlib
from matplotlib import rc
matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
matplotlib.rcParams['text.latex.preamble'].append(r'\usepackage{amsmath}')
matplotlib.rcParams['xtick.top'] = True
matplotlib.rcParams['ytick.right'] = True
import matplotlib.pyplot as plt
import seaborn as sns
import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True
import sys
import argparse as a
import warnings

#Get Options

parser = a.ArgumentParser(description='Signal vs Background plot')
parser.add_argument('-d', '--data', default=None, help='Path to data txt file')
parser.add_argument('-m', '--mc', default=None, help='Path to total SM mc txt file')
parser.add_argument('-x', '--NoX', action='store_true', help='This argument suppresses the X11 display')
parser.add_argument('-o', '--NoOutput', action='store_true', help='This argument suppresses the output of PDF plots')
parser.add_argument('-v', '--verbose', action='store_true', help='Increased verbosity level')
parser.add_argument('--style', default=None, help='Optional drawing style, e.g. \"ggplot\" in Matplotlib')
args=parser.parse_args()

if args.verbose:
    parser.print_help()
else:
    warnings.filterwarnings("ignore")

if args.style:
    plt.style.use(args.style)

print '\nFAST Dataframe Data vs MC Plot-O-Matic\n'

if args.data:
    df_data = pd.read_csv(args.data, delimiter=r'\s+', comment='#')
if args.mc:
    df_mc = pd.read_csv(args.mc, delimiter=r'\s+', comment='#')

bins_alphat = np.array([0.,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.8,1.,1.2,1.4,1.6,1.8,2.,2.5])
bins_BDP = np.array([0.,0.1,0.2,0.3,0.4,0.5,0.6,0.8,1.,1.2,1.4,1.6,1.8,3.])
width_alphat = bins_alphat[1:] - bins_alphat[:-1]
width_alphat = np.append(width_alphat, [1.])
width_BDP = bins_BDP[1:] - bins_BDP[:-1]
width_BDP = np.append(width_BDP, [1.])


dict = {'alphaT': {'column': 'alphaT', 'bins': bins_alphat, 'width': width_alphat, 'title': '$\\alpha_{T}$'},
        'BDP': {'column': 'biasedDPhi', 'bins': bins_BDP, 'width': width_BDP, 'title': '$\\Delta\Phi^{*}$'},
        }

variables = ['alphaT', 'BDP']

for var in variables:
    plt.figure()

    if args.data:
        df_data_temp = df_data.groupby(dict[var]['column']).sum().reset_index()
        df_data_temp['nvar'] = df_data_temp['nvar'].apply(np.sqrt)
        data_widths = dict[var]['width'][np.digitize(df_data_temp[dict[var]['column']], dict[var]['bins']) - 1]

        data_hist, data_bins, crap = plt.hist(df_data_temp[dict[var]['column']], bins=dict[var]['bins'], weights=df_data_temp['n']/data_widths)
        err_hist, err_bins, crap2 = plt.hist(df_data_temp[dict[var]['column']], bins=dict[var]['bins'], weights=df_data_temp['nvar']/data_widths)
        mid = 0.5*(data_bins[1:] + data_bins[:-1])
        plt.clf()
        plt.errorbar(mid, data_hist, yerr=err_hist, label='Data', ls='none', fmt='.', color='black')

    if args.mc:
        df_mc_temp = df_mc.groupby(dict[var]['column']).sum().reset_index()
        df_mc_temp['nvar'] = df_mc_temp['nvar'].apply(np.sqrt)
        mc_widths = dict[var]['width'][np.digitize(df_mc_temp[dict[var]['column']], dict[var]['bins']) - 1]

        df_qcd = df_mc.loc[(df_mc['process'] == 'QCD')]
        df_qcd = df_qcd.groupby(dict[var]['column']).sum().reset_index()
        df_qcd['nvar'] = df_qcd['nvar'].apply(np.sqrt)
        qcd_widths = dict[var]['width'][np.digitize(df_qcd[dict[var]['column']], dict[var]['bins']) - 1]

        df_residual = df_mc.loc[(df_mc['process'] != 'QCD')]
        df_residual = df_residual.groupby(dict[var]['column']).sum().reset_index()
       	df_residual['nvar'] = df_residual['nvar'].apply(np.sqrt)
        residual_widths = dict[var]['width'][np.digitize(df_residual[dict[var]['column']], dict[var]['bins']) - 1]

        plt.hist(df_mc_temp[dict[var]['column']], bins=dict[var]['bins'], histtype='step', color='cyan', linewidth=2, label='Total Standard Model', weights=df_mc_temp['n']/mc_widths)
        plt.hist(df_qcd[dict[var]['column']], bins=dict[var]['bins'], histtype='step', color='green', linestyle='dashed', linewidth=2, label='QCD', weights=df_qcd['n']/qcd_widths)
        plt.hist(df_residual[dict[var]['column']], bins=dict[var]['bins'], histtype='step', color='red', linestyle='dashed', linewidth=2, label='V + jets, $t \overline{t}$, Residual SM', weights=df_residual['n']/residual_widths)

    plt.yscale('log')
    plt.title('CMS $FAST-RA1$', loc='left')
    plt.title('35.9$fb^{-1}$', loc='right')
    plt.xlabel(dict[var]['title'], horizontalalignment='right')
    plt.legend(loc='best', fontsize='small')
    if not args.NoOutput:
        plt.savefig(var + '.pdf')
        print('Saved ' + var + '.pdf output file')
    if not args.NoX:
        plt.show()
