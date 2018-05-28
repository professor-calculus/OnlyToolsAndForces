#!/usr/bin/env python
import math
import os
import pandas as pd
import numpy as np
import uproot
import matplotlib
from matplotlib import rc
matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
matplotlib.rcParams['text.latex.preamble'].append(r'\usepackage{amsmath}')
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import argparse as a
import warnings
from tqdm import tqdm
from text_histogram import histogram
import itertools


parser = a.ArgumentParser(description='ROOTCuts for Experimental Analysis')
parser.add_argument('-f', '--files', nargs='*', required=True, help='Path to flat tree .root file(s)')
parser.add_argument('-t', '--type', required=True, help='Type of sample: SIGNAL, QCD, TTJETS, WJETS, ZJETS, etc')
parser.add_argument('-l', '--Lumi', type=float, default=1., help='Luminosity in pb')
parser.add_argument('--Msq', type=float, default=1000., help='Squark mass in GeV/c**2')
parser.add_argument('--Mlsp', type=float, default=10., help='LSP mass in GeV/c**2')
parser.add_argument('-c', '--CrossSec', type=float, default=1., help='Cross-Section in inverse pb')
parser.add_argument('-p', '--Prospino', default=None, help='Prospino input (reversed). Takes priority over -c')
parser.add_argument('-x', '--NoX', action='store_true', help='This argument suppresses showing plots via X-forwarding')
parser.add_argument('--kdeplot', action='store_true', help='Use kdeplot in Seaborn instead of matplotlib histogram')
parser.add_argument('--kdeplot_fill', action='store_true', help='Same as --kdeplot but area under each line is filled')
parser.add_argument('-o', '--NoOutput', action='store_true', help='This argument suppresses the output of PDF plots')
parser.add_argument('--OutDir', default='ROOTAnalysis_output', help='Where to write the output')
parser.add_argument('-v', '--verbose', action='store_true', help='Verbose mode, shows lepton vetoes, no of events which pass cuts etc.')
args=parser.parse_args()

#If not running on signal:
print('Running on {0} MC sample'.format(args.type))

#First we open the Delphes root file.
tree = uproot.open(args.files[0])["doubleBFatJetPairTree"]
events = uproot.open(args.files[0])["eventCountTree"]
#tree.recover()

nentries = 0.
nEvents = events.arrays(["nEvtsRunOver"], outputtype=tuple)
for nevts in nEvents:
    nentries += nevts[0]

#Let's create a dataframe to store the output in...
columns = ['Type', 'M_sq', 'M_lsp', 'crosssec', 'evtWeight', 'HT', 'MHT', 'NJet', 'NSlimBJet', 'NDoubleBTag', 'FatBJetMass_A', 'FatBJetMass_B']
df = pd.DataFrame(columns=columns)

#Also a dataframe to store the bins:
df_binned = pd.DataFrame(columns=['Type', 'M_sq', 'M_lsp', 'HT_bin', 'MHT_bin', 'n_Jet_bin', 'n_doubleBJet_bin', 'Yield'])

#Read in from Prospino
if args.Prospino:
    df_xsec = pd.read_csv(args.Prospino, delimiter=r'\s+')
    xsec = df_xsec['NLO_ms[pb]'].sum()
else:
    xsec = args.CrossSec

print('Cross-section = {}pb'.format(xsec))


#Make the output directories
directory = args.OutDir + '_{0}'.format(args.type)
suffix = 1
while os.path.exists(directory):
    suffix += 1
    directory = args.OutDir + '_{0}_{1}'.format(args.type, suffix)
print('Files will be written to: {0}'.format(directory))
os.makedirs(directory)

#Weight events to sum to Lumi*CrossSec
eventweight = (args.Lumi * xsec)/float(nentries)

MHT_bins = np.array([200., 400., 600., 900., 999999.])
HT_bins = np.array([1500., 2500., 3500., 99999.])
n_Jet_bins = np.array([6, 99])
n_doubleBJet_bins = np.array([0,1,2,99])

M_Z = 91.188


combined_weight, HT, MHT, NJet, NSlimBJet, fatJetA_bTagDiscrim, fatJetB_bTagDiscrim, fatJetA_mass, fatJetB_mass = tree.arrays(["weight_combined", "ht", "mht", "nrSlimJets", "nrSlimBJets", "fatJetA_doubleBtagDiscrim", "fatJetB_doubleBtagDiscrim", "fatJetA_softDropMassPuppi", "fatJetB_softDropMassPuppi"], outputtype=tuple)

sample_type = []
msq = []
mlsp = []
crosssec = []
mht = []
met = []
ht = []
N_doubleBJet = []
N_jet = []
eventWeight = []

fatDoubleBJet_A_mass = []
fatDoubleBJet_B_mass = []
fatDoubleBJet_A_discrim = []
fatDoubleBJet_B_discrim = []

cut_mht = []

binned_msq = []
binned_mlsp = []
binned_type = []
binned_HT_bin = []
binned_MHT_bin = []
binned_N_jet_bin = []
binned_N_doublebjet_bin = []
binned_yield = []

for mhtBin in [200, 400, 600, 900]:
    for htBin in [1500, 2500, 3500]:
        for nJetBin in [6]:
            for nDoubleBJetBin in [0,1,2]:
                binned_msq.append(args.Msq)
                binned_mlsp.append(args.Mlsp)
                binned_type.append(args.type)
                binned_HT_bin.append(htBin)
                binned_MHT_bin.append(mhtBin)
                binned_N_jet_bin.append(nJetBin)
                binned_N_doublebjet_bin.append(nBJetBin)
                binned_yield.append(0.)
binned_msq.append(args.Msq)
binned_mlsp.append(args.Mlsp)

eventpass = 0.

DoubleBDiscrim = 0.3 #Set this to be loose, tight WP etc.


for combined_weight_i, HT_i, MHT_i, NJet_i, NSlimBJet_i, fatJetA_bTagDiscrim_i, fatJetB_bTagDiscrim_i, fatJetA_mass_i, fatJetB_mass_i \
                                                 in tqdm(itertools.izip(combined_weight, HT, MHT, NJet, NSlimBJet), total=int(nentries), desc='Go Go Go!'):
    n_doublebjet = 0
    NJet6 = False
    HT1500 = False
    MHT200 = False
    DoubleBJet_pass = False

    weight = eventweight
    if args.verbose:
        print(weight)

    #The easy, pre-calculated variables:
    sample_type.append(args.type)
    msq.append(args.Msq)
    mlsp.append(args.Mlsp)
    crosssec.append(xsec)
    eventWeight.append(eventweight)
    mht.append(MHT_i)
    ht.append(HT_i)
    N_jet.append(NJet_i)
    fatDoubleBJet_A_discrim.append(fatJetA_bTagDiscrim_i)
    fatDoubleBJet_B_discrim.append(fatJetB_bTagDiscrim_i)

    # Number of double b-tagged jets
    if fatJetA_bTagDiscrim_i > DoubleBDiscrim:
        fatDoubleBJet_A_mass.append(fatJetA_mass_i)
        if (85. < fatJetA_mass_i < 145.):
            n_doublebjet += 1
    else:
        fatDoubleBJet_A_mass.append(-1.)
    if fatJetB_bTagDiscrim_i > DoubleBDiscrim:
        fatDoubleBJet_B_mass.append(fatJetB_mass_i)
        if (85. < fatJetB_mass_i < 145.):
            n_doublebjet += 1
    else:
        fatDoubleBJet_B_mass.append(-1.)
    
    N_doubleBJet.append(n_doublebjet)

    if MHT_i > 200.:
        MHT200 = True
    if HT_i > 1500.:
        HT1500 = True
    if NJet_i > 5:
        NJet6 = True

    if n_doublebjet > 0 or NSlimBJet_i > 1:
        DoubleBJet_pass = True

    All_Cuts = [NJet6, HT1500, MHT200, DoubleBJet_pass]
    if args.verbose:
        print(All_Cuts)
    if All_Cuts.count(False) == 0 and n_bjet in [2, 3] and HT > 1200.:
        #'M_sq', 'M_lsp', 'HT_bin', 'MHT_bin', 'n_Jet_bin', 'n_bJet_bin', 'Yield'
        binned_msq.append(args.Msq)
        binned_mlsp.append(args.Mlsp)
        binned_type.append(args.type)
        if args.verbose:
            print(HT)
            print(HT_bins)
        binned_HT_bin.append(HT_bins[np.digitize([HT], HT_bins)[0] - 1])
        binned_MHT_bin.append(MHT_bins[np.digitize([mht_temp], MHT_bins)[0] - 1])
        binned_N_jet_bin.append(n_Jet_bins[np.digitize([n_jet], n_Jet_bins)[0] - 1])
        binned_N_bjet_bin.append(n_doubleBJet_bins[np.digitize([n_doublebjet], n_doubleBJet_bins)[0] - 1])
        binned_yield.append(weight)
        eventpass += 1.

    if args.verbose:
        print('{0} events passed so far...'.format(eventpass))

percentpass = 100.*float(eventpass)/nentries
print('{0} of {1}, or {2} percent of events passed cuts'.format(int(eventpass), int(nentries), percentpass))

print('\n Signal Region:')
df_binned = pd.DataFrame({
    'Type': binned_type,
    'M_sq': binned_msq,
    'M_lsp': binned_mlsp,
    'HT_bin': binned_HT_bin,
    'MHT_bin': binned_MHT_bin,
    'n_Jet_bin': binned_N_jet_bin,
    'n_DoubleBJet_bin': binned_N_doublebjet_bin,
    'Yield': binned_yield,
    })
print(df_binned)
df_binned = df_binned.groupby(by=['Type', 'M_sq', 'M_lsp', 'HT_bin', 'MHT_bin', 'n_Jet_bin', 'n_DoubleBJet_bin']).sum()
df_binned.reset_index(inplace=True)
print(df_binned)


if not args.NoOutput:
    df_binned.to_csv(os.path.join(directory, 'ROOTCuts_binned.txt'), sep='\t', index=False)


df = pd.DataFrame({
    'Type': sample_type,
    'M_sq': msq,
    'M_lsp': mlsp,
    'crosssec': crosssec,
    'MHT': mht,
    'HT': ht,
    'NJet': N_jet,
    'NBJet': N_bjet,
    'NDoubleBJet': N_doubleBJet,
    'FatDoubleBJetA_mass': fatDoubleBJet_A_mass,
    'FatDoubleBJetB_mass': fatDoubleBJet_B_mass,
    'FatDoubleBJetA_discrim': fatDoubleBJet_A_discrim,
    'FatDoubleBJetB_discrim': fatDoubleBJet_B_discrim,
    })

print(df)
if not args.NoOutput:
    df.to_csv(os.path.join(directory, 'ROOTAnalysis.txt'), sep='\t', index=False)


plottables = ['MHT', 'HT', 'NJet', 'NDoubleBJet']


bins_HT = np.linspace(0.,5000.,160)
bins_MHT = np.linspace(0.,2000.,200)
bins_DelR = np.linspace(0.,5.,100)
bins_BMass = np.linspace(0.,500.,100)
bins_njet = np.arange(0, 20, 1)
bins_ndoublebjet = np.arange(0, 3, 1)


dict = {'MHT': {'bins': bins_MHT, 'title': 'Missing $H_{T}$ / GeV'},
        'HT': {'bins': bins_HT, 'title': 'Total $H_{T}$ / GeV'},
        'NJet': {'bins': bins_njet, 'title': 'Number of Jets'},
        'NDoubleBJet': {'bins': bins_ndoublebjet, 'title': 'Number of Double-$b$-tagged Fat Jets'},
        }

for thing in plottables:
    print('Plot of ' + thing)
    histogram(df[thing], buckets=20)
    plt.clf()
    if args.kdeplot or args.kdeplot_fill:
        sns.kdeplot(df[thing], shade=args.kdeplot_fill)
    else:
        plt.hist(df[thing], bins=dict[thing]['bins'])
    plt.xlabel(dict[thing]['title'])
    if not args.NoOutput:
        plt.savefig(os.path.join(directory, thing + '.pdf'))
        print('Saved ' + os.path.join(directory, thing + '.pdf') + ' output file')
    if not args.NoX:
        plt.show()
