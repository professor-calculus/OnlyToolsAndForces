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


parser = a.ArgumentParser(description='ROOTCuts for Experimental Analysis')
parser.add_argument('-f', '--files', nargs='*', required=True, help='Path to flat tree .root file(s)')
parser.add_argument('-t', '--type', required=True, help='Type of sample: SIGNAL, QCD, TTJETS, WJETS, ZJETS, etc')
parser.add_argument('-l', '--Lumi', type=float, default=1., help='Luminosity in pb')
parser.add_argument('--Msq', type=float, default=1000., help='Squark mass in GeV/c**2')
parser.add_argument('--Mlsp', type=float, default=10., help='LSP mass in GeV/c**2')
parser.add_argument('-c', '--CrossSec', type=float, default=1., help='Cross-Section in pb')
parser.add_argument('-p', '--Prospino', default=None, help='Prospino input (reversed). Takes priority over -c')
parser.add_argument('-x', '--NoX', action='store_true', help='This argument suppresses showing plots via X-forwarding')
parser.add_argument('--kdeplot', action='store_true', help='Use kdeplot in Seaborn instead of matplotlib histogram')
parser.add_argument('--kdeplot_fill', action='store_true', help='Same as --kdeplot but area under each line is filled')
parser.add_argument('-o', '--NoOutput', action='store_true', help='This argument suppresses the output of PDF plots')
parser.add_argument('--OutDir', default='ROOTAnalysis_Blind_output', help='Where to write the output')
parser.add_argument('-v', '--verbose', action='store_true', help='Verbose mode, shows lepton vetoes, no of events which pass cuts etc.')
args=parser.parse_args()

#If not running on signal:
print('Running on {0} sample'.format(args.type))

#Now the file has opened nicely, let's define some useful functions:
def Delta_Phi( Phi1, Phi2 ):
    if Phi1 - Phi2 > math.pi:
        delPhi = Phi1 - Phi2 - 2.*math.pi
    elif Phi1 - Phi2 < -1.*math.pi:
        delPhi = Phi1 - Phi2 + 2.*math.pi
    else:
        delPhi = Phi1 - Phi2
    return math.fabs(delPhi);

def Delta_R( eta1, phi1, eta2, phi2 ):
    dR2 = (eta1 - eta2)**2 + (Delta_Phi(phi1, phi2))**2
    dR = math.sqrt(dR2)
    return dR;

def Transverse_Mass( PT1, PT2, Phi1, Phi2 ):
    dPhi = Delta_Phi(Phi1, Phi2)
    m_T2 = 2.*PT1*PT2*( 1 - math.cos(dPhi) )
    m_T = math.sqrt(m_T2)
    return m_T;

def Invariant_Mass(PT1, PT2, Eta1, Eta2, Phi1, Phi2):
    dPhi = Delta_Phi(Phi1, Phi2)
    m2 = 2.*PT1*PT2*(math.cosh(Eta1-Eta2) - math.cos(dPhi))
    m = math.sqrt(m2)
    return m;

# Save original command for later use
commandString = ' '.join(sys.argv[0:])
print(commandString)

# Global variables defined once:
MHT_bins = np.array([200., 400., 600., 999999.])
HT_bins = np.array([1500., 2500., 3500., 99999.])
n_Jet_bins = np.array([6, 99])
n_doubleBJet_bins = np.array([0,1,2,99])
n_Muon_bins = np.array([-1,0,1,2,999])
DoubleBDiscrim = 0.3 #Set this to be loose, tight WP etc.
M_Z = 91.188

# Read in from Prospino
if args.Prospino:
    df_xsec = pd.read_csv(args.Prospino, delimiter=r'\s+')
    xsec = df_xsec['NLO_ms[pb]'].sum()
else:
    xsec = args.CrossSec

print('Cross-section = {}pb'.format(xsec))

print('Looping over {0} files'.format(len(args.files)))

# Make the output directories
directory = args.OutDir + '_{0}'.format(args.type)
suffix = 1
while os.path.exists(directory):
    suffix += 1
    directory = args.OutDir + '_{0}_{1}'.format(args.type, suffix)
thedirectories = '{0}_{1}_[{2}-{3}]'.format(args.OutDir, args.type, suffix, suffix+len(args.files)-1)
print('Output will be written to {0}'.format(thedirectories))

# Total number of events being run over:
nentries = 0.
for thefile in args.files:
    events = uproot.open(thefile)["eventCountTree"]
    nEvents = events.arrays(["nEvtsRunOver"], outputtype=tuple)
    for nevts in nEvents[0]:
        nentries += nevts
print('{0} events total')

### NEW! Loop over files and write to separate output, then combine later
for thefile in tqdm(args.files, total=len(args.files), desc='File:'):

    #Make the output directories
    directory = args.OutDir + '_{0}'.format(args.type)
    suffix = 1
    while os.path.exists(directory):
        suffix += 1
        directory = args.OutDir + '_{0}_{1}'.format(args.type, suffix)
    os.makedirs(directory)

    # Save original command for later use
    if not args.NoOutput:
        f = open(os.path.join(directory, 'command.txt'), 'w')
        f.write(commandString)
        f.close()

    #Weight events to sum to Lumi*CrossSec
    eventweight = (args.Lumi * xsec)/float(nentries)

    sample_type = []
    msq = []
    mlsp = []
    crosssec = []
    mht = []
    ht = []
    N_jet = []
    N_fatJet = []
    LeadJetPt = []
    eventWeight = []
    NoEntries = []

    n_muons = []
    n_selectedMuons = []
    muon_MHT_transverse_mass = []
    muons_inv_mass = []

    cut_mht = []

    eventCounter = 0

    DoubleBDiscrim = 0.3 #Set this to be loose, tight WP etc.

    for combined_weight, HT, MHT, MHT_phi, NJet, NFatJet, LeadSlimJet_p4, muonA_p4, muonB_p4, nMuons \
                                                    in tqdm(uproot.iterate(thefile, "doubleBFatJetPairTree", ["weight_combined", "ht", "mht", "mht_phi", "nrSlimJets", "nrFatJets", "slimJetA_p4", "muonA_p4", "muonB_p4", "nrMuons"], entrysteps=10000, outputtype=tuple)):
        for combined_weight_i, HT_i, MHT_i, MHT_phi_i, NJet_i, NFatJet_i, LeadSlimJet_p4_i, muonA_p4_i, muonB_p4_i, nMuons_i \
                                                        in tqdm(zip(combined_weight, HT, MHT, MHT_phi, NJet, NFatJet, LeadSlimJet_p4, muonA_p4, muonB_p4, nMuons), initial=eventCounter, total=nentries, desc='Go go go!'):

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
            N_fatJet.append(NFatJet_i)
            n_muons.append(nMuons_i)
            LeadJetPt.append(LeadSlimJet_p4_i.pt)
            NoEntries.append(nentries)

            # Transverse mass between Missing-HT and muon (in case of one muon)
            if nMuons_i == 1:
                muon_MHT_mT = Transverse_Mass(muonA_p4_i.pt, MHT_i, muonA_p4_i.phi(), MHT_phi_i)
            else:
                muon_MHT_mT = 0.
            muon_MHT_transverse_mass.append(muon_MHT_mT)

            # Invariant mass of muons (if 2 muons)
            if nMuons_i == 2:
                muons_Minv = Invariant_Mass(muonA_p4_i.pt, muonB_p4_i.pt, muonA_p4_i.eta, muonB_p4_i.eta, muonA_p4_i.phi(), muonB_p4_i.phi())
            else:
                muons_Minv = 0.
            muons_inv_mass.append(muons_Minv)

            # Number of selected muons (i.e. meets other cuts)
            if nMuons_i == 0:
                nMuons_selected = 0
            elif ((nMuons_i == 1) and (muon_MHT_mT < 100.)):
                nMuons_selected = 1
            elif ((nMuons_i == 2) and (muons_Minv > 75.) and (muons_Minv < 105.)):
                nMuons_selected = 2
            else:
                nMuons_selected = -1

            n_selectedMuons.append(nMuons_selected)

        # Keeps the event counter updated
        eventCounter += 10000


    df = pd.DataFrame({
        'Type': sample_type,
        'M_sq': msq,
        'M_lsp': mlsp,
        'crosssec': crosssec,
        'MHT': mht,
        'HT': ht,
        'NJet': N_jet,
        'NFatJet': N_fatJet,
        'LeadSlimJet_Pt': LeadJetPt,
        'nMuons': n_muons,
        'Muon_MHT_TransMass': muon_MHT_transverse_mass,
        'Muons_InvMass': muons_inv_mass,
        'NoEntries': NoEntries
        })

    print(df)
    if not args.NoOutput:
        df.to_csv(os.path.join(directory, 'ROOTAnalysis.txt'), sep='\t', index=False)


    plottables = ['MHT', 'HT', 'NJet', 'nMuons', 'LeadSlimJet_Pt']


    bins_HT = np.linspace(0.,5000.,160)
    bins_MHT = np.linspace(0.,2000.,200)
    bins_DelR = np.linspace(0.,5.,100)
    bins_njet = np.arange(0, 20, 1)
    bins_nmuons = np.arange(0, 10, 1)

    dict = {'MHT': {'bins': bins_MHT, 'title': 'Missing $H_{T}$ / GeV'},
            'HT': {'bins': bins_HT, 'title': 'Total $H_{T}$ / GeV'},
            'NJet': {'bins': bins_njet, 'title': 'Number of Jets'},
            'nMuons': {'bins': bins_nmuons, 'title': 'Number of Muons'},
            'LeadSlimJet_Pt': {'bins': bins_MHT, 'title': 'Lead AK4 Jet P_{T}'},
            }

    for thing in plottables:
        print('Plot of ' + thing)
        df_reduced = df.iloc[:1000]
        histogram(df_reduced[thing], buckets=20)
        plt.figure()
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
