#!/usr/bin/env python
import math
import os
import pandas as pd
import numpy as np
import uproot
import matplotlib
matplotlib.use('Agg')
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
from itertools import combinations


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
parser.add_argument('--OutDir', default='ROOTAnalysis_output', help='Where to write the output')
parser.add_argument('-v', '--verbose', action='store_true', help='Verbose mode, shows lepton vetoes, no of events which pass cuts etc.')
args=parser.parse_args()

# If not running on signal:
print('Running on {0} MC sample'.format(args.type))

# Let's define some useful functions:
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

# Original command to run script
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

    # Make the output directories
    directory = args.OutDir + '_{0}'.format(args.type)
    suffix = 1
    while os.path.exists(directory):
        suffix += 1
        directory = args.OutDir + '_{0}_{1}'.format(args.type, suffix)
    os.makedirs(directory)

    # Original command to run script
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
    NoEntries = []
    mht = []
    met = []
    ht = []
    N_doubleBJet = []
    N_doublejet_mass_window = []
    N_jet = []
    N_fatJet = []
    N_bjet = []
    N_loosebjet = []
    LeadJetPt = []
    AK8DelR = []
    eventWeight = []

    fatDoubleBJet_A_mass = []
    fatDoubleBJet_B_mass = []
    fatDoubleBJet_A_pt = []
    fatDoubleBJet_B_pt = []
    fatDoubleBJet_A_discrim = []
    fatDoubleBJet_B_discrim = []
    maxFatJet_discrim = []
    fatDoubleBJet_maxDiscrim_mass = []

    nH2bb = []
    H2bb_1_DelR = []
    H2bb_2_DelR = []

    bb_Pair_1_DelR = []
    bb_Pair_2_DelR = []
    bb_Pair_1_Mass = []
    bb_Pair_2_Mass = []

    n_loosemuons = []
    n_tightmuons = []
    n_electrons = []
    n_photons = []
    n_tracks = []
    muon_MHT_transverse_mass = []
    muons_inv_mass = []

    cut_mht = []

    binned_msq = []
    binned_mlsp = []
    binned_type = []
    binned_HT_bin = []
    binned_MHT_bin = []
    binned_N_jet_bin = []
    binned_N_bJet_bin = []
    binned_N_bJet_actual = []
    binned_N_doublebjet_bin = []
    binned_N_doublebjet_masswindow_bin = []
    binned_N_muons = []
    binned_yield = []

    for mhtBin in [200, 400, 600]:
        for htBin in [1500, 2500, 3500]:
            for nJetBin in [6]:
                for nMuons in [-1, 0, 1, 2]:
                    for nDoubleBJetBin in [0, 1, 2]:
                        for nFatJetMass in [0, 1, 2]:
                            if nDoubleBJetBin == 0:
                                for i in [3,4,5]:
                                    binned_msq.append(args.Msq)
                                    binned_mlsp.append(args.Mlsp)
                                    binned_type.append(args.type)
                                    binned_HT_bin.append(htBin)
                                    binned_MHT_bin.append(mhtBin)
                                    binned_N_jet_bin.append(nJetBin)
                                    binned_N_bJet_bin.append(3)
                                    binned_N_bJet_actual.append(i)
                                    binned_N_doublebjet_bin.append(nDoubleBJetBin)
                                    binned_N_doublebjet_masswindow_bin.append(nFatJetMass)
                                    binned_N_muons.append(nMuons)
                                    binned_yield.append(0.)
                            elif nDoubleBJetBin == 1:
                                for i in [1,2,3,4,5]:
                                    binned_msq.append(args.Msq)
                                    binned_mlsp.append(args.Mlsp)
                                    binned_type.append(args.type)
                                    binned_HT_bin.append(htBin)
                                    binned_MHT_bin.append(mhtBin)
                                    binned_N_jet_bin.append(nJetBin)
                                    binned_N_bJet_bin.append(2)
                                    binned_N_bJet_actual.append(i)
                                    binned_N_doublebjet_bin.append(nDoubleBJetBin)
                                    binned_N_doublebjet_masswindow_bin.append(nFatJetMass)
                                    binned_N_muons.append(nMuons)
                                    binned_yield.append(0.)
                            elif nDoubleBJetBin == 2:
                                for i in [0,1,2,3,4]:
                                    binned_msq.append(args.Msq)
                                    binned_mlsp.append(args.Mlsp)
                                    binned_type.append(args.type)
                                    binned_HT_bin.append(htBin)
                                    binned_MHT_bin.append(mhtBin)
                                    binned_N_jet_bin.append(nJetBin)
                                    binned_N_bJet_bin.append(0)
                                    binned_N_bJet_actual.append(i)
                                    binned_N_doublebjet_bin.append(nDoubleBJetBin)
                                    binned_N_doublebjet_masswindow_bin.append(nFatJetMass)
                                    binned_N_muons.append(nMuons)
                                    binned_yield.append(0.)

    eventpass = 0.
    eventCounter = 0

    for combined_weight, HT, MHT, MHT_phi, NJet, NFatJet, NSlimLooseBJet, NSlimBJet, LeadSlimJet_p4, muonA_p4, muonB_p4, nLooseMuons, nTightMuons, nrElectrons, nrPhotons, nrTracks, fatJetA_bTagDiscrim, fatJetB_bTagDiscrim, fatJetA_mass, fatJetB_mass, fatJetA_p4, fatJetB_p4, bJetA_p4, bJetB_p4, bJetC_p4, bJetD_p4, nHiggs2bb, Higgs2bb_1_DelR, Higgs2bb_2_DelR \
                                                    in tqdm(uproot.iterate(thefile, "doubleBFatJetPairTree", ["weight_combined", "ht", "mht", "mht_phi", "nrSlimJets", "nrFatJets", "nrSepSlimLooseBJets", "nrSepSlimMediumBJets", "slimJetA_p4", "muonA_p4", "muonB_p4", "nrLooseMuons", "nrTightMuons", "nrElectrons", "nrPhotons", "nrTracks", "fatJetA_doubleBtagDiscrim", "fatJetB_doubleBtagDiscrim", "fatJetA_softDropMassPuppi", "fatJetB_softDropMassPuppi", "fatJetA_p4", "fatJetB_p4", "bJetA_p4", "bJetB_p4", "bJetC_p4", "bJetD_p4", "nHiggsTobb", "DelR_bb_Higgs1", "DelR_bb_Higgs2"], entrysteps=10000, outputtype=tuple)):
        for combined_weight_i, HT_i, MHT_i, MHT_phi_i, NJet_i, NFatJet_i, NSlimLooseBJet_i, NSlimBJet_i, LeadSlimJet_p4_i, muonA_p4_i, muonB_p4_i, nLooseMuons_i, nTightMuons_i, nrElectrons_i, nrPhotons_i, nrTracks_i, fatJetA_bTagDiscrim_i, fatJetB_bTagDiscrim_i, fatJetA_mass_i, fatJetB_mass_i, fatJetA_p4_i, fatJetB_p4_i, bJetA_p4_i, bJetB_p4_i, bJetC_p4_i, bJetD_p4_i, nHiggs2bb_i, Higgs2bb_1_DelR_i, Higgs2bb_2_DelR_i \
                                                        in tqdm(zip(combined_weight, HT, MHT, MHT_phi, NJet, NFatJet, NSlimLooseBJet, NSlimBJet, LeadSlimJet_p4, muonA_p4, muonB_p4, nLooseMuons, nTightMuons, nrElectrons, nrPhotons, nrTracks, fatJetA_bTagDiscrim, fatJetB_bTagDiscrim, fatJetA_mass, fatJetB_mass, fatJetA_p4, fatJetB_p4, bJetA_p4, bJetB_p4, bJetC_p4, bJetD_p4, nHiggs2bb, Higgs2bb_1_DelR, Higgs2bb_2_DelR), initial=eventCounter, total=nentries, desc='{0} events passed'.format(eventpass)):
            n_doublebjet = 0
            n_doublejet_mass_window = 0
            NJet6 = False
            HT1500 = False
            MHT200 = False
            DoubleBJet_pass = False
            Pass_ElectronVeto = False
            Pass_PhotonVeto = False
            Pass_TrackVeto = False

            weight = eventweight
            if args.verbose:
                print(weight)

            #The easy, pre-calculated variables:
            sample_type.append(args.type)
            msq.append(args.Msq)
            mlsp.append(args.Mlsp)
            crosssec.append(xsec)
            NoEntries.append(nentries)
            eventWeight.append(eventweight)
            mht.append(MHT_i)
            ht.append(HT_i)
            N_jet.append(NJet_i)
            N_fatJet.append(NFatJet_i)
            N_bjet.append(NSlimBJet_i)
            N_loosebjet.append(NSlimLooseBJet_i)
            LeadJetPt.append(LeadSlimJet_p4_i.pt)
            nH2bb.append(nHiggs2bb_i)
            H2bb_1_DelR.append(Higgs2bb_1_DelR_i)
            H2bb_2_DelR.append(Higgs2bb_2_DelR_i)

            # Set double-b-tag discrim to -2 (out of usual range) if fat jet does not exist.
            if NFatJet_i > 0:
                fatJetA_discrim_val = fatJetA_bTagDiscrim_i
                fatJetA_mass_val = fatJetA_mass_i
                if NFatJet_i > 1:
                    fatJetB_discrim_val = fatJetB_bTagDiscrim_i
                    fatJetB_mass_val = fatJetB_mass_i
                    if fatJetA_discrim_val > fatJetB_discrim_val:
                        fatDoubleBJet_maxDiscrim_mass.append(fatJetA_mass_val)
                    else:
                        fatDoubleBJet_maxDiscrim_mass.append(fatJetB_mass_val)
                else:
                    fatJetB_discrim_val = -2.
                    fatJetB_mass_val = -1.
                    fatDoubleBJet_maxDiscrim_mass.append(fatJetA_mass_val)
            else:
                fatJetA_discrim_val = -2.
                fatJetB_discrim_val = -2.
                fatJetA_mass_val = -1.
                fatJetB_mass_val = -1.
                fatDoubleBJet_maxDiscrim_mass.append(-1.)

            fatDoubleBJet_A_discrim.append(fatJetA_discrim_val)
            fatDoubleBJet_B_discrim.append(fatJetB_discrim_val)
            maxFatJet_discrim.append(max(fatJetA_discrim_val, fatJetB_discrim_val))

            n_loosemuons.append(nLooseMuons_i)
            n_tightmuons.append(nTightMuons_i)
            n_electrons.append(nrElectrons_i)
            n_photons.append(nrPhotons_i)
            n_tracks.append(nrTracks_i)

            # Number of double b-tagged jets
            fatDoubleBJet_A_mass.append(fatJetA_mass_val)
            if (fatJetB_p4_i.pt > 300.):
                if fatJetB_discrim_val > DoubleBDiscrim:
                    n_doublebjet += 1
                if (85. < fatJetB_mass_val < 145.):
                    n_doublejet_mass_window += 1

            fatDoubleBJet_B_mass.append(fatJetB_mass_val)
            if (fatJetB_p4_i.pt > 300.):
                if fatJetB_discrim_val > DoubleBDiscrim:
                    n_doublebjet += 1
                if (85. < fatJetB_mass_val < 145.):
                    n_doublejet_mass_window += 1
            
            N_doubleBJet.append(n_doublebjet)

            #Angular separation of AK8 jets:
            if NFatJet_i > 1:
                dR = Delta_R(fatJetA_p4_i.eta, fatJetA_p4_i.phi, fatJetB_p4_i.eta, fatJetB_p4_i.phi)
            else:
                dR = -1.
            AK8DelR.append(dR)

            if MHT_i > 200.:
                MHT200 = True
            if HT_i > 1500.:
                HT1500 = True
            if NJet_i > 5:
                NJet6 = True

            # Require at least 2 b-jets if only one double-b-jet and at least 3 b-jets if no double-b-jets.
            if ((n_doublebjet > 1) or (n_doublebjet == 1 and NSlimBJet_i > 0 and NSlimLooseBJet_i > 1) or (n_doublebjet == 0 and NSlimBJet_i > 2 and NSlimLooseBJet_i > 3)):
                DoubleBJet_pass = True

            # Invariant mass between single b (one pair if 1 double b jet, 2 if no double b)
            # Arranged such that if 1 double b then single b pair is 2 closest in dR from top 4 tags
            # If no double b then pick closest pair in dR, then other pair from top 4 tag scores
            temp_dR = 9999.
            temp_Mbb = [-1., -1.]
            temp_dR_both = [-1., -1.]
            vector_bjet = [bJetA_p4_i, bJetB_p4_i, bJetC_p4_i, bJetD_p4_i]
            temp_nb = min(4, NSlimLooseBJet_i)
            if NSlimLooseBJet_i > 1:
                for combo in combinations(vector_bjet[:temp_nb], 2):
                    if ((combo[0].pt < 10.) or (combo[1].pt < 10.)): continue
                    dR_ = Delta_R(combo[0].eta, combo[0].phi, combo[1].eta, combo[1].phi)
                    if dR_ < temp_dR:
                        temp_dR = dR_
                        temp_dR_both[0] = dR_
                        temp_Mbb[0] = Invariant_Mass(combo[0].pt, combo[1].pt, combo[0].eta, combo[1].eta, combo[0].phi, combo[1].phi)
                        if NSlimLooseBJet_i > 3:
                            other_bjets = []
                            for temp_bjet in vector_bjet:
                                if (temp_bjet != combo[0] and temp_bjet != combo[1]): other_bjets.append(temp_bjet)
                            if len(other_bjets) < 2: continue
                            if ((other_bjets[0].pt < 10.) or (other_bjets[1].pt < 10.)): continue
                            temp_Mbb[1] = Invariant_Mass(other_bjets[0].pt, other_bjets[1].pt, other_bjets[0].eta, other_bjets[1].eta, other_bjets[0].phi, other_bjets[1].phi)
                            temp_dR_both[1] = Delta_R(other_bjets[0].eta, other_bjets[0].phi, other_bjets[1].eta, other_bjets[1].phi)
            bb_Pair_1_Mass.append(temp_Mbb[0])
            bb_Pair_1_DelR.append(temp_dR_both[0])
            bb_Pair_2_Mass.append(temp_Mbb[1])
            bb_Pair_2_DelR.append(temp_dR_both[1])

            # Transverse mass between Missing-HT and muon (in case of one muon)
            if nTightMuons_i == 1:
                muon_MHT_mT = Transverse_Mass(muonA_p4_i.pt, MHT_i, muonA_p4_i.phi, MHT_phi_i)
            else:
                muon_MHT_mT = 0.
            muon_MHT_transverse_mass.append(muon_MHT_mT)

            # Invariant mass of muons (if 2 muons)
            if nTightMuons_i == 2:
                muons_Minv = Invariant_Mass(muonA_p4_i.pt, muonB_p4_i.pt, muonA_p4_i.eta, muonB_p4_i.eta, muonA_p4_i.phi, muonB_p4_i.phi)
            else:
                muons_Minv = 0.
            muons_inv_mass.append(muons_Minv)

            # Number of selected muons (i.e. meets other cuts)
            if nLooseMuons_i == 0:
                nMuons_selected = 0
            elif ((nTightMuons_i == 1) and (muon_MHT_mT < 100.)):
                nMuons_selected = 1
            elif ((nTightMuons_i == 2) and (muons_Minv > 75.) and (muons_Minv < 105.)):
                nMuons_selected = 2
            else:
                nMuons_selected = -1

            # Electron, Photon, Track vetoes
            if nrElectrons_i == 0:
                Pass_ElectronVeto = True
            if nrPhotons_i == 0:
                Pass_PhotonVeto = True
            if nrTracks_i == 0:
                Pass_TrackVeto = True

            All_Cuts = [NJet6, HT1500, MHT200, DoubleBJet_pass, Pass_ElectronVeto, Pass_PhotonVeto, Pass_TrackVeto]
            Most_Cuts = [NJet6, HT1500, MHT200, Pass_ElectronVeto, Pass_PhotonVeto, Pass_TrackVeto]
            if args.verbose:
                print(All_Cuts)
            if All_Cuts.count(False) == 0:
                #'M_sq', 'M_lsp', 'HT_bin', 'MHT_bin', 'n_Jet_bin', 'n_bJet_bin', 'Yield'
                binned_msq.append(args.Msq)
                binned_mlsp.append(args.Mlsp)
                binned_type.append(args.type)
                if args.verbose:
                    print(HT)
                    print(HT_bins)
                binned_HT_bin.append(HT_bins[np.digitize([HT_i], HT_bins)[0] - 1])
                binned_MHT_bin.append(MHT_bins[np.digitize([MHT_i], MHT_bins)[0] - 1])
                binned_N_jet_bin.append(n_Jet_bins[np.digitize([NJet_i], n_Jet_bins)[0] - 1])
                binned_N_bJet_actual.append(NSlimBJet_i)
                if n_doublebjet == 0:
                    binned_N_bJet_bin.append(3)
                elif n_doublebjet == 1:
                    binned_N_bJet_bin.append(2)
                elif n_doublebjet == 2:
                    binned_N_bJet_bin.append(0)
                binned_N_doublebjet_bin.append(n_doubleBJet_bins[np.digitize([n_doublebjet], n_doubleBJet_bins)[0] - 1])
                binned_N_muons.append(n_Muon_bins[np.digitize([nMuons_selected], n_Muon_bins)[0] - 1])
                binned_yield.append(weight)
                eventpass += 1.
            '''
            elif ((Most_Cuts.count(False) == 0) & (n_doublebjet == 1) & (NSlimBJet_i == 1)):
                #'M_sq', 'M_lsp', 'HT_bin', 'MHT_bin', 'n_Jet_bin', 'n_bJet_bin', 'Yield'
                binned_msq.append(args.Msq)
                binned_mlsp.append(args.Mlsp)
                binned_type.append(args.type)
                if args.verbose:
                    print(HT)
                    print(HT_bins)
                binned_HT_bin.append(HT_bins[np.digitize([HT_i], HT_bins)[0] - 1])
                binned_MHT_bin.append(MHT_bins[np.digitize([MHT_i], MHT_bins)[0] - 1])
                binned_N_jet_bin.append(n_Jet_bins[np.digitize([NJet_i], n_Jet_bins)[0] - 1])
                binned_N_bJet_actual.append(NSlimBJet_i)
                binned_N_bJet_bin.append(NSlimBJet_i)
                binned_N_doublebjet_bin.append(n_doubleBJet_bins[np.digitize([n_doublebjet], n_doubleBJet_bins)[0] - 1])
                binned_N_muons.append(n_Muon_bins[np.digitize([nMuons_selected], n_Muon_bins)[0] - 1])
                binned_yield.append(weight)
                eventpass += 1.
            '''

            if args.verbose:
                print('{0} events passed so far...'.format(eventpass))

        # Keeps the event counter updated
        eventCounter += 10000

    percentpass = 100.*float(eventpass)/nentries
    print('{0} of {1}, or {2} percent of events passed cuts'.format(int(eventpass), int(nentries), percentpass))

    print('\n All which pass baseline selection:')
    df_binned = pd.DataFrame({
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
    print(df_binned)

    print('\n Signal Region:')
    df_SR = df_binned.loc[df_binned['n_Muons_bin'] == 0]
    df_SR = df_SR.groupby(by=['Type', 'M_sq', 'M_lsp', 'HT_bin', 'MHT_bin', 'n_Jet_bin', 'n_bJet_bin', 'n_DoubleBJet_bin', 'n_Muons_bin']).sum()
    df_SR.reset_index(inplace=True)
    print(df_SR)

    print('\n 0b1mu Control Region:')
    df_SM0b = df_binned.loc[((df_binned['n_Muons_bin'] == 1) & (df_binned['n_bJet_actual'] == 0))]
    df_SM0b = df_SM0b.groupby(by=['Type', 'M_sq', 'M_lsp', 'HT_bin', 'MHT_bin', 'n_Jet_bin', 'n_bJet_bin', 'n_DoubleBJet_bin', 'n_Muons_bin']).sum()
    df_SM0b.reset_index(inplace=True)
    print(df_SM0b)

    print('\n 1b1mu Control Region:')
    df_SM1b = df_binned.loc[((df_binned['n_Muons_bin'] == 1) & (df_binned['n_bJet_actual'] == 1))]
    df_SM1b = df_SM1b.groupby(by=['Type', 'M_sq', 'M_lsp', 'HT_bin', 'MHT_bin', 'n_Jet_bin', 'n_bJet_bin', 'n_DoubleBJet_bin', 'n_Muons_bin']).sum()
    df_SM1b.reset_index(inplace=True)
    print(df_SM1b)

    print('\n 2b1mu Control Region:')
    df_SM2b = df_binned.loc[((df_binned['n_Muons_bin'] == 1) & (df_binned['n_bJet_actual'] == 2))]
    df_SM2b = df_SM2b.groupby(by=['Type', 'M_sq', 'M_lsp', 'HT_bin', 'MHT_bin', 'n_Jet_bin', 'n_bJet_bin', 'n_DoubleBJet_bin', 'n_Muons_bin']).sum()
    df_SM2b.reset_index(inplace=True)
    print(df_SM2b)

    print('\n 1mu Control Region:')
    df_SM = df_binned.loc[((df_binned['n_Muons_bin'] == 1))]
    df_SM = df_SM.groupby(by=['Type', 'M_sq', 'M_lsp', 'HT_bin', 'MHT_bin', 'n_Jet_bin', 'n_bJet_bin', 'n_DoubleBJet_bin', 'n_Muons_bin']).sum()
    df_SM.reset_index(inplace=True)
    print(df_SM)

    print('\n 0b2mu Control Region:')
    df_DM0b = df_binned.loc[((df_binned['n_Muons_bin'] == 2) & (df_binned['n_bJet_actual'] == 0))]
    df_DM0b = df_DM0b.groupby(by=['Type', 'M_sq', 'M_lsp', 'HT_bin', 'MHT_bin', 'n_Jet_bin', 'n_bJet_bin', 'n_DoubleBJet_bin', 'n_Muons_bin']).sum()
    df_DM0b.reset_index(inplace=True)
    print(df_DM0b)

    print('\n 1b2mu Control Region:')
    df_DM1b = df_binned.loc[((df_binned['n_Muons_bin'] == 2) & (df_binned['n_bJet_actual'] == 1))]
    df_DM1b = df_DM1b.groupby(by=['Type', 'M_sq', 'M_lsp', 'HT_bin', 'MHT_bin', 'n_Jet_bin', 'n_bJet_bin', 'n_DoubleBJet_bin', 'n_Muons_bin']).sum()
    df_DM1b.reset_index(inplace=True)
    print(df_DM1b)

    print('\n 2b2mu Control Region:')
    df_DM2b = df_binned.loc[((df_binned['n_Muons_bin'] == 2) & (df_binned['n_bJet_actual'] == 2))]
    df_DM2b = df_DM2b.groupby(by=['Type', 'M_sq', 'M_lsp', 'HT_bin', 'MHT_bin', 'n_Jet_bin', 'n_bJet_bin', 'n_DoubleBJet_bin', 'n_Muons_bin']).sum()
    df_DM2b.reset_index(inplace=True)
    print(df_DM2b)

    print('\n 2mu Control Region:')
    df_DM = df_binned.loc[df_binned['n_Muons_bin'] == 2]
    df_DM = df_DM.groupby(by=['Type', 'M_sq', 'M_lsp', 'HT_bin', 'MHT_bin', 'n_Jet_bin', 'n_bJet_bin', 'n_DoubleBJet_bin', 'n_Muons_bin']).sum()
    df_DM.reset_index(inplace=True)
    print(df_DM)

    if not args.NoOutput:
        df_binned.to_csv(os.path.join(directory, 'ROOTCuts_binned.txt'), sep='\t', index=False)
        df_SR.to_csv(os.path.join(directory, 'Signal_Region.txt'), sep='\t', index=False)
        df_SM0b.to_csv(os.path.join(directory, 'SingleMuon_0b_Control_Region.txt'), sep='\t', index=False)
        df_SM1b.to_csv(os.path.join(directory, 'SingleMuon_1b_Control_Region.txt'), sep='\t', index=False)
        df_SM2b.to_csv(os.path.join(directory, 'SingleMuon_2b_Control_Region.txt'), sep='\t', index=False)
        df_SM.to_csv(os.path.join(directory, 'SingleMuon_Control_Region.txt'), sep='\t', index=False)
        df_DM0b.to_csv(os.path.join(directory, 'DoubleMuon_0b_Control_Region.txt'), sep='\t', index=False)
        df_DM1b.to_csv(os.path.join(directory, 'DoubleMuon_1b_Control_Region.txt'), sep='\t', index=False)
        df_DM2b.to_csv(os.path.join(directory, 'DoubleMuon_2b_Control_Region.txt'), sep='\t', index=False)
        df_DM.to_csv(os.path.join(directory, 'DoubleMuon_Control_Region.txt'), sep='\t', index=False)


    df = pd.DataFrame({
        'Type': sample_type,
        'M_sq': msq,
        'M_lsp': mlsp,
        'crosssec': crosssec,
        'NoEntries': NoEntries,
        'MHT': mht,
        'HT': ht,
        'NJet': N_jet,
        'NFatJet': N_fatJet,
        'NBJet': N_bjet,
        'NLooseBJet': N_loosebjet,
        'NDoubleBJet': N_doubleBJet,
        'LeadSlimJet_Pt': LeadJetPt,
        'FatDoubleBJetA_mass': fatDoubleBJet_A_mass,
        'FatDoubleBJetB_mass': fatDoubleBJet_B_mass,
        'FatDoubleBJetA_discrim': fatDoubleBJet_A_discrim,
        'FatDoubleBJetB_discrim': fatDoubleBJet_B_discrim,
        'MaxFatJetDoubleB_discrim': maxFatJet_discrim,
        'FatJet_MaxDoubleB_discrim_mass': fatDoubleBJet_maxDiscrim_mass,
        'FatJetAngularSeparation': AK8DelR,
        'nLooseMuons': n_loosemuons,
        'nTightMuons': n_tightmuons,
        'nElectrons': n_electrons,
        'nPhotons': n_photons,
        'nTracks': n_tracks,
        'Muon_MHT_TransMass': muon_MHT_transverse_mass,
        'Muons_InvMass': muons_inv_mass,
        'nHiggs2bb': nH2bb,
        'Higgs2bb_DelR_1': H2bb_1_DelR,
        'Higgs2bb_DelR_2': H2bb_2_DelR,
        'bb_Pair1_Mass': bb_Pair_1_Mass,
        'bb_Pair2_Mass': bb_Pair_2_Mass,
        'bb_Pair1_DelR': bb_Pair_1_DelR,
        'bb_Pair2_DelR': bb_Pair_2_DelR,
        })

    print(df)
    if not args.NoOutput:
        df.to_csv(os.path.join(directory, 'ROOTAnalysis.txt'), sep='\t', index=False)


    plottables = ['MHT', 'HT', 'NJet', 'NBJet', 'NDoubleBJet', 'NFatJet', 'nLooseMuons', 'nTightMuons', 'nElectrons', 'nPhotons', 'nTracks', 'MaxFatJetDoubleB_discrim', 'FatJet_MaxDoubleB_discrim_mass', 'LeadSlimJet_Pt', 'nHiggs2bb', 'Higgs2bb_DelR_1', 'bb_Pair1_Mass', 'bb_Pair2_Mass', 'bb_Pair1_DelR', 'bb_Pair2_DelR']


    bins_HT = np.linspace(0.,5000.,160)
    bins_MHT = np.linspace(0.,2000.,200)
    bins_DelR = np.linspace(0.,5.,100)
    bins_BMass = np.linspace(0.,500.,100)
    bins_njet = np.arange(0, 20, 1)
    bins_nfatjet = np.arange(0, 8, 1)
    bins_ndoublebjet = np.arange(-1, 4, 1)
    bins_nmuons = np.arange(0, 10, 1)
    bins_doublebdiscrim = np.linspace(-1., 1.)

    dict = {'MHT': {'bins': bins_MHT, 'title': 'Missing $H_{T}$ / GeV'},
            'HT': {'bins': bins_HT, 'title': 'Total $H_{T}$ / GeV'},
            'NJet': {'bins': bins_njet, 'title': 'Number of Jets'},
            'NBJet': {'bins': bins_njet, 'title': 'Number of $b$-tagged Jets'},
            'NDoubleBJet': {'bins': bins_ndoublebjet, 'title': 'Number of Double-$b$-tagged Fat Jets'},
            'NFatJet': {'bins': bins_nfatjet, 'title': 'Number of AK8 Fat Jets'},
            'nLooseMuons': {'bins': bins_nmuons, 'title': 'Number of Loose Muons'},
            'nTightMuons': {'bins': bins_nmuons, 'title': 'Number of Tight Muons'},
            'nElectrons': {'bins': bins_nmuons, 'title': 'Number of Electrons'},
            'nPhotons': {'bins': bins_nmuons, 'title': 'Number of Photons'},
            'nTracks': {'bins': bins_nmuons, 'title': 'Number of Isolated Tracks'},
            'MaxFatJetDoubleB_discrim': {'bins': bins_doublebdiscrim, 'title': 'Highest Double-b discriminator score'},
            'FatJet_MaxDoubleB_discrim_mass': {'bins': bins_BMass, 'title': 'Soft-Drop Mass of AK8 Jet with Highest Double-b discriminator score'},
            'LeadSlimJet_Pt': {'bins': bins_MHT, 'title': 'Lead AK4 Jet P_{T}'},
            'nHiggs2bb': {'bins': bins_ndoublebjet, 'title': 'Number of Higgs bosons decaying to bb (MC Truth)'},
            'Higgs2bb_DelR_1': {'bins': bins_DelR, 'title': 'Delta-R between bb pair from Higgs (MC Truth)'},
            'bb_Pair1_Mass': {'bins': bins_BMass, 'title': 'Inv. mass of best bb pair'},
            'bb_Pair2_Mass': {'bins': bins_BMass, 'title': 'Inv. mass of other bb pair'},
            'bb_Pair1_DelR': {'bins': bins_DelR, 'title': 'Delta-R between best bb pair'},
            'bb_Pair2_DelR': {'bins': bins_DelR, 'title': 'Delta-R between other bb pair'},
            }

    for thing in plottables:
        print('Plot of ' + thing)
        # df_reduced = df.iloc[:1000]
        # histogram(df_reduced[thing], buckets=20)
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

print('Iterated over all files')
