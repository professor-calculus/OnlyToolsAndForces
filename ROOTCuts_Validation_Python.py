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


parser = a.ArgumentParser(description='ROOTCuts in Python')
parser.add_argument('-d', '--delphes', nargs='*', required=True, help='Path to Delphes .root file(s)')
parser.add_argument('-b', '--BKG', action='store_true', help='Flag to indicate sample is BKG and not Signal')
parser.add_argument('-l', '--Lumi', type=float, default=1., help='Luminosity in pb')
parser.add_argument('--Msq', type=float, default=1000., help='Squark mass in GeV/c')
parser.add_argument('--Mlsp', type=float, default=10., help='LSP mass in GeV/c')
parser.add_argument('-c', '--CrossSec', type=float, default=1., help='Cross-Section in inverse pb')
parser.add_argument('-p', '--Prospino', default=None, help='Prospino input (reversed). Takes priority over -c')
parser.add_argument('-x', '--NoX', action='store_true', help='This argument suppresses showing plots via X-forwarding')
parser.add_argument('--kdeplot', action='store_true', help='Use kdeplot in Seaborn instead of matplotlib histogram')
parser.add_argument('--kdeplot_fill', action='store_true', help='Same as --kdeplot but area under each line is filled')
parser.add_argument('-o', '--NoOutput', action='store_true', help='This argument suppresses the output of PDF plots')
parser.add_argument('--OutDir', default='ROOTCuts_output', help='Where to write the output')
parser.add_argument('-v', '--verbose', action='store_true', help='Verbose mode, shows lepton vetoes, no of events which pass cuts etc.')
args=parser.parse_args()

#If not running on signal:
if args.BKG:
    print('Notice: Running over Background events (e.g. QCD, TTJets etc), not Signal!')

#First we open the Delphes root file.
tree = uproot.open(args.delphes[0])["Delphes"]
#tree.recover()

#Let's create a dataframe to store the output in...
columns = ['M_sq', 'M_lsp', 'crosssec', 'MET', 'MHT', 'HT', 'Higgs_PT', 'Higgs1_PT', 'Higgs2_PT', 'bJetsDelR', 'bDPhi', 'NJet', 'NBJet', 'NVeto']
df = pd.DataFrame(columns=columns)

#Also a dataframe to store the bins:
df_binned = pd.DataFrame(columns=['M_sq', 'M_lsp', 'HT_bin', 'MHT_bin', 'n_Jet_bin', 'n_bJet_bin', 'Yield'])

#Read in from Prospino
if args.Prospino:
    df_xsec = pd.read_csv(args.Prospino, delimiter=r'\s+')
    xsec = df_xsec['NLO_ms[pb]'].sum()
else:
    xsec = args.CrossSec

print('Cross-section = {}pb'.format(xsec))


#Make the output directories
directory = args.OutDir
suffix = 1
while os.path.exists(directory):
    suffix += 1
    directory = args.OutDir + '_{0}'.format(suffix)
print('Files will be written to: {0}'.format(directory))
os.makedirs(directory)

#Weight events to sum to Lumi*CrossSec
nentries = float(len(tree["MissingET.MET"]))
print('{0} Entries'.format(nentries))
eventweight = (args.Lumi * xsec)/(nentries)

MHT_bins = np.array([200., 400., 600., 900., 999999.])
HT_bins = np.array([1200., 99999.])
n_Jet_bins = np.array([6, 99])
n_bJet_bins = np.array([2,3,99])

M_Z = 91.188

leptonVetoPass = 0
jet1PTVeto = 0
chargedHadronFractionPass = 0
htPass = 0
mhtPass = 0
jetVetoPass = 0
mhtOverMetPass = 0
alphaTPass = 0
biasedDeltaPhiPass = 0

eventpass = 0.

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
    m2 = 2.*PT1*PT2*(math.cosh(Eta1-Eta2) - math.cos(Phi1-Phi2))
    m = math.sqrt(m2)
    return m;

def makeAlphaT(jets_phi, jets_pt, jets_eta, jets_mass, mht, ht):

    if len(jets_phi) < 2:
        return -1;

    et = []

    for pt, eta, phi, mass in zip(jets_pt, jets_eta, jets_phi, jets_mass):
        theta = 2*math.atan(math.exp(-1.*eta))
        et.append(math.sqrt(pt**2 + (mass*math.sin(theta))**2))

    nJet   = len(jets_pt)

    minDeltaHt = -1
    # loop over possible combinations
    for i in range( 1 << (nJet-1) ):
        deltaHt = 0
        # loop over jets
        for j in range(nJet):
            deltaHt += et[j] * ( 1 - 2 * (int(i>>j)&1) )
        if math.fabs(deltaHt) < minDeltaHt or minDeltaHt < 0:
            minDeltaHt = math.fabs(deltaHt)

    return 0.5 * ( ht - minDeltaHt ) / math.sqrt( ht*ht - mht*mht );

def alphaT_Thresholds(HT):
    if 200. < HT < 250.:
        threshold = 0.65
    elif 250. < HT < 300.:
        threshold = 0.6
    elif 300. < HT < 350.:
        threshold = 0.55
    elif 350. < HT < 400.:
        threshold = 0.53
    elif 400. < HT < 900.:
        threshold = 0.52
    else:
        threshold = 0.
    return threshold;

for evtweight, jet_mass, jet_pt, jet_phi, jet_eta, jet_btag, jet_nc, jet_nn, MET,\
    PID, UID, M1, PT, Eta, Phi, eEta, ePT, uEta, uPT, uPhi, pEta, pPT, pPhi in tree.iterate(["Event.Weight", "Jet.Mass", "Jet.PT", "Jet.Phi", "Jet.Eta", "Jet.BTag", "Jet.NCharged", "Jet.NNeutrals", "MissingET.MET", "Particle.PID",
                                                                                            "Particle.fUniqueID", "Particle.M1", "Particle.PT", "Particle.Eta", "Particle.Phi", "Electron.Eta", "Electron.PT",
                                                                                            "Muon.Eta", "Muon.PT", "Muon.Phi", "Photon.Eta", "Photon.PT", "Photon.Phi"], entrysteps=5000, outputtype=tuple):
    for evtweighti, jet_massi, jet_pti, jet_phii, jet_etai, jet_btagi, jet_nci, jet_nni, METi, PIDi, UIDi, M1i, PTi, Etai, Phii, eEtai, ePTi, uEtai, uPTi, uPhii, pEtai, pPTi, pPhii in tqdm(zip(evtweight, jet_mass, jet_pt, jet_phi, jet_eta, jet_btag, jet_nc, jet_nn, MET, PID, UID, M1, PT, Eta, Phi, eEta, ePT, uEta, uPT, uPhi, pEta, pPT, pPhi), total=5000, desc='Go Go Go!'):
        mht_x = 0.
        mht_y = 0.
        HT = 0.
        n_bjet = 0
        n_jet = 0
        nVeto = 0
        nJetVeto = 0
        nMuon_CR = 0
        SingleMuon_CR = 0
        DoubleMuon_CR = 0
        LeadJetPT100 = False
        NoVetoObjects = False
        NJet6 = False
        MhtOverMet1p25 = False
        HT200 = False
        MHT200 = False
        BDPhi0p5 = False
        LeadJetCHF = False

        DeltaR = []
        HiggsPT = []
        BDP = []

        goodjets_eta = []
        goodjets_phi = []
        goodjets_pt = []
        goodjets_mass = []

        bjets_eta = []
        bjets_phi = []
        bjets_pt = []

        weight = eventweight*evtweighti[0]
        if args.verbose:
            print(weight)

        bPID=[]
        bUID = []
        bM1 = []
        bPT = []
        bEta = []
        bPhi = []

        muonPT = []
        muonEta = []
        muonPhi = []

        higgs_UID = []

        d = {}
        d_jet = {}

        # Object event vetoes
        for eEtaj, ePTj in zip(eEtai, ePTi):
            if abs(eEtaj) < 2.5 and ePTj > 10.:
                if args.verbose:
                    print('electron PT: {0}, Eta: {1}'.format(ePTj, eEtaj))
                nVeto += 1
        for uEtaj, uPTj in zip(uEtai, uPTi):
            if abs(uEtaj) < 2.5 and uPTj > 10.:
                if args.verbose:
                    print('muon PT: {0}, Eta: {1}'.format(uPTj, uEtaj))
                nVeto += 1
        for pEtaj, pPTj in zip(pEtai, pPTi):
            if abs(pEtaj) < 2.5 and pPTj > 25.:
                if args.verbose:
                    print('photon PT: {0}, Eta: {1}'.format(pPTj, pEtaj))
                nVeto += 1


        # Number of jets, number of b-tagged jets, HT, MHT
        for PTj, BTagj, Etaj, Phij, Massj in zip(jet_pti, jet_btagi, jet_etai, jet_phii, jet_massi):
            if PTj > 40. and abs(Etaj) < 2.4:
                n_jet += 1
                mht_x += -1.*PTj*math.cos(Phij)
                mht_y += PTj*math.sin(Phij)
                HT += PTj
            if PTj > 40. and abs(Etaj) > 2.4:
                if args.verbose:
                    print('jet PT: {0}, Eta: {1}'.format(PTj, Etaj))
                nJetVeto += 1
        mht_temp = math.sqrt(mht_x**2 + mht_y**2)

        if mht_temp > 200.:
            MHT200 = True
        if HT > 200.:
            HT200 = True
        if n_jet > 5:
            NJet6 = True
        if nVeto == 0:
            NoVetoObjects = True
        if nJetVeto == 0:
            NoVetoJets = True

        # Biased Delta-Phi and Lead Jet CHF
        if n_jet > 1:
            if 0.1 < float(jet_nci[0])/float(jet_nci[0] + jet_nni[0]) < 0.95:
                LeadJetCHF = True
            if jet_pti[0] > 100.:
                LeadJetPT100 = True
            for PTj, Etaj, Phij in zip(jet_pti, jet_etai, jet_phii):
                if PTj > 40. and abs(Etaj) < 2.4:
                    jet_px, jet_py = PTj*math.cos(Phij),-PTj*math.sin(Phij)
                    newPhi = math.atan2(-mht_y-jet_py, mht_x+jet_px)

                    if newPhi - Phij > math.pi:
                        BDP.append(abs(newPhi - Phij - 2.*math.pi))
                    elif newPhi - Phij < -1.*math.pi:
                        BDP.append(abs(newPhi - Phij + 2.*math.pi))
                    else:
                        BDP.append(abs(newPhi - Phij))
            BDP_temp = min(BDP)
        else:
            BDP_temp = -1.
        
        if BDP_temp > 0.5:
            BDPhi0p5 = True

        # alpha_T
        temp_alpha_T = makeAlphaT(goodjets_phi, goodjets_pt, goodjets_eta, goodjets_mass, mht_temp, HT)

        # Missing-ET
        for METj in METi:
            met_temp = METj
        if met_temp != 0:
            if mht_temp/met_temp < 1.25:
                MhtOverMet1p25 = True

        All_Cuts = [LeadJetPT100, NoVetoObjects, NJet6, MhtOverMet1p25, MHT200, BDPhi0p5, LeadJetCHF]

        '''
        leptonVetoPass = 0
        jet1PTVeto = 0
        chargedHadronFractionPass = 0
        htPass = 0
        mhtPass = 0
        jetVetoPass = 0
        mhtOverMetPass = 0
        alphaTPass = 0
        biasedDeltaPhiPass = 0
        '''

        if NoVetoObjects:
            leptonVetoPass += 1
            if LeadJetPT100:
                jet1PTVeto += 1
                if LeadJetCHF:
                    chargedHadronFractionPass += 1
                    if HT200:
                        htPass += 1
                        if MHT200:
                            mhtPass += 1
                            if NoVetoJets:
                                jetVetoPass += 1
                                if MhtOverMet1p25:
                                    mhtOverMetPass += 1
                                    if temp_alpha_T > alphaT_Thresholds(HT):
                                        alphaTPass += 1
                                        if BDPhi0p5:
                                            biasedDeltaPhiPass += 1

totalYield = 100.
leptonVetoYield = totalYield*float(leptonVetoPass)/float(nentries)
jet1PTVetoYield = totalYield*float(jet1PTVeto)/float(nentries)
chargedHadronFractionYield = totalYield*float(chargedHadronFractionPass)/float(nentries)
htYield = totalYield*float(htPass)/float(nentries)
mhtYield = totalYield*float(mhtPass)/float(nentries)
jetVetoYield = totalYield*float(jetVetoPass)/float(nentries)
mhtOverMetYield = totalYield*float(mhtOverMetPass)/float(nentries)
alphaTYield = totalYield*float(alphaTPass)/float(nentries)
biasedDeltaPhiYield = totalYield*float(biasedDeltaPhiPass)/float(nentries)

Name = ['Total', 'leptonVetoYield', 'jet1PTVetoYield', 'chargedHadronFractionYield', 'htYield', 'mhtYield', 'jetVetoYield', 'mhtOverMetYield', 'alphaTYield', 'biasedDeltaPhiYield']
Yields = [100., leptonVetoYield, jet1PTVetoYield, chargedHadronFractionYield, htYield, mhtYield, jetVetoYield, mhtOverMetYield, alphaTYield, biasedDeltaPhiYield]

df = pd.DataFrame({
    'Name': Name,
    'Percentage': Yields,
    })

print(df)
df.to_csv('validationCutFlow.txt', sep='\t', index=False)