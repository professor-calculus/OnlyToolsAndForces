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
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import argparse as a
import warnings
from tqdm import tqdm
from text_histogram import histogram


parser = a.ArgumentParser(description='ROOTCuts in Python')
parser.add_argument('-d', '--delphes', nargs='*', required=True, help='Path to Delphes .root file(s)')
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
eventweight = (args.Lumi * xsec)/(nentries)

MHT_bins = np.array([200., 400., 600., 900., 999999.])
HT_bins = np.array([1200., 99999.])
n_Jet_bins = np.array([6, 99])
n_bJet_bins = np.array([2,3,99])

M_Z = 91.188


evtweight, jet_pt, jet_phi, jet_eta, jet_btag, jet_nc, jet_nn, MET,\
PID, UID, M1, PT, Eta, Phi, eEta, ePT, uEta, uPT, uPhi, pEta, pPT, pPhi = tree.arrays(["Event.Weight", "Jet.PT", "Jet.Phi", "Jet.Eta", "Jet.BTag", "Jet.NCharged", "Jet.NNeutrals", "MissingET.MET", "Particle.PID",
                                                                                            "Particle.fUniqueID", "Particle.M1", "Particle.PT", "Particle.Eta", "Particle.Phi", "Electron.Eta", "Electron.PT",
                                                                                            "Muon.Eta", "Muon.PT", "Muon.Phi", "Photon.Eta", "Photon.PT", "Photon.Phi"], outputtype=tuple)

msq = []
mlsp = []
crosssec = []
mht = []
met = []
ht = []
N_bjet = []
N_jet = []
del_R = []
biased_d_phi = []
higgs_pt = []
higgs1_pt = []
higgs2_pt = []
cut_mht = []
N_veto = []

Yield = []

binned_msq = []
binned_mlsp = []
binned_HT_bin = []
binned_MHT_bin = []
binned_N_jet_bin = []
binned_N_bjet_bin = []
binned_yield = []

for mhtBin in [200, 400, 600, 900]:
    for htBin in [1200]:
        for nJetBin in [6]:
            for nBJetBin in [2,3]:
                binned_msq.append(args.Msq)
                binned_mlsp.append(args.Mlsp)
                binned_HT_bin.append(htBin)
                binned_MHT_bin.append(mhtBin)
                binned_N_jet_bin.append(nJetBin)
                binned_N_bjet_bin.append(nBJetBin)
                binned_yield.append(0.)

CR_msq = []
CR_mlsp = []
CR_HT_bin = []
CR_MHT_bin = []
CR_N_jet_bin = []
CR_N_bjet_bin = []
CR_yield = []

for mhtBin in [200, 400, 600, 900]:
    for htBin in [1200]:
        for nJetBin in [6]:
            for nBJetBin in [2,3]:
                CR_msq.append(args.Msq)
                CR_mlsp.append(args.Mlsp)
                CR_HT_bin.append(htBin)
                CR_MHT_bin.append(mhtBin)
                CR_N_jet_bin.append(nJetBin)
                CR_N_bjet_bin.append(nBJetBin)
                CR_yield.append(0.)

eventpass = 0.

def Delta_Phi( Phi1, Phi2 ):
    if Phi1 - Phi2 > math.pi:
        delPhi = Phi1 - Phi2 - 2.*math.pi
    elif Phi1 - Phi2 < -1.*math.pi:
        delPhi = Phi1 - Phi2 + 2.*math.pi
    else:
        delPhi = Phi1 - Phi2
    return abs(delPhi);

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


for evtweighti, jet_pti, jet_phii, jet_etai, jet_btagi, jet_nci, jet_nni, METi, PIDi, UIDi, M1i, PTi, Etai, Phii, eEtai, ePTi, uEtai, uPTi, uPhii, pEtai, pPTi, pPhii in tqdm(zip(evtweight, jet_pt, jet_phi, jet_eta, jet_btag, jet_nc, jet_nn, MET, PID, UID, M1, PT, Eta, Phi, eEta, ePT, uEta, uPT, uPhi, pEta, pPT, pPhi), total=int(nentries), desc='Go Go Go!'):
    mht_x = 0.
    mht_y = 0.
    HT = 0.
    n_bjet = 0
    n_jet = 0
    nVeto = 0
    nMuon_CR = 0
    SingleMuon_CR = 0
    DoubleMuon_CR = 0
    LeadJetPT100 = False
    NoVetoObjects = False
    NJet6 = False
    MhtOverMet1p25 = False
    HT1200 = False
    MHT200 = False
    BDPhi0p5 = False
    LeadJetCHF = False

    DeltaR = []
    HiggsPT = []
    BDP = []

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

    msq.append(args.Msq)
    mlsp.append(args.Mlsp)
    crosssec.append(xsec)

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

    # Control region
    isMuon = False
    isSingleMuon = False
    isDoubleMuon = False
    for uEtaj, uPTj, uPhij in zip(uEtai, uPTi, uPhii):
        if uPTj > 30.:
            for eta, phi, pt in zip(jet_etai, jet_phii, jet_pti):
                if abs(eta) < 2.4 and pt > 40.:
                    if Delta_R(uEtaj, uPTj, eta, phi) > 0.5:
                        isMuon = True
                    else:
                        isMuon = False
            if isMuon:
                nMuon_CR += 1
                muonPT.append(uPTj)
                muonEta.append(uEtaj)
                muonPhi.append(uPhij)
            for pt, phi, eta in zip(jet_pti, jet_phii, jet_etai):
                if pt > 40. and eta < 2.4:
                    if 30. < Transverse_Mass(uPTj, pt, uPhij, phi) < 125.:
                        isSingleMuon = True
                    else:
                        isSingleMuon = False
            if isMuon and isSingleMuon:
                SingleMuon_CR += 1
                if args.verbose:
                    print('Single Muon')

    if nMuon_CR == 2:
        if abs(Invariant_Mass(muonPT[0], muonPT[1], muonEta[0], muonEta[1], muonPhi[0], muonPhi[1]) - M_Z) < 25.:
            DoubleMuon_CR += 1
            if args.verbose:
                print('Double Muon')


    
    #Reduced array of b-quarks and Higgs bosons
    for PIDj, UIDj, M1j, PTj, Etaj, Phij in zip(PIDi, UIDi, M1i, PTi, Etai, Phii):
        if abs(PIDj) == 5 or PIDj == 35:
            bPID.append(PIDj)
            bUID.append(UIDj)
            bM1.append(M1j)
            bPT.append(PTj)
            bEta.append(Etaj)
            bPhi.append(Phij)

    #Dictionary of particles so we can look up by unique ID per event
    UID_temp = 0
    for PIDj, M1j, PTj, Etaj, Phij in zip(PIDi, M1i, PTi, Etai, Phii):
        d[UID_temp] = {}
        d[UID_temp]['PID'] = PIDj
        d[UID_temp]['PT'] = PTj
        d[UID_temp]['Eta'] = Etaj
        d[UID_temp]['Phi'] = Phij
        d[UID_temp]['M1'] = M1j
        UID_temp += 1

    # MC Truth b-jet delta-R and Higgs PT.
    for PIDj, UIDj, M1j, PTj, Etaj, Phij in zip(bPID, bUID, bM1, bPT, bEta, bPhi):
        if PIDj == 35:
            HiggsPT.append(PTj)
    for PIDj, UIDj, M1j, PTj, Etaj, Phij in zip(bPID, bUID, bM1, bPT, bEta, bPhi):
        for PIDk, UIDk, M1k, PTk, Etak, Phik in zip(bPID, bUID, bM1, bPT, bEta, bPhi):
            if UIDj < UIDk and abs(PIDj) == 5 and abs(PIDk) == 5 and M1j == M1k and M1j in d and d[M1j]['PID'] == 35:
                DeltaR.append(math.sqrt(Delta_R(Etaj, Phij, Etak, Phik))
    if len(DeltaR) > 0:
        del_R.append(min(DeltaR))
    else:
        del_R.append(-1.)
    if len(HiggsPT) > 0:
        higgs_pt.append(float(sum(HiggsPT))/float(len(HiggsPT)))
    else:
        higgs_pt.append(-1.)
    if len(HiggsPT) > 0:
        higgs1_pt.append(HiggsPT[0])
        if len(HiggsPT) > 1:
            higgs2_pt.append(HiggsPT[1])

    # Number of jets, number of b-tagged jets, HT, MHT
    for PTj, BTagj, Etaj, Phij in zip(jet_pti, jet_btagi, jet_etai, jet_phii):
        if PTj > 40. and abs(Etaj) < 2.4:
            n_jet += 1
            if BTagj:
                n_bjet += 1
            mht_x += -1.*PTj*math.cos(Phij)
            mht_y += PTj*math.sin(Phij)
            HT += PTj
        if PTj > 40. and abs(Etaj) > 2.4:
            if args.verbose:
                print('jet PT: {0}, Eta: {1}'.format(PTj, Etaj))
            nVeto += 1
    mht_temp = math.sqrt(mht_x**2 + mht_y**2)
    mht.append(mht_temp)
    ht.append(HT)
    N_jet.append(n_jet)
    N_bjet.append(n_bjet)


    if mht_temp > 200.:
        MHT200 = True
    if HT > 1200.:
        HT1200 = True
    if n_jet > 5:
        NJet6 = True
    if nVeto == 0:
        NoVetoObjects = True
    N_veto.append(nVeto)

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
    biased_d_phi.append(BDP_temp)
    
    if BDP_temp > 0.5:
        BDPhi0p5 = True

    # Missing-ET
    for METj in METi:
        met.append(METj)
        met_temp = METj
    if met_temp != 0:
        if mht_temp/met_temp < 1.25:
            MhtOverMet1p25 = True

    All_Cuts = [LeadJetPT100, NoVetoObjects, NJet6, MhtOverMet1p25, HT1200, MHT200, BDPhi0p5, LeadJetCHF]
    if args.verbose:
        print(All_Cuts)
    if All_Cuts.count(False) == 0 and n_bjet in [2, 3]:
        #'M_sq', 'M_lsp', 'HT_bin', 'MHT_bin', 'n_Jet_bin', 'n_bJet_bin', 'Yield'
        binned_msq.append(args.Msq)
        binned_mlsp.append(args.Mlsp)
        if args.verbose:
            print(HT)
            print(HT_bins)
        binned_HT_bin.append(HT_bins[np.digitize([HT], HT_bins)[0] - 1])
        binned_MHT_bin.append(MHT_bins[np.digitize([mht_temp], MHT_bins)[0] - 1])
        binned_N_jet_bin.append(n_Jet_bins[np.digitize([n_jet], n_Jet_bins)[0] - 1])
        binned_N_bjet_bin.append(n_bJet_bins[np.digitize([n_bjet], n_bJet_bins)[0] - 1])
        binned_yield.append(weight)
        eventpass += 1.

    # Control Region Events
    elif (LeadJetPT100 and NJet6 and HT1200 and MHT200 and LeadJetCHF and (n_bjet in [2,3])) and ( (3.0 > mht_temp/met_temp > 1.25) or (0.2 < BDP_temp < 0.5) or (SingleMuon_CR == 1 and nMuon_CR == 1) or DoubleMuon_CR == 1):
        #'M_sq', 'M_lsp', 'HT_bin', 'MHT_bin', 'n_Jet_bin', 'n_bJet_bin', 'Yield'
        if BDP_temp < 0.5 and args.verbose:
            print('BDP')
        CR_msq.append(args.Msq)
        CR_mlsp.append(args.Mlsp)
        CR_HT_bin.append(HT_bins[np.digitize([HT], HT_bins)[0] - 1])
        CR_MHT_bin.append(MHT_bins[np.digitize([mht_temp], MHT_bins)[0] - 1])
        CR_N_jet_bin.append(n_Jet_bins[np.digitize([n_jet], n_Jet_bins)[0] - 1])
        CR_N_bjet_bin.append(n_bJet_bins[np.digitize([n_bjet], n_bJet_bins)[0] - 1])
        CR_yield.append(weight)

    if args.verbose:
        print('{0} events passed so far...'.format(eventpass))

percentpass = 100.*float(eventpass)/nentries
print('{0} of {1}, or {2} percent of events passed cuts'.format(int(eventpass), int(nentries), percentpass))

print('\n Signal Region:')
df_binned = pd.DataFrame({
    'M_sq': binned_msq,
    'M_lsp': binned_mlsp,
    'HT_bin': binned_HT_bin,
    'MHT_bin': binned_MHT_bin,
    'n_Jet_bin': binned_N_jet_bin,
    'n_bJet_bin': binned_N_bjet_bin,
    'Yield': binned_yield,
    })
print(df_binned)
df_binned = df_binned.groupby(by=['M_sq', 'M_lsp', 'HT_bin', 'MHT_bin', 'n_Jet_bin', 'n_bJet_bin']).sum()
df_binned.reset_index(inplace=True)
print(df_binned)

#Control Region
df_CR = pd.DataFrame({
    'M_sq': CR_msq,
    'M_lsp': CR_mlsp,
    'HT_bin': CR_HT_bin,
    'MHT_bin': CR_MHT_bin,
    'n_Jet_bin': CR_N_jet_bin,
    'n_bJet_bin': CR_N_bjet_bin,
    'Yield': CR_yield,
    })
df_CR = df_CR.groupby(by=['M_sq', 'M_lsp', 'HT_bin', 'MHT_bin', 'n_Jet_bin', 'n_bJet_bin']).sum()
df_CR.reset_index(inplace=True)
print('\n Control Region:')
print(df_CR)

if not args.NoOutput:
    df_binned.to_csv(os.path.join(directory, 'ROOTCuts_binned.txt'), sep='\t', index=False)
    df_CR.to_csv(os.path.join(directory, 'ROOTCuts_CR.txt'), sep='\t', index=False)

print(N_veto)

#columns = ['M_sq', 'M_lsp', 'crosssec', 'MET', 'MHT', 'HT', 'Higgs_PT', 'bJetsDelR', 'bDPhi']
df = pd.DataFrame({
    'M_sq': msq,
    'M_lsp': mlsp,
    'crosssec': crosssec,
    'MET': met,
    'MHT': mht,
    'HT': ht,
    'Higgs_PT': higgs_pt,
    'Higgs1_PT': higgs1_pt,
    'Higgs2_PT': higgs2_pt,
    'bJetsDelR': del_R,
    'bDPhi': biased_d_phi,
    'NJet': N_jet,
    'NBJet': N_bjet,
    'NVeto': N_veto,
    })

print(df)
if not args.NoOutput:
    df.to_csv(os.path.join(directory, 'ROOTCuts.txt'), sep='\t', index=False)

plottables = ['MET', 'MHT', 'HT', 'Higgs_PT', 'bJetsDelR', 'bDPhi', 'NJet', 'NBJet']
bins_HT = np.linspace(0.,8000.,160)
bins_MHT = np.linspace(0.,2000.,200)
bins_DelR = np.linspace(0.,5.,100)
bins_BMass = np.linspace(0.,500.,100)
bins_njet = np.arange(0, 20, 1)
bins_nbjet = np.arange(0, 14, 1)
bins_BDP = np.linspace(0.,3.,60)


dict = {'MET': {'bins': bins_MHT, 'title': 'Missing $E_{T}$ / GeV'},
        'MHT': {'bins': bins_MHT, 'title': 'Missing $H_{T}$ / GeV'},
        'HT': {'bins': bins_HT, 'title': 'Total $H_{T}$ / GeV'},
        'bJetsDelR': {'bins': bins_DelR, 'title': 'b-Jets $\Delta R$'},
        'Higgs_PT': {'bins': bins_MHT, 'title': 'Higgs $p_{T}$'},
        'bDPhi': {'bins': bins_BDP, 'title': '$\Delta\Phi^{*}$'},
        'NJet': {'bins': bins_njet, 'title': 'Number of Jets'},
        'NBJet': {'bins': bins_nbjet, 'title': 'Number of Bottom Quark Jets'},
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
