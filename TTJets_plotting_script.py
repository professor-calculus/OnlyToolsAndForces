#!/bin/bash

2DFatJetScorePlotter.py -f ROOTAnalysis_output_*/ROOTAnalysis.txt -x --region Signal --type TTJets
2DFatJetScorePlotter.py -f ROOTAnalysis_output_*/ROOTAnalysis.txt -x --region 2b1mu --type TTJets
2DFatJetScorePlotter.py -f ROOTAnalysis_output_*/ROOTAnalysis.txt -x --region 0b1mu --type TTJets
2DFatJetScorePlotter.py -f ROOTAnalysis_output_*/ROOTAnalysis.txt -x --region 2mu --type TTJets
2DFatJetScorePlotter.py -f ROOTAnalysis_output_*/ROOTAnalysis.txt -x --region 0b2mu --type TTJets

2DFatJetScorePlotter.py -f ROOTAnalysis_output_*/ROOTAnalysis.txt -x --region Signal --HT 1500 --type TTJets
2DFatJetScorePlotter.py -f ROOTAnalysis_output_*/ROOTAnalysis.txt -x --region 2b1mu --HT 1500 --type TTJets
2DFatJetScorePlotter.py -f ROOTAnalysis_output_*/ROOTAnalysis.txt -x --region 0b1mu --HT 1500 --type TTJets
2DFatJetScorePlotter.py -f ROOTAnalysis_output_*/ROOTAnalysis.txt -x --region 2mu --HT 1500 --type TTJets
2DFatJetScorePlotter.py -f ROOTAnalysis_output_*/ROOTAnalysis.txt -x --region 0b2mu --HT 1500 --type TTJets

2DFatJetMassPlotter.py -f ROOTAnalysis_output_*/ROOTAnalysis.txt -x --region Signal --type TTJets
2DFatJetMassPlotter.py -f ROOTAnalysis_output_*/ROOTAnalysis.txt -x --region 2b1mu --type TTJets
2DFatJetMassPlotter.py -f ROOTAnalysis_output_*/ROOTAnalysis.txt -x --region 0b1mu --type TTJets
2DFatJetMassPlotter.py -f ROOTAnalysis_output_*/ROOTAnalysis.txt -x --region 2mu --type TTJets
2DFatJetMassPlotter.py -f ROOTAnalysis_output_*/ROOTAnalysis.txt -x --region 0b2mu --type TTJets

2DFatJetMassPlotter.py -f ROOTAnalysis_output_*/ROOTAnalysis.txt -x --region Signal --HT 1500 --type TTJets
2DFatJetMassPlotter.py -f ROOTAnalysis_output_*/ROOTAnalysis.txt -x --region 2b1mu --HT 1500 --type TTJets
2DFatJetMassPlotter.py -f ROOTAnalysis_output_*/ROOTAnalysis.txt -x --region 0b1mu --HT 1500 --type TTJets
2DFatJetMassPlotter.py -f ROOTAnalysis_output_*/ROOTAnalysis.txt -x --region 2mu --HT 1500 --type TTJets
2DFatJetMassPlotter.py -f ROOTAnalysis_output_*/ROOTAnalysis.txt -x --region 0b2mu --HT 1500 --type TTJets
