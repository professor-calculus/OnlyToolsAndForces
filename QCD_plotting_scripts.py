#!/bin/bash

2DFatJetScorePlotter.py -f ROOTAnalysis.txt -x --region Signal --type QCD
2DFatJetScorePlotter.py -f ROOTAnalysis.txt -x --region 2b1mu --type QCD
2DFatJetScorePlotter.py -f ROOTAnalysis.txt -x --region 0b1mu --type QCD
2DFatJetScorePlotter.py -f ROOTAnalysis.txt -x --region 2mu --type QCD
2DFatJetScorePlotter.py -f ROOTAnalysis.txt -x --region 0b2mu --type QCD

2DFatJetScorePlotter.py -f ROOTAnalysis.txt -x --region Signal --HT 1500 --type QCD
2DFatJetScorePlotter.py -f ROOTAnalysis.txt -x --region 2b1mu --HT 1500 --type QCD
2DFatJetScorePlotter.py -f ROOTAnalysis.txt -x --region 0b1mu --HT 1500 --type QCD
2DFatJetScorePlotter.py -f ROOTAnalysis.txt -x --region 2mu --HT 1500 --type QCD
2DFatJetScorePlotter.py -f ROOTAnalysis.txt -x --region 0b2mu --HT 1500 --type QCD

2DFatJetMassPlotter.py -f ROOTAnalysis.txt -x --region Signal --type QCD
2DFatJetMassPlotter.py -f ROOTAnalysis.txt -x --region 2b1mu --type QCD
2DFatJetMassPlotter.py -f ROOTAnalysis.txt -x --region 0b1mu --type QCD
2DFatJetMassPlotter.py -f ROOTAnalysis.txt -x --region 2mu --type QCD
2DFatJetMassPlotter.py -f ROOTAnalysis.txt -x --region 0b2mu --type QCD

2DFatJetMassPlotter.py -f ROOTAnalysis.txt -x --region Signal --HT 1500 --type QCD
2DFatJetMassPlotter.py -f ROOTAnalysis.txt -x --region 2b1mu --HT 1500 --type QCD
2DFatJetMassPlotter.py -f ROOTAnalysis.txt -x --region 0b1mu --HT 1500 --type QCD
2DFatJetMassPlotter.py -f ROOTAnalysis.txt -x --region 2mu --HT 1500 --type QCD
2DFatJetMassPlotter.py -f ROOTAnalysis.txt -x --region 0b2mu --HT 1500 --type QCD
