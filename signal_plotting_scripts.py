#!/bin/bash

2DFatJetScorePlotter.py -f ROOTAnalysis.txt -x --region Signal
2DFatJetScorePlotter.py -f ROOTAnalysis.txt -x --region 2b1mu
2DFatJetScorePlotter.py -f ROOTAnalysis.txt -x --region 0b1mu
2DFatJetScorePlotter.py -f ROOTAnalysis.txt -x --region 2mu
2DFatJetScorePlotter.py -f ROOTAnalysis.txt -x --region 0b2mu

2DFatJetScorePlotter.py -f ROOTAnalysis.txt -x --region Signal --HT 1500
2DFatJetScorePlotter.py -f ROOTAnalysis.txt -x --region 2b1mu --HT 1500
2DFatJetScorePlotter.py -f ROOTAnalysis.txt -x --region 0b1mu --HT 1500
2DFatJetScorePlotter.py -f ROOTAnalysis.txt -x --region 2mu --HT 1500
2DFatJetScorePlotter.py -f ROOTAnalysis.txt -x --region 0b2mu --HT 1500

2DFatJetMassPlotter.py -f ROOTAnalysis.txt -x --region Signal
2DFatJetMassPlotter.py -f ROOTAnalysis.txt -x --region 2b1mu
2DFatJetMassPlotter.py -f ROOTAnalysis.txt -x --region 0b1mu
2DFatJetMassPlotter.py -f ROOTAnalysis.txt -x --region 2mu
2DFatJetMassPlotter.py -f ROOTAnalysis.txt -x --region 0b2mu

2DFatJetMassPlotter.py -f ROOTAnalysis.txt -x --region Signal --HT 1500
2DFatJetMassPlotter.py -f ROOTAnalysis.txt -x --region 2b1mu --HT 1500
2DFatJetMassPlotter.py -f ROOTAnalysis.txt -x --region 0b1mu --HT 1500
2DFatJetMassPlotter.py -f ROOTAnalysis.txt -x --region 2mu --HT 1500
2DFatJetMassPlotter.py -f ROOTAnalysis.txt -x --region 0b2mu --HT 1500
