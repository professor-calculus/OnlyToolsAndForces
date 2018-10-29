#!/bin/bash

2DFatJetScorePlotter.py -f ROOTAnalysis.txt -x --region Signal
2DFatJetScorePlotter.py -f ROOTAnalysis.txt -x --region Signal --nHiggs2bb
2DFatJetScorePlotter.py -f ROOTAnalysis.txt -x --region 2b1mu
2DFatJetScorePlotter.py -f ROOTAnalysis.txt -x --region 0b1mu
2DFatJetScorePlotter.py -f ROOTAnalysis.txt -x --region 2mu
2DFatJetScorePlotter.py -f ROOTAnalysis.txt -x --region 0b2mu
2DFatJetScorePlotter.py -f ROOTAnalysis.txt -x --region 1b1mu
2DFatJetScorePlotter.py -f ROOTAnalysis.txt -x --region 1b2mu

2DFatJetScorePlotter.py -f ROOTAnalysis.txt -x --region Signal --HT 1500
2DFatJetScorePlotter.py -f ROOTAnalysis.txt -x --region Signal --HT 1500 --nHiggs2bb
2DFatJetScorePlotter.py -f ROOTAnalysis.txt -x --region 2b1mu --HT 1500
2DFatJetScorePlotter.py -f ROOTAnalysis.txt -x --region 0b1mu --HT 1500
2DFatJetScorePlotter.py -f ROOTAnalysis.txt -x --region 2mu --HT 1500
2DFatJetScorePlotter.py -f ROOTAnalysis.txt -x --region 0b2mu --HT 1500
2DFatJetScorePlotter.py -f ROOTAnalysis.txt -x --region 1b1mu --HT 1500
2DFatJetScorePlotter.py -f ROOTAnalysis.txt -x --region 1b2mu --HT 1500

2DFatJetMassPlotter.py -f ROOTAnalysis.txt -x --region Signal --minDiscrim 0.3
2DFatJetMassPlotter.py -f ROOTAnalysis.txt -x --region Signal --nHiggs2bb --minDiscrim 0.3
2DFatJetMassPlotter.py -f ROOTAnalysis.txt -x --region 2b1mu --minDiscrim 0.3
2DFatJetMassPlotter.py -f ROOTAnalysis.txt -x --region 0b1mu --minDiscrim 0.3
2DFatJetMassPlotter.py -f ROOTAnalysis.txt -x --region 2mu --minDiscrim 0.3
2DFatJetMassPlotter.py -f ROOTAnalysis.txt -x --region 0b2mu --minDiscrim 0.3
2DFatJetMassPlotter.py -f ROOTAnalysis.txt -x --region 1b1mu --minDiscrim 0.3
2DFatJetMassPlotter.py -f ROOTAnalysis.txt -x --region 1b2mu --minDiscrim 0.3

2DFatJetMassPlotter.py -f ROOTAnalysis.txt -x --region Signal --HT 1500 --minDiscrim 0.3
2DFatJetMassPlotter.py -f ROOTAnalysis.txt -x --region Signal --HT 1500 --nHiggs2bb --minDiscrim 0.3
2DFatJetMassPlotter.py -f ROOTAnalysis.txt -x --region 2b1mu --HT 1500 --minDiscrim 0.3
2DFatJetMassPlotter.py -f ROOTAnalysis.txt -x --region 0b1mu --HT 1500 --minDiscrim 0.3
2DFatJetMassPlotter.py -f ROOTAnalysis.txt -x --region 2mu --HT 1500 --minDiscrim 0.3
2DFatJetMassPlotter.py -f ROOTAnalysis.txt -x --region 0b2mu --HT 1500 --minDiscrim 0.3
2DFatJetMassPlotter.py -f ROOTAnalysis.txt -x --region 1b1mu --HT 1500 --minDiscrim 0.3
2DFatJetMassPlotter.py -f ROOTAnalysis.txt -x --region 1b2mu --HT 1500 --minDiscrim 0.3
