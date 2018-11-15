#!/bin/bash

2DFatJetScorePlotter.py -f ROOTAnalysis.txt -x --region Signal
2DFatJetScorePlotter.py -f ROOTAnalysis.txt -x --region Signal --nHiggs2bb

2DFatJetScorePlotter.py -f ROOTAnalysis.txt -x --region Signal --HT 1500
2DFatJetScorePlotter.py -f ROOTAnalysis.txt -x --region Signal --HT 1500 --nHiggs2bb

2DFatJetScorePlotter.py -f ROOTAnalysis.txt -x --region Signal --HT 3500
2DFatJetScorePlotter.py -f ROOTAnalysis.txt -x --region Signal --HT 3500 --nHiggs2bb

2DFatJetMassPlotter.py -f ROOTAnalysis.txt -x --region Signal --minDiscrim 0.3
2DFatJetMassPlotter.py -f ROOTAnalysis.txt -x --region Signal --nHiggs2bb --minDiscrim 0.3

2DFatJetMassPlotter.py -f ROOTAnalysis.txt -x --region Signal --HT 1500 --minDiscrim 0.3
2DFatJetMassPlotter.py -f ROOTAnalysis.txt -x --region Signal --HT 1500 --nHiggs2bb --minDiscrim 0.3

2DFatJetMassPlotter.py -f ROOTAnalysis.txt -x --region Signal --minDiscrim 0.6
2DFatJetMassPlotter.py -f ROOTAnalysis.txt -x --region Signal --nHiggs2bb --minDiscrim 0.6

2DFatJetMassPlotter.py -f ROOTAnalysis.txt -x --region Signal --minDiscrim 0.8
2DFatJetMassPlotter.py -f ROOTAnalysis.txt -x --region Signal --nHiggs2bb --minDiscrim 0.8

2DFatJetMassPlotter.py -f ROOTAnalysis.txt -x --region Signal --minDiscrim 0.9
2DFatJetMassPlotter.py -f ROOTAnalysis.txt -x --region Signal --nHiggs2bb --minDiscrim 0.9

2DFatJetMassPlotter.py -f ROOTAnalysis.txt -x --region Signal --HT 1500 --minDiscrim 0.6
2DFatJetMassPlotter.py -f ROOTAnalysis.txt -x --region Signal --HT 1500 --nHiggs2bb --minDiscrim 0.6

2DFatJetMassPlotter.py -f ROOTAnalysis.txt -x --region Signal --HT 1500 --minDiscrim 0.8
2DFatJetMassPlotter.py -f ROOTAnalysis.txt -x --region Signal --HT 1500 --nHiggs2bb --minDiscrim 0.8

2DFatJetMassPlotter.py -f ROOTAnalysis.txt -x --region Signal --HT 1500 --minDiscrim 0.9
2DFatJetMassPlotter.py -f ROOTAnalysis.txt -x --region Signal --HT 1500 --nHiggs2bb --minDiscrim 0.9

2DFatJetMassPlotter.py -f ROOTAnalysis.txt -x --region Signal --HT 3500 --minDiscrim 0.3
2DFatJetMassPlotter.py -f ROOTAnalysis.txt -x --region Signal --HT 3500 --nHiggs2bb --minDiscrim 0.3

2DFatJetMassPlotter.py -f ROOTAnalysis.txt -x --region Signal --HT 3500 --minDiscrim 0.6
2DFatJetMassPlotter.py -f ROOTAnalysis.txt -x --region Signal --HT 3500 --nHiggs2bb --minDiscrim 0.6

2DFatJetMassPlotter.py -f ROOTAnalysis.txt -x --region Signal --HT 3500 --minDiscrim 0.8
2DFatJetMassPlotter.py -f ROOTAnalysis.txt -x --region Signal --HT 3500 --nHiggs2bb --minDiscrim 0.8

2DFatJetMassPlotter.py -f ROOTAnalysis.txt -x --region Signal --HT 3500 --minDiscrim 0.9
2DFatJetMassPlotter.py -f ROOTAnalysis.txt -x --region Signal --HT 3500 --nHiggs2bb --minDiscrim 0.9
