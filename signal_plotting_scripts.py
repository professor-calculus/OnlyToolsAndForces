#!/bin/bash

OnlyToolsAndForces/2DFatJetScorePlotter.py -f ROOTAnalysis.txt -x --region Signal
OnlyToolsAndForces/2DFatJetScorePlotter.py -f ROOTAnalysis.txt -x --region 2b1mu
OnlyToolsAndForces/2DFatJetScorePlotter.py -f ROOTAnalysis.txt -x --region 0b1mu
OnlyToolsAndForces/2DFatJetScorePlotter.py -f ROOTAnalysis.txt -x --region 2mu
OnlyToolsAndForces/2DFatJetScorePlotter.py -f ROOTAnalysis.txt -x --region 0b2mu

OnlyToolsAndForces/2DFatJetScorePlotter.py -f ROOTAnalysis.txt -x --region Signal --HT 1500
OnlyToolsAndForces/2DFatJetScorePlotter.py -f ROOTAnalysis.txt -x --region 2b1mu --HT 1500
OnlyToolsAndForces/2DFatJetScorePlotter.py -f ROOTAnalysis.txt -x --region 0b1mu --HT 1500
OnlyToolsAndForces/2DFatJetScorePlotter.py -f ROOTAnalysis.txt -x --region 2mu --HT 1500
OnlyToolsAndForces/2DFatJetScorePlotter.py -f ROOTAnalysis.txt -x --region 0b2mu --HT 1500

OnlyToolsAndForces/2DFatJetMassPlotter.py -f ROOTAnalysis.txt -x --region Signal
OnlyToolsAndForces/2DFatJetMassPlotter.py -f ROOTAnalysis.txt -x --region 2b1mu
OnlyToolsAndForces/2DFatJetMassPlotter.py -f ROOTAnalysis.txt -x --region 0b1mu
OnlyToolsAndForces/2DFatJetMassPlotter.py -f ROOTAnalysis.txt -x --region 2mu
OnlyToolsAndForces/2DFatJetMassPlotter.py -f ROOTAnalysis.txt -x --region 0b2mu

OnlyToolsAndForces/2DFatJetMassPlotter.py -f ROOTAnalysis.txt -x --region Signal --HT 1500
OnlyToolsAndForces/2DFatJetMassPlotter.py -f ROOTAnalysis.txt -x --region 2b1mu --HT 1500
OnlyToolsAndForces/2DFatJetMassPlotter.py -f ROOTAnalysis.txt -x --region 0b1mu --HT 1500
OnlyToolsAndForces/2DFatJetMassPlotter.py -f ROOTAnalysis.txt -x --region 2mu --HT 1500
OnlyToolsAndForces/2DFatJetMassPlotter.py -f ROOTAnalysis.txt -x --region 0b2mu --HT 1500
