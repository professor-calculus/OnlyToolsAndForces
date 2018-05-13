#!/usr/bin/env python
import pandas as pd
import numpy as np
import argparse as a
import pydoop.hdfs as hd

parser = a.ArgumentParser(description='Dataframe txt adder')
parser.add_argument('-f', '--files', required=True, help='Path to dataframe file(s)')
args=parser.parse_args()

df_list = []
file_list = []

for path in hd.ls(args.files):
    newpath = '{0}/ROOTCuts_output/ROOTCuts_binned.txt'.format(path)
    print(newpath)
    file_list.append(newpath)

for f in file_list:
    with hd.open(f) as file:
        binned_msq = []
        binned_mlsp = []
        binned_HT_bin = []
        binned_MHT_bin = []
        binned_N_jet_bin = []
        binned_N_bjet_bin = []
        binned_yield = []
        df_temp = pd.read_csv(file, delimiter=r'\s+')
        for mhtBin in [200, 400, 600, 900]:
            for htBin in [1200]:
                for nJetBin in [6]:
                    for nBJetBin in [2,3]:
                        binned_msq.append(df_temp['M_sq'][0])
                        binned_mlsp.append(df_temp['M_lsp'][0])
                        binned_HT_bin.append(htBin)
                        binned_MHT_bin.append(mhtBin)
                        binned_N_jet_bin.append(nJetBin)
                        binned_N_bjet_bin.append(nBJetBin)
                        binned_yield.append(0.)
        df_binned = pd.DataFrame({
        'M_sq': binned_msq,
        'M_lsp': binned_mlsp,
        'HT_bin': binned_HT_bin,
        'MHT_bin': binned_MHT_bin,
        'n_Jet_bin': binned_N_jet_bin,
        'n_bJet_bin': binned_N_bjet_bin,
        'Yield': binned_yield,
        })
        df_new = df_binned.append(df_temp)
        df_new = df_new.groupby(by=['M_sq', 'M_lsp', 'HT_bin', 'MHT_bin', 'n_Jet_bin', 'n_bJet_bin']).sum()
        df_new.reset_index(inplace=True)
        #print(df_new)    
        df_list.append(df_new)

df_big = pd.concat(df_list)
df_big['M_sq'] = df_big['M_sq'].astype(int)
df_big['M_lsp'] = df_big['M_lsp'].astype(int)

print(df_big)
df_big.to_csv('ROOTCuts_combined.txt', sep='\t', index=False)

for mht in [200, 400, 600, 900]:
    for n_b in [2,3]:
        df = df_big.loc[((df_big['MHT_bin'] == mht) & (df_big['n_bJet_bin'] == n_b))]
        df = df.drop(['HT_bin', 'MHT_bin', 'n_Jet_bin', 'n_bJet_bin'], axis=1)
        name = '{0}b_MHT{1}.txt'.format(n_b, mht)
        df.to_csv(name, sep='\t', index=False, header=False)

for n_b in [2,3]:
    df = df_big.loc[(df_big['n_bJet_bin'] == n_b)]
    df = df.groupby(by=['M_sq', 'M_lsp']).sum()
    df.reset_index(inplace=True)
    df = df.drop(['HT_bin', 'MHT_bin', 'n_Jet_bin', 'n_bJet_bin'], axis=1)
    df.to_csv('{0}b_MHT_All.txt'.format(n_b), sep='\t', index=False, header=False)

df = df_big.groupby(by=['M_sq', 'M_lsp']).sum()
df.reset_index(inplace=True)
df = df.drop(['HT_bin', 'MHT_bin', 'n_Jet_bin', 'n_bJet_bin'], axis=1)
df.to_csv('MHT_All.txt', sep='\t', index=False, header=False)
