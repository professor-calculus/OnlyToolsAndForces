#!/usr/bin/env python

import pandas as pd
import uproot
import argparse as a

parser = a.ArgumentParser(description='Thing to get mu values from HiggsCombine output')
parser.add_argument('-f', '--files', nargs='*', required=True, help='Path to HiggsCombine output root files')
parser.add_argument('-p', '--prefix', required=True, help='Output file prefix')
args = parser.parse_args()

Msq = []
Mlsp = []
mu_obs = []
mu_exp = []

for file in args.files:

    temp_Msq = float(file.split('_')[-2][:-2])
    R = file.split('_')[-1][:-5]
    Msq.append(temp_Msq)
    if R == 'R0p99':
        Mlsp.append(1.16)
    elif R == 'R0p555':
        Mlsp.append(100.)
    elif R == 'R0p384':
        Mlsp.append(200.)


    f = uproot.open(file)['limit']
    for x in f.iterate(['limit'], outputtype=tuple):
        for y in x:
            mu_exp.append(y[2])
            mu_obs.append(y[5])

df_exp = pd.DataFrame({
    'Msq': Msq,
    'Mlsp': Mlsp,
    'mu': mu_exp,
    })

df_obs = pd.DataFrame({
    'Msq': Msq,
    'Mlsp': Mlsp,
    'mu': mu_obs,
    })

print('Observed mu values:')
print(df_obs)

print('Expected mu values:')
print(df_exp)

if args.prefix:
    prefix = args.prefix + '_mu'
else:
    prefix = 'mu'

df_obs.to_csv(prefix + '_observed.txt', header=False, index=False, sep='\t', columns=['Msq', 'Mlsp', 'mu'])
df_exp.to_csv(prefix + '_expected.txt', header=False, index=False, sep='\t', columns=['Msq', 'Mlsp', 'mu'])
