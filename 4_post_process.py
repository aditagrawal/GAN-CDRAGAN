from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import pandas as pd
import logging
import argparse
import os
import sys
import json
from sklearn.preprocessing import QuantileTransformer, StandardScaler
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("--settings_file", help="JSON settings file")
parser.add_argument("--version", help="GAN configuration version")

args = parser.parse_args()
settings_file = args.settings_file
version = args.version

with open(settings_file, "r") as jfile:
    settings = json.load(jfile)

sys.path.append(settings['path_to_library'])
from utils import *

header_file = settings['header_file']
gen_scaled_train_file = settings['gen_scaled_train_file']
gen_train_file = settings['gen_train_file']

json_file_special = settings['special_pars']
json_file_boxcox = settings['boxcox_pars']
json_file_scale = settings['scaling_pars']
json_file_onehot = settings['onehot_pars']
pickle_file_quantile = settings['quantile_pars']

if version != '':
    gen_scaled_train_file = gen_scaled_train_file.format(version)
    gen_train_file = gen_train_file.format(version)

header = pd.read_csv(header_file)
types = header.iloc[0].tolist()
colNumerical = [header.columns[i] for i,x in enumerate(types) if x in ['N']]
colTarget = [header.columns[i] for i,x in enumerate(types) if x in ['L']][0]

# read generated data
generated = pd.read_csv(gen_scaled_train_file)
generated[colTarget] = generated[colTarget] + 0.5
generated[colTarget] = generated[colTarget].astype(int)

# unscale the features
scl = PandasStandardScaler(json_file=json_file_scale)
generated = scl.back_transform(generated)

# inverse BoxCox on the features
bcx = PandasBoxCox(json_file=json_file_boxcox)
generated = bcx.back_transform(generated)

# inverse special treatment
mtr = PandasSpecialTreatment(json_file=json_file_special)
generated = mtr.back_transform(generated)

# inverse one hot encoding
onehot = PandasOneHotEncoder(json_file=json_file_onehot)
generated = onehot.back_transform(generated)

generated['unq1'] = [200000000 + x for x in range(len(generated))]
cols = [x for x in header.columns if x in generated.columns]
generated = generated[cols]
generated.to_csv(gen_train_file, index=False, header=True)

