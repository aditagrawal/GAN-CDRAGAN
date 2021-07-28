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
import h5py
from sklearn.preprocessing import QuantileTransformer, StandardScaler
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("--settings_file", help="JSON settings file")
parser.add_argument("--fit_boxcox", help="whether to fit Box-Cox transformation or use existed JSON file")

args = parser.parse_args()
settings_file = args.settings_file
fit_boxcox = int(args.fit_boxcox)

with open(settings_file, "r") as jfile:
    settings = json.load(jfile)

sys.path.append(settings['path_to_library'])
from utils import *

train_file = settings['train_file']
header_file = settings['header_file']
scaled_train_file = settings['scaled_train_file']
scaled_hdf_file = settings['scaled_hdf_file']
scaled_header_file = settings['scaled_header_file']
missing_train_file = settings['missing_train_file']
highfreq_train_file = settings['highfreq_train_file']
viz_folder = settings['viz_folder']

json_file_missing = settings['missing_pars']
json_file_special = settings['special_pars']
pickle_file_boxcox = settings['pickle_boxcox_pars']
json_file_boxcox = settings['boxcox_pars']
json_file_scale = settings['scaling_pars']
json_file_onehot = settings['onehot_pars']
pickle_file_quantile = settings['quantile_pars']
xgboost_model = settings['xgboost_model']

output_folder = os.path.dirname(scaled_train_file)
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
if not os.path.exists(viz_folder):
    os.makedirs(viz_folder)

# read header and define feature list
header = pd.read_csv(header_file)
types = header.iloc[0].tolist()
colNumerical = [header.columns[i] for i,x in enumerate(types) if x in ['N']]
colTarget = [header.columns[i] for i,x in enumerate(types) if x in ['L']]
colCategorical = [header.columns[i] for i,x in enumerate(types) if x in ['C']]
colKey = [header.columns[i] for i,x in enumerate(types) if x in ['K']][0]

dtypes = dict(zip([x for x in range(len(types))], [np.float if x in ['N','L'] else str for x in types]))

# read data
train = pd.read_csv(train_file, dtype=dtypes)
print("Data loaded")

for col in colTarget:
    train[col] = train[col] - 0.5

# one hot encode categorical features
onehot = PandasOneHotEncoder(json_file=json_file_onehot)
train,colOnehot = onehot.fit_transform(train, colCategorical)

# treat missing values and special values
mtr = PandasSpecialTreatment(json_file=json_file_special)
train,colMissing, colHighFreq = mtr.fit_transform(train, colNumerical)

train = xgboost_capping(train, colNumerical, xgboost_model, header_file)

# Box-Cox the features
bcx = PandasBoxCox(json_file=json_file_boxcox, viz_folder=viz_folder)
if fit_boxcox == 1:
    train = bcx.fit_transform(train, colNumerical, visualize=1)
else:
    train = bcx.transform(train, colNumerical)

# scale the features
scl = PandasStandardScaler(json_file=json_file_scale)
train = scl.fit_transform(train, colNumerical)

# replace missing values
if True:
    train = replace_missing_values(train, colNumerical, method='median')

if False:
    xgboost_folder = os.path.join(os.path.dirname(scaled_train_file), 'xgboost')
    train = replace_missing_values(train, colNumerical, method='xgboost', feats=colNumerical+colOnehot+colTarget, output_folder = xgboost_folder)

# create new header
header = pd.DataFrame()
for col in train.columns:
    header[col] = ['']
    if col in colOnehot:
        header.loc[0, col] = 'G'
    if col in colNumerical:
        header.loc[0, col] = 'N'
    if col in colMissing:
        header.loc[0, col] = 'G'
    if col in colHighFreq:
        header.loc[0, col] = 'G'
    if col in colTarget:
        header.loc[0, col] = 'G'
header.to_csv(scaled_header_file, index=False)

# save data to CSV
train = train.sample(frac=1)
train.to_csv(scaled_train_file, index=False)

'''
# save data to HDF
colsN = [header.columns[i] for i,x in enumerate(header.loc[0]) if x in ['N']]
colsG = [header.columns[i] for i,x in enumerate(header.loc[0]) if x in ['G']]
dt = h5py.special_dtype(vlen=str)
with h5py.File(scaled_hdf_file, 'w') as H:
    H.create_dataset( 'trainN', data=train[colsN].values, dtype='f')
    H['trainN'].attrs['columns'] = [np.string_(x) for x in colsN]
    H.create_dataset( 'trainG', data=train[colsG].values, dtype='f')
    H['trainG'].attrs['columns'] = [np.string_(x) for x in colsG]
#train.to_hdf(scaled_hdf_file, 'data', mode='w', format='fixed', data_columns=True)
'''
