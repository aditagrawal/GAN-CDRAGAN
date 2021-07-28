from __future__ import division
from __future__ import print_function

import pandas as pd
import numpy as np
from numpy.linalg import norm
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import json
from collections import OrderedDict
from scipy.stats import boxcox, entropy
from scipy.special import inv_boxcox

from math import fabs as fabs
import operator
import os, sys
import re
import xgboost as xgb

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.backends.backend_pdf
    import matplotlib.cm as cm
    import matplotlib.gridspec as gridspec
    canplot = True
except ImportError:
    canplot = False

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
try:
    from tsne import bh_sne
    cantsne = True
except ImportError:
    cantsne = False

def tsne_fit_transform(data, perplexity=50.0, nsvd=30):
    if nsvd > 0:
        svd = TruncatedSVD(n_components=nsvd)
        data = svd.fit_transform(data)
    data = StandardScaler().fit_transform(data)
    data = bh_sne(data, perplexity=perplexity)
    return data

class PandasOneHotEncoder():
    """
    Function to convert categorical columns to one hot encoded binary columns
    Author - Dmitry Efimov (dmitry.efimov@aexp.com)

    Mandatory input variables:
        data - Pandas data frame
        cols - list of columns that should be converted

    Optional input variables:
        prefix - prefix for the encoded columns
        drop - if True, then drop original columns

    Output -
        Pandas data frame with encoded columns

    """
    def __init__(self, json_file, remove_original=True, drop_first=False):
        self.remove_original = remove_original
        self.drop_first = drop_first
        self.colnames = []
        self.json_file = json_file

    def fit_transform(self, data, cols):
        if os.path.exists(self.json_file):
            self.load_json()
        else:
            self.stats = OrderedDict()
        colsIndicator = []
        for col in cols:
            data.loc[data[col].isnull(), col] = 'null'
            self.stats[col] = OrderedDict()
            self.stats[col]['values'] = data[col].unique().tolist()
            if len([x for x in self.stats[col]['values'] if not x in ['0','1']]) == 0:
                self.stats[col]['type'] = 'indicator'
                colsIndicator.append(col)
                data[col] = data[col].astype(float)
                data[col] = data[col] - 0.5
            else:
                self.stats[col]['type'] = 'categorical'
        self.save_json()
        cols = [x for x in cols if not x in colsIndicator]
        data_onehot = pd.get_dummies(data[cols], dummy_na=False, drop_first=self.drop_first, columns=cols)
        data_onehot = data_onehot - 0.5
        data = pd.concat([data, data_onehot], axis=1)
        if self.remove_original:
            data.drop(cols, axis=1, inplace=True)
        colsOnehot = list(data_onehot.columns)
        colsOnehot.extend(colsIndicator)
        print("One hot encoding done")
        return data, colsOnehot

    def transform(self, data, cols):
        if os.path.exists(self.json_file):
            self.load_json()
        else:
            print("Error: run fit_transform for one-hot encoding first")
            sys.exit(-1)
        colsCategorical = []
        colsOnehot = []
        colsIndicator = []
        for col in self.stats.keys():
            data.loc[data[col].isnull(), col] = 'null'
            if self.stats[col]['type'] == 'indicator':
                data[col] = data[col].astype(float)
                data[col] = data[col] - 0.5
                colsIndicator.append(col)
            else:
                values = self.stats[col]['values']
                colsOnehot.extend([col + '_' + str(x) for x in values])
                colsCategorical.append(col)
        data_onehot = pd.get_dummies(data[colsCategorical], dummy_na=False, drop_first=self.drop_first, columns=colsCategorical)
        data_onehot = data_onehot - 0.5
        for col in [x for x in colsOnehot if not x in data_onehot.columns]:
            data_onehot[col] = -0.5
        colsDrop = [x for x in data_onehot.columns if not x in colsOnehot]
        if len(colsDrop) > 0:
            print("Dropped one-hot encoded columns: " + ','.join(colsDrop))
            data_onehot.drop(colsDrop, axis=1, inplace=True)
        data = pd.concat([data, data_onehot], axis=1)
        if self.remove_original:
            data.drop(colsCategorical, axis=1, inplace=True)
        colsOnehot.extend(colsIndicator)
        print("One hot encoding done")
        return data, colsOnehot

    def back_transform(self, data, cols=[]):
        self.load_json()
        if len(cols) == 0:
            cols = [x for x in self.stats.keys()]
        for col in cols:
            if self.stats[col]['type'] == 'indicator':
                continue
            values = self.stats[col]['values']
            onehot_cols = [col + '_' + str(x) for x in values]
            values, onehot_cols = map(list, zip(*[(values[i], x) for i,x in enumerate(onehot_cols) if x in data.columns]))
            data[col] = [values[i] for i in data[onehot_cols].values.argmax(axis=1)]
            data.loc[data[col] == 'null', col] = ''
            data.drop(onehot_cols, axis=1, inplace=True)
            print("Categorical features restored from one hot encodings")
        return data

    def load_json(self):
        with open(self.json_file, "r") as jfile:
            self.stats = json.load(jfile, object_pairs_hook=OrderedDict)

    def save_json(self):
        with open(self.json_file, "w") as jfile:
            json.dump(self.stats, jfile, separators=(',', ':'), indent=4)

class LaplaceSmoothing():
    """
    Class to convert categorical columns to numerical using Laplace smoothing
    Author - Dmitry Efimov (dmitry.efimov@aexp.com)

    Mandatory input variables:
        data - Pandas data frame
        target - target column name
        cols - list of columns that should be converted

    Optional input variables:
        stats - Pandas data frame that contains statistics: this parameter is necessary if you convert test data
                (contains 4 columns: colname, colvalue, sum, count)
        replace - if True, the original features will be replaced by likelihoods

    Output -
        data frame with converted columns
        data frame with statistics (optional)

    """
    def __init__(self, json_file, replace=True):
        self.replace = replace
        self.stats = OrderedDict()
        self.json_file = json_file

    def fit_transform(self, data, target, cols):
        # initialize stats dictionary
        self.stats = OrderedDict()
        # calculate global average
        self.global_avg = data[target].mean()
        # calculate average by columns
        for col in cols:
            likeli = data.groupby(col).agg({target: [np.sum, len]}).reset_index(drop=False)
            likeli.columns = [col, 'sum', 'count']
            likeli['smoothed_mean'] = (likeli['sum'] + 30.0*self.global_avg)/(likeli['count']+30.0)
            self.stats[col] = likeli
            data = pd.merge(data, likeli[[col, 'smoothed_mean']], on=col, how='left')
            data['smoothed_mean'] = data['smoothed_mean'].fillna(self.global_avg)
            if self.replace:
                data.drop(col, axis=1, inplace=True)
                data.rename(columns={'smoothed_mean': col}, inplace=True)
            else:
                data.rename(columns={'smoothed_mean': col+'_likeli'}, inplace=True)
        self.save_json()
        return data

    def fit(self, data, target, cols):
        # initialize stats dictionary
        self.stats = OrderedDict()
        # calculate global average
        self.global_avg = data[target].mean()
        # calculate average by columns
        for col in cols:
            likeli = data.groupby(col).agg({target: [np.sum, len]}).reset_index(drop=False)
            likeli.columns = [col, 'sum', 'count']
            likeli['smoothed_mean'] = (likeli['sum'] + 30.0*self.global_avg)/(likeli['count']+30.0)
            self.stats[col] = likeli
        self.save_json()

    def transform(self, data, target, cols):
        self.load_json()
        for col in cols:
            if not col in self.stats:
                continue
            likeli = self.stats[col]
            data = pd.merge(data, likeli[[col, 'smoothed_mean']], on=col, how='left')
            data['smoothed_mean'] = data['smoothed_mean'].fillna(self.global_avg)
            if self.replace:
                data.drop(col, axis=1, inplace=True)
                data.rename(columns={'smoothed_mean': col}, inplace=True)
            else:
                data.rename(columns={'smoothed_mean': col+'_likeli'}, inplace=True)
        return data

    def load_json(self):
        with open(self.json_file, "r") as jfile:
            param = json.load(jfile, object_pairs_hook=OrderedDict)
        if 'global_avg' in param.keys():
            self.global_avg = param['global_avg']
        self.stats = {}
        for col, likeli in param.items():
            if col == 'global_avg':
                continue
            self.stats[col] = pd.DataFrame.from_dict(likeli, orient='columns')

    def save_json(self):
        json_dict = {}
        json_dict['global_avg'] = self.global_avg
        for col, likeli in self.stats.items():
            json_dict[col] = self.stats[col].to_dict(orient='list', into=OrderedDict)
        with open(self.json_file, "w") as jfile:
            json.dump(json_dict, jfile, indent=4)

def replace_missing_values(data, cols, method='median', feats=[], output_folder=''):
    datasize = len(data)
    if method == 'xgboost':
        import xgboost as xgb
        param = {}
        param["booster"] = "gbtree"
        param["objective"] = "reg:linear"
        param["eta"] = 0.01
        param["max_depth"] = 5
        param["min_child_weight"] = 30
        param["silent"] = 0
        param["nthread"] = 20
        param["seed"] = 23023
        param["missing"] = -9999.0
        num_round = 200
        data[cols] = data[cols].fillna(-9999.0)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

    for col in cols:
        if method == 'median':
            col_median = data[col].median()
            data[col].fillna(col_median, inplace = True)
        if method == 'zero':
            data[col].fillna(0.0, inplace = True)
        if method == 'random':
            sample = data.loc[~data[col].isnull(), col].values
            sample_size = datasize - len(sample)
            data.loc[data[col].isnull(), col] = np.random.choice(sample, size=sample_size)
        if method == 'xgboost':
            colFeats = [x for x in feats if not x in [col]]
            if len(data.loc[data[col]==-9999.0]) == 0: continue
            if not os.path.exists(os.path.join(output_folder, 'xgboost_'+col+'.model')):
                dataD = xgb.DMatrix(data.loc[data[col]!=-9999.0, colFeats].values, label=data.loc[data[col]!=-9999.0, col].values)
                bst = xgb.train(param, dataD, num_round)
                bst.save_model(os.path.join(output_folder, 'xgboost_'+col+'.model'))

    if method == 'xgboost':
        for col in cols:
            if os.path.exists(os.path.join(output_folder, 'xgboost_'+col+'.model')):
                bst = xgb.Booster({'nthread':10})
                bst.load_model(os.path.join(output_folder, 'xgboost_'+col+'.model'))
                colFeats = [x for x in feats if not x in [col]]
                nmissing = len(data.loc[data[col] == -9999.0])
                if nmissing > 0:
                    print(col)
                    dataD = xgb.DMatrix(data.loc[data[col]==-9999.0, colFeats].values)
                    preds = bst.predict(dataD)
                    print(preds[0:10])
                    data.loc[data[col] == -9999.0, col] = preds

    return data

class PandasMissingTreatment():
    def __init__(self, json_file):
        self.stats = OrderedDict()
        self.json_file = json_file

    def fit_transform(self, data, cols):
        colMissing = []
        datasize = len(data)
        for col in cols:
            nmissing = len(data.loc[data[col].isnull()])
            self.stats[col] = OrderedDict()
            self.stats[col]['nmissing'] = 1.0*nmissing/datasize
            # check how many missing values: 
            # if it is more than 0.2% => add missing column
            if 1.0*nmissing/datasize > 0.002:
                data[col+'_missing'] = 0.5
                data.loc[data[col].isnull(), col+'_missing'] = -0.5
                colMissing.append(col + '_missing')
                self.stats[col]['missing_column'] = col + '_missing'
            # save median to JSON file just in case
            self.stats[col]['median'] = data[col].median()
        self.save_json()
        print("Treatment of missing values done")
        return data, colMissing

    def transform(self, data, cols):
        if os.path.exists(self.json_file):
            self.load_json()
        else:
            print("Error: run fit_transform for treatment of missing values first")
            sys.exit(-1)
        datasize = len(data)
        colMissing = []
        if len(cols) == 0:
            cols = [x for x in self.stats.keys()]
        for col in cols:
            if not col in self.stats.keys():
                continue
            if 'missing_column' in self.stats[col].keys():
                col_missing = self.stats[col]['missing_column']
                data[col_missing] = 0.5
                data.loc[data[col].isnull(), col_missing] = -0.5
                colMissing.append(col_missing)
        print("Treatment of missing values done")
        return data, colMissing

    def back_transform(self, data, cols=[]):
        self.load_json()
        datasize = len(data)
        if len(cols) == 0:
            cols = [x for x in self.stats.keys()]
        for col in cols:
            nmissing = self.stats[col]['nmissing']
            if 'missing_column' in self.stats[col].keys():
                col_missing = self.stats[col]['missing_column']
                if not col_missing in data.columns:
                    continue
                k = int(np.round(nmissing*datasize))
                data.loc[np.argsort(data[col_missing].values)[0:k], col] = np.nan
        return data

    def load_json(self):
        with open(self.json_file, "r") as jfile:
            self.stats = json.load(jfile, object_pairs_hook=OrderedDict)

    def save_json(self):
        with open(self.json_file, "w") as jfile:
            json.dump(self.stats, jfile, indent=4)

class PandasSpecialTreatment():
    def __init__(self, json_file):
        self.stats = OrderedDict()
        self.json_file = json_file

    def fit_transform(self, data, cols):
        colMissing = []
        colHighFreq = []
        datasize = len(data)
        for col in cols:
            nmissing = len(data.loc[data[col].isnull()])
            data_counts = data[col].value_counts()
            nhighfreq = data_counts.max()
            highfreq = data_counts.idxmax()
            self.stats[col] = OrderedDict()
            self.stats[col]['nmissing'] = 1.0*nmissing/datasize
            self.stats[col]['nhighfreq'] = 1.0*nhighfreq/(datasize-nmissing)
            self.stats[col]['highfreq'] = highfreq
            # check how many missing values: 
            # if it is more than 0.2% => add missing column
            if 1.0*nmissing/datasize > 0.002:
                data[col+'_missing'] = 0.5
                data.loc[data[col].isnull(), col+'_missing'] = -0.5
                colMissing.append(col + '_missing')
                self.stats[col]['missing_column'] = col + '_missing'
            # check how many high frequent values: 
            # if it is more than 50% => add high frequency column + replace high frequent value by missing and add column
            if 1.0*nhighfreq/(datasize-nmissing) > 0.5 and len(data_counts)>30:
                data[col+'_highfreq'] = 0.5
                data.loc[data[col] == highfreq, col+'_highfreq'] = -0.5
                data.loc[data[col] == highfreq, col] = np.nan
                colHighFreq.append(col + '_highfreq')
                self.stats[col]['highfreq_column'] = col + '_highfreq'
            # save median to JSON file just in case
            self.stats[col]['median'] = data[col].median()
        self.save_json()
        print("Special treatment done")
        return data, colMissing, colHighFreq

    def transform(self, data, cols):
        self.load_json()
        datasize = len(data)
        colMissing = []
        colHighFreq = []
        if len(cols) == 0:
            cols = [x for x in self.stats.keys()]
        for col in cols:
            if not col in self.stats.keys():
                continue
            if 'missing_column' in self.stats[col].keys():
                col_missing = self.stats[col]['missing_column']
                data[col_missing] = 0.5
                data.loc[data[col].isnull(), col_missing] = -0.5
                colMissing.append(col_missing)
            if 'highfreq_column' in self.stats[col].keys():
                col_highfreq = self.stats[col]['highfreq_column']
                highfreq = self.stats[col]['highfreq']
                data[col_highfreq] = 0.5
                data.loc[data[col] == highfreq, col_highfreq] = -0.5
                data.loc[data[col] == highfreq, col] = np.nan
                colHighFreq.append(col_highfreq)
        print("Special treatment done")
        return data, colMissing, colHighFreq

    def back_transform(self, data, cols=[]):
        self.load_json()
        datasize = len(data)
        if len(cols) == 0:
            cols = [x for x in self.stats.keys()]
        for col in cols:
            nmissing = self.stats[col]['nmissing']
            if 'highfreq_column' in self.stats[col].keys():
                col_highfreq = self.stats[col]['highfreq_column']
                if not col_highfreq in data.columns:
                    continue
                nhighfreq = self.stats[col]['nhighfreq']
                highfreq = self.stats[col]['highfreq']
                k = int(np.round(nhighfreq*(datasize-nmissing)))
                data.loc[np.argsort(data[col_highfreq].values)[0:k], col] = highfreq
            if 'missing_column' in self.stats[col].keys():
                col_missing = self.stats[col]['missing_column']
                if not col_missing in data.columns:
                    continue
                k = int(np.round(nmissing*datasize))
                data.loc[np.argsort(data[col_missing].values)[0:k], col] = np.nan
        return data

    def load_json(self):
        with open(self.json_file, "r") as jfile:
            self.stats = json.load(jfile, object_pairs_hook=OrderedDict)

    def save_json(self):
        with open(self.json_file, "w") as jfile:
            json.dump(self.stats, jfile, indent=4)

class XGBoostTreatment():
    """
    Class to scale numerical columns
    Author - Dmitry Efimov (dmitry.efimov@aexp.com)

    Mandatory input variables:
        data - Pandas data frame
        target - target column name
        cols - list of columns that should be converted

    Optional input variables:
        stats - Pandas data frame that contains statistics: this parameter is necessary if you convert test data
                (contains 4 columns: colname, colvalue, sum, count)
        replace - if True, the original features will be replaced by likelihoods

    Output -
        data frame with converted columns
        data frame with statistics (optional)

    """
    def __init__(self, json_file):
        self.stats = OrderedDict()
        self.json_file = json_file

    def fit_transform(self, data, xgboost_model, xgboost_header):
        if os.path.exists(self.json_file):
            self.load_json()
        else:
            # initialize stats dictionary
            self.stats = OrderedDict()
        bst = xgb.Booster({'nthread':8})
        bst.load_model(xgboost_model)
        bst_model = '\n'.join(bst.get_dump(with_stats=True))
        splits = re.findall('\[f([0-9]+)<([-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)\]', bst_model)
        header = pd.read_csv(xgboost_header)
        types = header.loc[0].values
        flist = [header.columns[i] for i,x in enumerate(types) if x in ['N','C','G']]
        for i in range(len(flist)):
            splits_subset = [float(x[1]) for x in splits if x[0] == str(i)]
            col = flist[i]
            if len(splits_subset) > 0 and col in data.columns:
                col_max = np.max(splits_subset)
                col_min = np.min(splits_subset)
                self.stats[col] = {}
                self.stats[col]['min_split'] = col_min
                self.stats[col]['max_split'] = col_max
                if col_max > col_min:
                    data[col] = (2.0*data[col] - (col_max + col_min))/(col_max - col_min)
                    data.loc[data[col] > 1, col] = 2.0
                    data.loc[data[col] < -1, col] = -2.0
                else:
                    data[col] = data[col] - col_min
                    data.loc[data[col] > 0, col] = 1.0
                    data.loc[data[col] < 0, col] = -1.0
        self.save_json()
        print("XGBoost treatment done")
        return data

    def fit(self, data, xgboost_model, xgboost_header):
        if os.path.exists(self.json_file):
            self.load_json()
        else:
            # initialize stats dictionary
            self.stats = OrderedDict()
        bst = xgb.Booster({'nthread':8})
        bst.load_model(xgboost_model)
        bst_model = '\n'.join(bst.get_dump(with_stats=True))
        splits = re.findall('\[f([0-9]+)<([-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)\]', bst_model)
        header = pd.read_csv(xgboost_header)
        types = header.loc[0].values
        flist = [header.columns[i] for i,x in enumerate(types) if x in ['N','C','G']]
        for i in range(len(flist)):
            splits_subset = [float(x[1]) for x in splits if x[0] == str(i)]
            col = flist[i]
            if len(splits_subset) > 0 and col in data.columns:
                col_max = np.max(splits_subset)
                col_min = np.min(splits_subset)
                self.stats[col] = {}
                self.stats[col]['min_split'] = col_min
                self.stats[col]['max_split'] = col_max
        self.save_json()
        print("XGBoost treatment done")

    def transform(self, data, cols=[]):
        self.load_json()
        if len(cols) == 0:
            cols = [x for x in self.stats.keys()]
        for col in cols:
            if col in self.stats.keys() and col in data.columns:
                col_min = self.stats[col]['min_split']
                col_max = self.stats[col]['max_split']
                if col_max > col_min:
                    data[col] = (2.0*data[col] - (col_max + col_min))/(col_max - col_min)
                    data.loc[data[col] > 1, col] = 2.0
                    data.loc[data[col] < -1, col] = -2.0
                else:
                    data[col] = data[col] - col_min
                    data.loc[data[col] > 0, col] = 1.0
                    data.loc[data[col] < 0, col] = -1.0
        print("XGBoost treatment done")
        return data

    def back_transform(self, data, cols=[]):
        self.load_json()
        if len(cols) == 0:
            cols = [x for x in self.stats.keys()]
        for col in cols:
            if col in self.stats.keys() and col in data.columns:
                col_min = self.stats[col]['min_split']
                col_max = self.stats[col]['max_split']
                if col_max > col_min:
                    data[col] = 0.5*(data[col]*(col_max - col_min) + (col_max + col_min))
                else:
                    data[col] = data[col] + col_min
        print("Back XGBoost treatment done")
        return data

    def load_json(self):
        with open(self.json_file, "r") as jfile:
            self.stats = json.load(jfile, object_pairs_hook=OrderedDict)

    def save_json(self):
        with open(self.json_file, "w") as jfile:
            json.dump(self.stats, jfile, indent=4)

class PandasStandardScaler():
    """
    Class to scale numerical columns
    Author - Dmitry Efimov (dmitry.efimov@aexp.com)

    Mandatory input variables:
        data - Pandas data frame
        target - target column name
        cols - list of columns that should be converted

    Optional input variables:
        stats - Pandas data frame that contains statistics: this parameter is necessary if you convert test data
                (contains 4 columns: colname, colvalue, sum, count)
        replace - if True, the original features will be replaced by likelihoods

    Output -
        data frame with converted columns
        data frame with statistics (optional)

    """
    def __init__(self, json_file):
        self.stats = OrderedDict()
        self.json_file = json_file

    def fit_transform(self, data, cols):
        if os.path.exists(self.json_file):
            self.load_json()
        else:
            # initialize stats dictionary
            self.stats = OrderedDict()
        # calculate statistics by columns
        medians = data[cols].median(axis=0)
        means = data[cols].mean(axis=0)
        stds = data[cols].std(axis=0)
        maxs = data[cols].max(axis=0)
        mins = data[cols].min(axis=0)
        for col in cols:
            self.stats[col] = {}
            if stds[col] == 0:
                data[col] = 0.0
            else:
                data[col+'_pre'] = (data[col] - means[col])/stds[col]
            if data[col + '_pre'].max() > 5.0 or data[col + '_pre'].min() < -5.0:
                data[col] = (2.0*data[col] - (maxs[col] + mins[col]))/(maxs[col] - mins[col])
                self.stats[col]['scaling_type'] = 'minmax'
            else:
                data[col] = data[col + '_pre']
                self.stats[col]['scaling_type'] = 'normal'
            data.drop(col+'_pre', axis=1, inplace=True)
            self.stats[col]['median'] = medians[col]
            self.stats[col]['mean'] = means[col]
            self.stats[col]['std'] = stds[col]
            self.stats[col]['max'] = maxs[col]
            self.stats[col]['min'] = mins[col]
        self.save_json()
        print("Scaling done")
        return data

    def fit(self, data, cols):
        if os.path.exists(self.json_file):
            self.load_json()
        else:
            # initialize stats dictionary
            self.stats = OrderedDict()
        # calculate statistics by columns
        medians = data[cols].median(axis=0)
        means = data[cols].mean(axis=0)
        stds = data[cols].std(axis=0)
        maxs = data[cols].max(axis=0)
        mins = data[cols].min(axis=0)
        for col in cols:
            self.stats[col] = OrderedDict()
            if stds[col] != 0:
                data[col+'_pre'] = (data[col] - means[col])/stds[col]
            if data[col + '_pre'].max() > 5.0 or data[col + '_pre'].min() < -5.0:
                self.stats[col]['scaling_type'] = 'minmax'
            else:
                self.stats[col]['scaling_type'] = 'normal'
            data.drop(col+'_pre', axis=1, inplace=True)
            self.stats[col]['median'] = medians[col]
            self.stats[col]['mean'] = means[col]
            self.stats[col]['std'] = stds[col]
            self.stats[col]['max'] = maxs[col]
            self.stats[col]['min'] = mins[col]
        self.save_json()
        print("Scaling done")

    def transform(self, data, cols):
        self.load_json()
        for col in cols:
            if self.stats[col]['std'] == 0:
                data[col] = 0.0
            else:
                if self.stats[col]['scaling_type'] == 'normal':
                    data[col] = (data[col] - self.stats[col]['mean'])/self.stats[col]['std']
                elif self.stats[col]['scaling_type'] == 'minmax':
                    data[col] = (2.0*data[col] - (self.stats[col]['max'] + self.stats[col]['min']))/(self.stats[col]['max'] - self.stats[col]['min'])
        print("Scaling done")
        return data

    def back_transform(self, data, cols=[]):
        self.load_json()
        if len(cols) == 0:
            cols = [x for x in self.stats.keys()]
        for col in cols:
            if col in self.stats.keys() and col in data.columns:
                if self.stats[col]['scaling_type'] == 'normal':
                    data[col] = data[col]*self.stats[col]['std'] + self.stats[col]['mean']
                elif self.stats[col]['scaling_type'] == 'minmax':
                    data[col] = 0.5*(data[col]*(self.stats[col]['max'] - self.stats[col]['min']) + (self.stats[col]['max'] + self.stats[col]['min']))
        print("Back scaling done")
        return data

    def load_json(self):
        with open(self.json_file, "r") as jfile:
            self.stats = json.load(jfile, object_pairs_hook=OrderedDict)

    def save_json(self):
        with open(self.json_file, "w") as jfile:
            json.dump(self.stats, jfile, indent=4)

class PandasBoxCox():
    """
    Class to apply Box-Cox transformation on Pandas data frame
    Author - Dmitry Efimov (dmitry.efimov@aexp.com)

    Mandatory input variables:
        data - Pandas data frame
        cols - list of columns that should be converted

    Optional input variables:
        stats - Pandas data frame that contains statistics: this parameter is necessary if you convert test data
                (contains 4 columns: colname, colvalue, sum, count)
        replace - if True, the original features will be replaced by likelihoods

    Output -
        data frame with converted columns
        data frame with statistics (optional)

    """
    def __init__(self, json_file, viz_folder=''):
        self.stats = OrderedDict()
        self.json_file = json_file
        self.viz_folder = viz_folder

    def fit_transform(self, data, cols, visualize=1):
        # initialize stats dictionary
        self.stats = OrderedDict()
        # calculate statistics by columns
        medians = data[cols].median(axis=0)
        mins = data[cols].min(axis=0)
        maxs = data[cols].max(axis=0)
        stds = data[cols].std(axis=0)
        for col in cols:
            #if len(data[col].unique()) < 10:
            #    continue
            self.stats[col] = OrderedDict()
            self.stats[col]['median'] = medians[col]
            self.stats[col]['min'] = mins[col]
            self.stats[col]['max'] = maxs[col]
            self.stats[col]['std'] = stds[col]
            # Box-Cox should be applied to positive columns only:
            if mins[col] <= 0:
                self.stats[col]['shift'] = 1.0 - mins[col]
            else:
                self.stats[col]['shift'] = 0.0
            if visualize == 1:
                orig_vals = data.loc[~data[col].isnull(), col].values
                shifted_vals = orig_vals + self.stats[col]['shift']
                log_vals = np.log(shifted_vals)
                boxcox_vals, self.stats[col]['lambd'] = boxcox(shifted_vals)
                plt.figure(num=1, figsize=(16,4))
                gs = gridspec.GridSpec(1, 3)
                gs.update(wspace=0.3)
                ax1 = plt.subplot(gs[0, 0])
                ax2 = plt.subplot(gs[0, 1])
                ax3 = plt.subplot(gs[0, 2])
                ax1.hist(orig_vals, bins=30)
                ax2.hist(log_vals, bins=30)
                ax3.hist(boxcox_vals, bins=30)
                ax1.set_yscale('log')
                ax2.set_yscale('log')
                ax3.set_yscale('log')
                font = {'size': 10}
                plt.gcf().suptitle(col, x=0.11, y=.95, horizontalalignment = 'left', fontsize = 11)
                ax1.set_title('Original', fontdict = font)
                ax2.set_title('Log', fontdict = font)
                ax3.set_title('Box-Cox, lambda = {0:.1f}'.format(self.stats[col]['lambd']), fontdict = font)
                plt.savefig(os.path.join(self.viz_folder, "boxcox_hist_" + col + ".png"))
                plt.clf()
                data.loc[~data[col].isnull(), col] = boxcox_vals
            else:
                data[col] = data[col] + self.stats[col]['shift']
                data.loc[~data[col].isnull(), col], self.stats[col]['lambd'] = boxcox(data.loc[~data[col].isnull(), col].values)
        self.save_json()
        print("Box-Cox transformation done")
        return data

    def fit(self, data, cols):
        # initialize stats dictionary
        self.stats = OrderedDict()
        # calculate statistics by columns
        medians = data[cols].median(axis=0)
        mins = data[cols].min(axis=0)
        maxs = data[cols].max(axis=0)
        stds = data[cols].std(axis=0)
        for col in cols:
            if not col in self.stats.keys():
                continue
            self.stats[col] = OrderedDict()
            self.stats[col]['median'] = medians[col]
            self.stats[col]['min'] = mins[col]
            self.stats[col]['max'] = maxs[col]
            self.stats[col]['std'] = stds[col]
            # Box-Cox should be applied to positive columns only:
            if mins[col] <= 0:
                self.stats[col]['shift'] = 1.0 - mins[col]
            else:
                self.stats[col]['shift'] = 0.0
            _, self.stats[col]['lambd'] = boxcox(data.loc[~data[col].isnull(),col].values + self.stats[col]['shift'])
        self.save_json()
        print("Box-Cox transformation done")

    def transform(self, data, cols):
        self.load_json()
        for col in cols:
            if col in self.stats.keys():
                data[col] = data[col] + self.stats[col]['shift']
                data.loc[data[col]<=0.0, col] = np.nan
                if self.stats[col]['lambd'] == 0.0:
                    data.loc[~data[col].isnull(), col] = np.log(data.loc[~data[col].isnull(), col])
                else:
                    data.loc[~data[col].isnull(), col] = (data.loc[~data[col].isnull(), col]**self.stats[col]['lambd'] - 1.0)/self.stats[col]['lambd']
        print("Box-Cox transformation done")
        return data

    def back_transform(self, data, cols=[]):
        self.load_json()
        if len(cols) == 0:
            cols = [x for x in self.stats.keys()]
        for col in cols:
            if col in self.stats.keys():
                if self.stats[col]['lambd'] == 0.0:
                    data[col] = np.exp(data[col]) - self.stats[col]['shift']
                else:
                    data[col] = data[col]*self.stats[col]['lambd'] + 1.0
                    data[col] = np.power(data[col], 1.0/self.stats[col]['lambd']) - self.stats[col]['shift']
                data[col] = data[col].clip(lower = self.stats[col]['min'] - 0.0*self.stats[col]['std'],
                                           upper = self.stats[col]['max'] + 0.0*self.stats[col]['std'])
        return data

    def load_json(self):
        with open(self.json_file, "r") as jfile:
            self.stats = json.load(jfile, object_pairs_hook=OrderedDict)

    def save_json(self):
        with open(self.json_file, "w") as jfile:
            json.dump(self.stats, jfile, indent=4)

def JSD(P, Q):
    """
    Function that computes Jensen-Shannon divergence
    Note: scipy entropy is Kullback-Leibler divergence
    Author - Dmitry Efimov (dmitry.efimov@aexp.com)
    """
    _P = P / norm(P, ord=1)
    _Q = Q / norm(Q, ord=1)
    _M = 0.5 * (_P + _Q)
    return 0.5 * (entropy(_P, _M) + entropy(_Q, _M))

def gini_binary(y_true, y_pred):
    """
    Function for calculating Gini coefficient if target variable is binary
    Author - Dmitry Efimov (dmitry.efimov@aexp.com)

    Mandatory input variables:
        y_true - one dimensional Numpy array with ground truth
        y_pred - one dimensional Numpy array with predicted values

    Output -
        Gini coefficient scaled to [0,1]

    """
    return 100.0*(2*metrics.roc_auc_score(y_true=y_true, y_score=y_pred)-1)

def gini(y_true, y_pred):
    """
    Function for calculating Gini coefficient
    This function can be applied to any continuous input
    Author - Dmitry Efimov (dmitry.efimov@aexp.com)

    Mandatory input variables:
        y_true - one dimensional Numpy array with ground truth
        y_pred - one dimensional Numpy array with predicted values

    Output -
        Gini coefficient scaled to [0,1]

    """
    # check and get number of samples
    assert y_true.shape == y_pred.shape
    n_samples = y_true.shape[0]

    # sort rows on prediction column (from largest to smallest)
    arr = np.array([y_true, y_pred]).transpose()
    true_order = arr[arr[:,0].argsort()][::-1,0]
    pred_order = arr[arr[:,1].argsort()][::-1,0]

    # get Lorenz curves
    L_true = np.cumsum(true_order) / np.sum(true_order)
    L_pred = np.cumsum(pred_order) / np.sum(pred_order)
    L_ones = np.linspace(1/n_samples, 1, n_samples)

    # get Gini coefficients (area between curves)
    # we can use continuous variables 
    # => we evaluate to AUC and after take the ratios to normalize them
    G_true = np.sum(L_ones - L_true)
    G_pred = np.sum(L_ones - L_pred)

    # normalize to true Gini coefficient
    return G_pred/G_true

def capture_rate(actual, predicted, at=2.0, sample_size=100.0):
    preds = np.column_stack((predicted,actual,np.ones(len(actual))))
    preds = preds[preds[:,0].argsort()][::-1]
    preds[np.where(preds[:,1]==0.0),2] = 100.0/sample_size
    topAtBps = (at/100)*np.sum(preds[:,2])
    threshold = np.abs(np.cumsum(preds[:,2]) - topAtBps).argmin()
    capture_rate = 100.0*preds[0:threshold,1].sum()/preds[:,1].sum()
    return capture_rate

def xgboost_get_importance(model_file, flist):
    import xgboost as xgb
    bst = xgb.Booster({'nthread':8})
    bst.load_model(model_file)
    importance = bst.get_score(importance_type='gain')
    importance = [x for x in reversed(sorted(importance.items(), key=operator.itemgetter(1)))]
    importance = [(flist[int(x[0][1:])], x[1]) for x in importance]
    return importance

def accuracy_index(true, pred, wgt=0, bins=10):
    """
    Calculate the accuracy index with weight
    If wgt is not specified, default wgt is set to one, i.e. no weight
    """
    df = pd.DataFrame({'true':true, 'pred':pred})
    if isinstance(wgt, int):
        df['wgt'] = 1
    else:
        df['wgt'] = wgt
    df = df.sort_values(by='pred', ascending=False)
    df['true'] = df['true'] * df['wgt']
    df['pred'] = df['pred'] * df['wgt']
    df['bin'] = np.floor(df['wgt'].cumsum() / df['wgt'].sum() * bins) + 1
    df['bin'] = np.where(df['bin'] > bins, bins, df['bin'])
    df_agg = df.groupby('bin').agg({'bin': 'count', 'true': 'sum', 'pred': 'sum'})
    try:
        accuracy = 1 - np.abs(df_agg['pred'] - df_agg['true']).sum() / np.abs(df_agg['true']).sum()
        return accuracy
    except ZeroDivisionError:
        return np.nan

def DiagonalCMV(final_line, opt_line, cmv, mode="round"):
    data = pd.DataFrame()
    data['final_line'] = final_line
    data['opt_line'] = opt_line
    data['cmv'] = cmv
    if mode == "round":
        lines = sorted(data['opt_line'].unique())
        diffs = np.abs(data[['final_line']].values - np.ones((len(data), len(lines)))*lines)
        data['final_line'] = np.reshape([lines[x] for x in np.argmin(diffs, axis=1)], (-1,1))
    def percentile(n):
        def percentile_(x):
            return np.percentile(x, n)
        percentile_.__name__ = 'percentile_%s' % n
        return percentile_

    data = data.loc[~data['cmv'].isnull()]
    cmv_percentile = data.groupby(['final_line', 'opt_line']).agg({'cmv': [percentile(1), percentile(99)]}).reset_index()
    cmv_percentile.columns = ['final_line','opt_line','percentile01','percentile99']
    data = pd.merge(data, cmv_percentile, on=['final_line','opt_line'])
    data = data.query('cmv > percentile01 & cmv < percentile99')
    if len(data) == 0: return 0.0
    cmv_data = data.groupby(['final_line', 'opt_line']).agg({'cmv': [len, np.mean, np.sum]}).reset_index()
    cmv_data.columns = ['final_line','opt_line','count','cmv', 'cmv_sum']

    cmv_means = data.groupby(['final_line', 'opt_line'])['cmv'].mean()
    cmv_diag = cmv_means.iloc[cmv_means.index.get_level_values('final_line') == cmv_means.index.get_level_values('opt_line')]
    cmv_diag_frame = cmv_diag.to_frame()
    cmv_diag_frame = cmv_diag_frame.reset_index(level=['final_line', 'opt_line'])

    row_counts = data.groupby('opt_line').size()
    row_counts_frame = row_counts.to_frame()
    row_counts_frame = row_counts_frame.reset_index()
    row_counts_frame.columns = ['opt_line', 'counts']

    cmv_calc = pd.merge(cmv_diag_frame, row_counts_frame, on='opt_line')
    cmv_calc['prod'] = cmv_calc['cmv']*cmv_calc['counts']
    cmv_val = cmv_calc['prod'].sum()/cmv_calc['counts'].sum()
    diag_count = cmv_calc['counts'].sum()

    return cmv_val

