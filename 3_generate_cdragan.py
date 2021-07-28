from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
import pandas as pd
import argparse
import os
import sys
import json
import re
import glob

parser = argparse.ArgumentParser()
parser.add_argument("--settings_file", help="JSON settings file")
parser.add_argument("--device", help="GPU device")
parser.add_argument("--version", help="GAN configuration version")
parser.add_argument("--segment", help="which segment to generate")
parser.add_argument("--num_sample_sim", help="how many samples to generate")
parser.add_argument("--using_iter", help="which model iteration to use for data generation")

args = parser.parse_args()
settings_file = args.settings_file
device = args.device
version = args.version
segment = args.segment
num_sample_sim = int(args.num_sample_sim)
using_iter = int(args.using_iter)

with open(settings_file, "r") as jfile:
    settings = json.load(jfile)

sys.path.append(settings['path_to_library'])
from gan_net import *

scaled_train_file = settings['scaled_train_file']
gen_scaled_train_file = settings['gen_scaled_train_file']
scaled_header_file = settings['scaled_header_file']
gan_pars = settings['gan_pars']
model_folder = settings['model_folder']
error_file = settings['error_file']

if version != '':
    gan_pars = gan_pars.format(version)
    model_folder = model_folder.format(version)
    error_file = error_file.format(version)
    gen_scaled_train_file = gen_scaled_train_file.format(version)

if num_sample_sim == 0:
    num_sample_sim = len(pd.read_csv(scaled_train_file, usecols = [0], dtype=object, header=True))

if using_iter == 0:
    errors = pd.read_csv(error_file, header=None)
    errors.columns = ['desc']
    errors['iter'] = errors['desc'].apply(lambda x: float(re.search("(?<=Iter:\s).*?(?=;)", x).group(0))).astype(int)
    errors['G_loss'] = errors['desc'].apply(lambda x: float(re.search("(?<=G loss:\s).*?(?=;)", x).group(0))).astype(float)
    errors['D_loss'] = errors['desc'].apply(lambda x: float(re.search("(?<=D loss:\s).*?(?=;)", x).group(0))).astype(float)
    errors['similarity'] = errors['desc'].apply(lambda x: float(re.search("(?<=similarity:\s).*?(?=$)", x).group(0))).astype(float)
    errors.drop('desc', axis=1, inplace=True)

    existed_iters = []
    for x in sorted(glob.glob(os.path.join(model_folder,"check_point_*.ckpt.meta"))):
        existed_iters.append(int(re.split("_|\.", os.path.basename(x))[2]))
    data_iters = pd.DataFrame()
    data_iters['iter'] = existed_iters

    errors = pd.merge(errors, data_iters, on='iter', how='inner')
    errors = errors.sort_values(by='similarity', ascending=True).reset_index(drop=True)
    print(errors)
    using_iter = errors.loc[0, 'iter']
    print("The best iteration is %d" % (using_iter))

net = CDRAGAN(scaled_header_file,
              gan_pars,
              model_folder,
              error_file,
              device)

net.generate(scaled_train_file,
             gen_scaled_train_file,
             num_sample_sim,
             using_iter,
             how=segment)

'''
num_sample_sim = num_sample_sim*2
net.generate_disc(scaled_train_file,
                  gen_scaled_train_file,
                  num_sample_sim,
                  using_iter,
                  how=segment)
'''
