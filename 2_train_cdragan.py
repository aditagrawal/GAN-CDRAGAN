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

parser = argparse.ArgumentParser()
parser.add_argument("--settings_file", help="JSON settings file")
parser.add_argument("--device", help="GPU device")
parser.add_argument("--version", help="GAN configuration version")

args = parser.parse_args()
settings_file = args.settings_file
device = args.device
version = args.version

with open(settings_file, "r") as jfile:
    settings = json.load(jfile)

sys.path.append(settings['path_to_library'])
from gan_net import *

scaled_train_file = settings['scaled_train_file']
scaled_hdf_file = settings['scaled_hdf_file']
scaled_header_file = settings['scaled_header_file']
gan_pars = settings['gan_pars']
model_folder = settings['model_folder']
error_file = settings['error_file']
gen_scaled_train_file = settings['gen_scaled_train_file']

if version != '':
    gan_pars = gan_pars.format(version)
    model_folder = model_folder.format(version)
    error_file = error_file.format(version)
    gen_scaled_train_file = gen_scaled_train_file.format(version)

net = CDRAGAN(scaled_header_file,
              gan_pars,
              model_folder,
              error_file,
              device)
net.fit(scaled_train_file,
        print_similarity=True,
        output_file = gen_scaled_train_file,
        colTarget = 'def_new')

