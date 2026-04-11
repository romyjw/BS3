import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
from torch import autograd as Grad

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
import matplotlib.tri as mtri
import matplotlib
from IPython.display import display, clear_output

import ipywidgets as widgets
import trimesh
import random
import math
import time
import os
import json
import copy

import open3d as o3d
print(o3d.__path__)
print("Using open3d version", o3d.__version__)
o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

from visuals import *
from bns_utils import *
from mesh_processing import *

import differential
import importlib
importlib.reload(differential)
from differential import *

two_pi = 2 * torch.pi
diffmod = DifferentialModule()

# -------------------------------
# Argument parsing
# -------------------------------
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--config-filepath", type=str, required=True)
parser.add_argument("--output-filepath", type=str, required=True)

args = parser.parse_args()

config_filepath = args.config_filepath
with open(config_filepath, "r") as f:
    config_dict = json.load(f)
    surface_config = config_dict['surface-config']
    training_config = config_dict['training-config']




output_filepath = args.output_filepath


# -------------------------------
# Device
# -------------------------------
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")
# -------------------------------


# -------------------------------
# BPS setup
# -------------------------------
import BPS
importlib.reload(BPS)
from BPS import BPS_fast

bps = BPS_fast(surface_config, device=device)

mobius_example = False
if 'mobius_example' in surface_config.keys() and surface_config['mobius_example']==True:
    mobius_example=True

if mobius_example==True:
    bps.do_mobius_edits()




# -------------------------------
# Sample generation
# -------------------------------
training_samples = bps.compute_samples(num_samples=training_config['num_samples_per_face'], num_bdry_samples= training_config['num_bdry_samples_per_edge']*3)


x = training_samples["uv"]



precomputed_training_data = bps.precompute_data_from_samples(
    x, detached=True, mobius_example=mobius_example
)

# -------------------------------
# Save
# -------------------------------
os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
torch.save(precomputed_training_data, output_filepath)

print(f"Saved precomputed samples to {output_filepath}")
