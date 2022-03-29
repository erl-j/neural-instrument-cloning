# imports and utils

import tensorflow.compat.v2 as tf
import ddsp.training
_AUTOTUNE = tf.data.experimental.AUTOTUNE
from IPython.display import Audio, display
from livelossplot import PlotLosses
import numpy as np
import random 
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
import time
import data
import random
import copy
import pydash
import tqdm
import soundfile
import os
import shared_model
import pandas as pd
import datetime
import glob
import argparse, sys

# define constants
CLIP_S=4
SAMPLE_RATE=48000
N_SAMPLES=SAMPLE_RATE*CLIP_S
SEED=1
FT_FRAME_RATE=250

tf.random.set_seed(
    SEED
)
np.random.seed(SEED)
random.seed(SEED)

# define some utilis
def play(audio):
  display(Audio(audio,rate=SAMPLE_RATE))


IR_DURATION=1
Z_SIZE= 512
N_INSTRUMENTS=200
BIDIRECTIONAL=True
USE_F0_CONFIDENCE=True
N_NOISE_MAGNITUDES=192
N_HARMONICS=192

# define loss
fft_sizes = [64]
while fft_sizes[-1]<SAMPLE_RATE//4:
    fft_sizes.append(fft_sizes[-1]*2)

spectral_loss = ddsp.losses.SpectralLoss(loss_type='L1',
                                            fft_sizes=fft_sizes,
                                            mag_weight=1.0,
                                            logmag_weight=1.0)
