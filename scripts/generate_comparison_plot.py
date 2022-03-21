#%% imports
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import stats
import tensorboard as tb
import os
import glob
import tensorflow as tf
from tensorflow.core.util import event_pb2
import math
#%%

ROOT_DIR="../comparison_plot_data/plot"

SCHEMES=["sax_whole","sax_partial","scratch"]


summaries={}

TRN_DATA_DURATIONS=[4,8,16,32,64,128,256]
for scheme in SCHEMES:
    scheme_dir=f"{ROOT_DIR}/{scheme}"
    fps=os.listdir(scheme_dir)

    for duration in TRN_DATA_DURATIONS:

        fp=list(filter(lambda x: f"trn_data_duration={duration}_" in x,fps))[0]
        for split in ["trn","tst"]:
            split_dir=f"{scheme_dir}/{fp}/{split}"

            event_fp=glob.glob(f"{split_dir}/*.v2")[-1]

            for summary in tf.compat.v1.train.summary_iterator(event_fp):
                for v in summary.summary.value:
                    #print(v.tag)
                    if v.tag=="loss":
                        #print(v.simple_value)
                        t = tf.make_ndarray(v.tensor)
                        #print(v.tag, summary.step, t, type(t))
            summaries[f"{scheme}_{duration}_{split}"]={"n_steps":summary.step,"loss":float(t),"has_audio_examples":os.path.exists(f"{scheme_dir}/{fp}/unseen estimate.wav")}
                
print(summaries)
# %%


PRETRAINED_DIR="../comparison_plot_data/plot/sax_pretrained_only"

pretrained_test_losses=[]

fps=os.listdir(PRETRAINED_DIR)

for fp in fps:
    for split in ["tst"]:
        split_dir=f"{PRETRAINED_DIR}/{fp}/**/{split}"

        event_fp=glob.glob(f"{split_dir}/*.v2",recursive=True)[0]

        for summary in tf.compat.v1.train.summary_iterator(event_fp):
            for v in summary.summary.value:
                #print(v.tag)
                if v.tag=="loss":
                    #print(v.simple_value)
                    t = tf.make_ndarray(v.tensor)
                    #print(v.tag, summary.step, t, type(t))
                pretrained_test_losses.append(float(t))
                
nearest=min(pretrained_test_losses)
furthest=max(pretrained_test_losses)
#%% PLOT

# plot losses

exponents=[math.log(d,2)-1 for d in TRN_DATA_DURATIONS]
for scheme in SCHEMES:
        for split in ["trn","tst"]:
            losses=[summaries[f"{scheme}_{duration}_{split}"]["loss"] for duration in TRN_DATA_DURATIONS]
            plt.plot(exponents,losses,label=f"{scheme}_{split}")

# set ticks equally spaced at trn data durations
plt.xticks(exponents,TRN_DATA_DURATIONS)
plt.legend()


# add dotted horizontal line at nearest and label on the right
plt.hlines(nearest,0,len(TRN_DATA_DURATIONS),linestyles="dotted")
plt.text(len(TRN_DATA_DURATIONS)-0.5,nearest,f"nearest")

# plt.hlines(furthest,0,len(TRN_DATA_DURATIONS),linestyles="dotted")
# plt.text(0.5,furthest,f"furthest",horizontalalignment="center")

# show plot
plt.legend()
plt.xlabel("Training data duration")
plt.ylabel("Loss")
plt.title("Loss vs Training data duration")
plt.show()

# %%
