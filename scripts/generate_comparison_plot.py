#%% imports

import pandas as pd
from matplotlib import pyplot as plt
from pyrsistent import s
import seaborn as sns
from scipy import stats
import tensorboard as tb
import os
import glob 
import tensorflow as tf
from tensorflow.core.util import event_pb2
import math
#%%

ROOT_DIR="../plot"



SCHEMES=os.listdir(ROOT_DIR)

SCHEMES=list(filter(lambda x: not x.startswith("sax_pretrained_only"),SCHEMES))

print(SCHEMES)

summaries={}

TRN_DATA_DURATIONS=[4,8,16,32,64,128,256]
for scheme in SCHEMES:
    scheme_dir=f"{ROOT_DIR}/{scheme}"
    fps=os.listdir(scheme_dir)
    for duration in TRN_DATA_DURATIONS:
        fp=list(filter(lambda x: f"trn_data_duration={duration}_" in x,fps))[0]
        for split in ["tst","trn"]:
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


PRETRAINED_DIR="../plot/sax_pretrained_only"

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

# make figure look nice
sns.set_style("whitegrid")
sns.set_context("paper")
# remove horizontal lines from grid but keep vertical lines
#sns.set_style("ticks", {'grid.linestyle': '-'})

scheme_display_names={
    "scratch":"FROM SCRATCH",
    "sax_whole":"FINETUNE WHOLE",
    "sax_partial":"FINETUNE PARTS"
}

print(SCHEMES)

scheme_display_names =  {**{k:k for k in SCHEMES}, **scheme_display_names}

print(scheme_display_names)

split_display_names={
    "tst":"val",
    "trn":"trn"
}

# plot losses, tst as solid, trn as dashed, colour according to scheme.
exponents=[math.log(d,2)-1 for d in TRN_DATA_DURATIONS]

EXPERIMENTS=[
    ["scratch_nof0c","scratch"],
    ["scratch","scratch_free_reverb"],
    #["scratch_nof0c","scratch_nof0c_free_reverb","scratch","scratch_free_reverb"],
    ["scratch","sax_partial","sax_whole"],
    ["scratch","sax_whole","swc_nosax_whole"],
    ["scratch","sax_partial","swc_nosax_partial"],
]

COLOR_PALETTE=sns.color_palette("deep",len(SCHEMES))

for DISPLAY_SCHEMES in EXPERIMENTS:

    # 

    PLOT_TST=True
    if PLOT_TST:
        for scheme in DISPLAY_SCHEMES:
                for split in ["tst"]:
                    losses=[summaries[f"{scheme}_{duration}_{split}"]["loss"] for duration in TRN_DATA_DURATIONS]  
                    color=COLOR_PALETTE[SCHEMES.index(scheme)]
                    plt.plot(exponents,losses,label=f"{scheme_display_names[scheme]}, {split_display_names[split]}",linestyle="--" if split=="trn" else "-",color=color)          

    PLOT_TRN=True
    if PLOT_TRN:
        for scheme in DISPLAY_SCHEMES:
                for split in ["trn"]:
                    losses=[summaries[f"{scheme}_{duration}_{split}"]["loss"] for duration in TRN_DATA_DURATIONS]  
                    color=COLOR_PALETTE[SCHEMES.index(scheme)]
                    plt.plot(exponents,losses,label=f"{scheme_display_names[scheme]}, {split_display_names[split]}",linestyle="--" if split=="trn" else "-",color=color,alpha=0.5)          

    # y axis starts at 0 and ends at the maximum loss
    plt.ylim(6,17)

    # set ticks equally spaced at trn data durations
    plt.xticks(exponents,TRN_DATA_DURATIONS)
    plt.legend()

    # add dotted horizontal line at nearest and label on the right

    #plt.axhline(nearest,0,len(TRN_DATA_DURATIONS),linestyle=":",alpha=1,label="NEAREST")
    #plt.text(len(TRN_DATA_DURATIONS)-0.5,nearest,f"nearest")

    # add crosses on points if they don't have audio examples
    for scheme in SCHEMES:
        for duration in TRN_DATA_DURATIONS:
            if not summaries[f"{scheme}_{duration}_tst"]["has_audio_examples"]:
                plt.plot(exponents[TRN_DATA_DURATIONS.index(duration)],summaries[f"{scheme}_{duration}_tst"]["loss"],'x')
    # show plot
    plt.legend()
    plt.xlabel("Training data duration (seconds)")
    plt.ylabel("Multiscale spectral loss")
    plt.title("Loss vs Training data duration")

    # make lines thicker
    for i,line in enumerate(plt.gca().lines):
        line.set_linewidth(2)


    # make figure wider
    plt.gcf().set_size_inches(10,10)

    plt.figure(figsize=(10,10))

    # hide legend

    plt.show()


# %%

# %%

# %%
