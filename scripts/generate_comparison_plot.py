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
from collections import *
#%%

ROOT_DIR="../paper/experiments"


# SCHEMES=os.listdir(ROOT_DIR)

# SCHEMES=list(filter(lambda x: not x.startswith("sax-pretraining"),SCHEMES))

# print(SCHEMES)

SCHEMES=[
'sax-parts',
'nosax-parts',
'sax-whole',
'nosax-whole', 
'init-whole',
'init-whole-fn_reverb',
'init-whole-free_reverb', 
'init-whole-no_f0conf',
'init-whole-free_reverb-no_tanh',
'init-whole-no_f0conf-free_reverb',
]


summaries={}


TRN_DATA_DURATIONS=[4,8,16,32,64,128,256]

fig,axes=plt.subplots(nrows=len(SCHEMES),ncols=len(TRN_DATA_DURATIONS),sharex=False,sharey=True,figsize=(2*8.27,2*11.69))


for scheme in SCHEMES:
    scheme_dir=f"{ROOT_DIR}/{scheme}"
    fps=os.listdir(scheme_dir)
    for duration in TRN_DATA_DURATIONS:
        fp=list(filter(lambda x: f"trn_data_duration={duration}_" in x,fps))[0]
        losses={"tst":{"step":[],"loss":[]},"trn":{"step":[],"loss":[]}}
        for split in ["tst","trn"]:
            split_dir=f"{scheme_dir}/{fp}/{split}"

            event_fp=glob.glob(f"{split_dir}/*.v2")[-1]


            for summary in tf.compat.v1.train.summary_iterator(event_fp):
                for v in summary.summary.value:
                    #print(v.tag)
                    if v.tag=="loss":
                        #print(v.simple_value)
                        t = tf.make_ndarray(v.tensor)
                        losses[split]["loss"].append(t)
                        losses[split]["step"].append(summary.step)
                        #print(v.tag, summary.step, t, type(t))
            summaries[f"{scheme}_{duration}_{split}"]={"n_steps":summary.step,"loss":float(t),"has_audio_examples":os.path.exists(f"{scheme_dir}/{fp}/unseen estimate.wav")}
        # add the losses to the plot
        axes[SCHEMES.index(scheme),TRN_DATA_DURATIONS.index(duration)].plot(losses["trn"]["step"],losses["trn"]["loss"],label=f"trn")
        axes[SCHEMES.index(scheme),TRN_DATA_DURATIONS.index(duration)].plot(losses["tst"]["step"],losses["tst"]["loss"],label=f"tst")
        # set y limit to 20
        axes[SCHEMES.index(scheme),TRN_DATA_DURATIONS.index(duration)].set_ylim(6,20)

        # remove margins
        axes[SCHEMES.index(scheme),TRN_DATA_DURATIONS.index(duration)].margins(x=0)
# give figure same aspect ratio as an A4 paper


FONT_SIZE=17

# label columns with training data durations
for i,duration in enumerate(TRN_DATA_DURATIONS):
    axes[0,i].set_title(f"{duration} s",fontsize=FONT_SIZE)

# label rows with schemes with a large font
for i,scheme in enumerate(SCHEMES):
    axes[i,0].set_ylabel(scheme,fontsize=FONT_SIZE)

# add legend for whole figure
axes[0,-1].legend(loc="upper right")

fig.tight_layout()

# reduce vertical space between subplots
fig.subplots_adjust(hspace=0.1)

# reduce horizontal space between subplots
fig.subplots_adjust(wspace=0.03)




plt.savefig("../paper/plots/loss_plots.png")
plt.show()

plt.clf()

print(summaries)
 # %%

PRETRAINED_DIR="../paper/experiments/sax-pretraining"

pretrained_test_losses=[]

fps=os.listdir(PRETRAINED_DIR)
event_fps=[]

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
                event_fps.append(event_fp)


                
sax_nearest=min(pretrained_test_losses)
sax_nearest_idx=pretrained_test_losses.index(sax_nearest)
print(event_fps[sax_nearest_idx])
sax_furthest=max(pretrained_test_losses)
 # %%

PRETRAINED_DIR="../paper/experiments/nosax-pretraining"

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
                event_fps.append(event_fp)

nosax_nearest=min(pretrained_test_losses)
nosax_nearest_idx=pretrained_test_losses.index(nosax_nearest)
print(event_fps[nosax_nearest_idx])

nosax_furthest=max(pretrained_test_losses)

#%% PLOT

# make figure look nice
sns.set_style("whitegrid")
sns.set_context("paper")
# remove horizontal lines from grid but keep vertical lines
#sns.set_style("ticks", {'grid.linestyle': '-'})

scheme_display_names={
}

print(SCHEMES)

scheme_display_names =  {**{k:k for k in SCHEMES}, **scheme_display_names}

print(scheme_display_names)

split_display_names={
    "tst":"test",
    "trn":"train"
}

# plot losses, tst as solid, trn as dashed, colour according to scheme.
exponents=[math.log(d,2)-1 for d in TRN_DATA_DURATIONS]

EXPERIMENTS=[
    ["init-whole-no_f0conf","init-whole"],
    ["init-whole","init-whole-fn_reverb","init-whole-free_reverb"],
    ["init-whole-no_f0conf","init-whole-no_f0conf-free_reverb","init-whole","init-whole-free_reverb"],
    ["init-whole","sax-parts","sax-whole"],
    ["init-whole","sax-whole","nosax-whole"],
    ["init-whole","sax-parts","nosax-parts"],
    ["init-whole","sax-parts","nosax-parts","sax-whole","nosax-whole"],
    ["init-whole","init-whole-free_reverb-no_tanh","init-whole-fn_reverb","init-whole-free_reverb"],
]

COLOR_PALETTE=sns.color_palette("deep",len(SCHEMES))

# assign colors to schemes
scheme_colors={}
for i,scheme in enumerate(SCHEMES):
    scheme_colors[scheme]=COLOR_PALETTE[i]



for DISPLAY_SCHEMES in EXPERIMENTS:
    # add dotted horizontal line at nearest

    if "sax-parts" in DISPLAY_SCHEMES and "sax-whole" in DISPLAY_SCHEMES:
        plt.axhline(sax_nearest,0,len(TRN_DATA_DURATIONS),linestyle=":",alpha=1,color="blue")
        # add label
        plt.text(2.3*len(TRN_DATA_DURATIONS)/4,sax_nearest+0.1,f"sax-nearest",fontsize=8.5,color="blue")
        if "nosax-parts" in DISPLAY_SCHEMES and "nosax-whole" in DISPLAY_SCHEMES:
            plt.axhline(nosax_nearest,0,len(TRN_DATA_DURATIONS),linestyle=":",alpha=1,color="red")
            # add label
            plt.text(2.3*len(TRN_DATA_DURATIONS)/4,nosax_nearest+0.1,f"nosax-nearest",fontsize=8.5,color="red")
        
    # 

    data={
            "x":[],
            "y":[],
            "approach":[],
            "split":[],
    }
    PLOT_TST=True
    if PLOT_TST:
        for scheme in DISPLAY_SCHEMES:
                for split in ["tst"]:
                    losses=[summaries[f"{scheme}_{duration}_{split}"]["loss"] for duration in TRN_DATA_DURATIONS]  
                    data["x"].extend(exponents)
                    data["y"].extend(losses)
                    data["approach"].extend([scheme_display_names[scheme]]*len(losses))
                    data["split"].extend([split_display_names[split]]*len(losses))

    PLOT_TRN=True
    if PLOT_TRN:
        for scheme in DISPLAY_SCHEMES:
                for split in ["trn"]:
                    losses=[summaries[f"{scheme}_{duration}_{split}"]["loss"] for duration in TRN_DATA_DURATIONS]  
                    data["x"].extend(exponents)
                    data["y"].extend(losses)
                    data["approach"].extend([scheme_display_names[scheme]]*len(losses))
                    data["split"].extend([split_display_names[split]]*len(losses))
        
    # use scheme_colors to assign colors to schemes

    sns.lineplot(data=data,x="x",y="y",hue="approach",style="split",palette=scheme_colors)
    
    # y axis starts at 0 and ends at the maximum loss
    plt.ylim(6,17)

    # set ticks equally spaced at trn data durations
    plt.xticks(exponents,TRN_DATA_DURATIONS)

    # add crosses on points if they don't have audio examples
    for scheme in SCHEMES:
        for duration in TRN_DATA_DURATIONS:
            if not summaries[f"{scheme}_{duration}_tst"]["has_audio_examples"]:
                plt.plot(exponents[TRN_DATA_DURATIONS.index(duration)],summaries[f"{scheme}_{duration}_tst"]["loss"],'x')

    plt.xlabel("Training data size (seconds)")
    plt.ylabel("Multiscale spectral loss")

    # make lines thicker
    for i,line in enumerate(plt.gca().lines):
        line.set_linewidth(1.5)

    plt.savefig("../paper/plots/"+"_".join(DISPLAY_SCHEMES)+".png")

    plt.show()



# %%

# %%

# %%

# %%
