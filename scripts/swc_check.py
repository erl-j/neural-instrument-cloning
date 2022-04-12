#%%
import glob
from xml.etree.ElementTree import TreeBuilder

# import youtube api
DATASET_DIR="../datasets/AIR/tfr"

# get all filenames in tst and trn split
tst_filenames=glob.glob(f"{DATASET_DIR}/tst/**/*-of-*",recursive=True)
dev_filenames=glob.glob(f"{DATASET_DIR}/dev/**/*-of-*",recursive=True)

print(tst_filenames)
print(dev_filenames)
#%%
# get youtube ids from filenames
tst_ids=[fn.split("/")[-1].split("_part")[0] for fn in tst_filenames]
trn_ids=[fn.split("/")[-1].split("_part")[0] for fn in dev_filenames]

# write to text file
with open("../paper/experiments/tst_ids.txt","w") as f:
    for id in tst_ids:
        f.write(id+"\n")
with open("../paper/experiments/trn_ids.txt","w") as f:
    for id in trn_ids:
        f.write(id+"\n")

# use youtube api to get video uploader


# %%
