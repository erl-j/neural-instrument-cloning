#%%

import glob

# import youtube api


DATASET_DIR="datasets/AIR/tfr/"

# get all filenames in tst and trn split
tst_filenames=glob.glob(f"{DATASET_DIR}/tst/*.tfr")
dev_filenames=glob.glob(f"{DATASET_DIR}/dev/*.tfr")

# get youtube ids from filenames
tst_ids=[fn.split("/")[-1].split("_part")[0] for fn in tst_filenames]
trn_ids=[fn.split("/")[-1].split("_part")[0] for fn in dev_filenames]

# use youtube api to get video uploader
tst_uploaders=[youtube.get_video_uploader(id) for id in tst_ids]
trn_uploaders=[youtube.get_video_uploader(id) for id in trn_ids]