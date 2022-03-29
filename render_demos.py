import os
import sys
import argparse
import glob

DEVICE=2
INSTRUMENT="Saxophone"

#checkpoint_path=f"checkpoints/48k_bidir_z512_conv_family_{INSTRUMENT}_f0c"
checkpoint_path=f"artefacts/training/Saxophone/ckpt-380000"
DATASET_PATHS=f"./datasets/AIR/tfr/tst/{INSTRUMENT}/*"


ds_paths=glob.glob(DATASET_PATHS)
print(ds_paths)
for ds_path in ds_paths:
    os.system(f"CUDA_VISIBLE_DEVICES={DEVICE} python adaptation.py --pretrained_checkpoint_path={checkpoint_path} --cloning_dataset_path={ds_path}")
    os.system(f"CUDA_VISIBLE_DEVICES={DEVICE} python adaptation.py --pretrained_checkpoint_path={checkpoint_path} --cloning_dataset_path={ds_path} --finetune_whole")

print("Done!")