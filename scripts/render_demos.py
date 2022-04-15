import os
import sys
import argparse
import glob
from xml.sax import default_parser_list

#DEVICE=2
#INSTRUMENT="Saxophone"

# read device and instrument from command line
parser = argparse.ArgumentParser()
parser.add_argument("--device", type=int)
parser.add_argument( "--checkpoint_path", type=str,)
parser.add_argument( "--dataset_pattern", type=str,)
args = parser.parse_args()



ds_paths=glob.glob(args.dataset_pattern)

for ds_path in reversed(ds_paths):
    os.system(f"CUDA_VISIBLE_DEVICES={args.device} python cloning.py --pretrained_checkpoint_path={args.checkpoint_path} --cloning_dataset_path={ds_path}")
    # os.system(f"CUDA_VISIBLE_DEVICES={DEVICE} python adaptation.py --pretrained_checkpoint_path={checkpoint_path} --cloning_dataset_path={ds_path} --finetune_whole")

print("Done!")