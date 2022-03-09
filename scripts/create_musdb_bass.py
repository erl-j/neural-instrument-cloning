import glob
import random
import os
import shutil

random.seed(42)

fp = glob.glob("./datasets/musdb18hq/*/*/bass.wav",recursive=True)

print(len(fp))

OUT_DIR =  "./datasets/musdb18hq_bass/"

# 10 files for testing
test_files = random.sample(fp, 10)

# rest in dev set
dev_files = [f for f in fp if f not in test_files]

# show intersection between test and dev
print(set(test_files).intersection(set(dev_files)))


# create directories for test and dev sets
os.makedirs(OUT_DIR + "tst/",exist_ok=True)
os.makedirs(OUT_DIR + "dev/",exist_ok=True)

# copy files to new directory
for f in test_files:
    song_name=f.split("/")[-2]
    # copy file to OUT_DIR
    shutil.copy(f, OUT_DIR+"tst/"+song_name+".wav")
for f in dev_files:
    song_name=f.split("/")[-2]
    # copy file to OUT_DIR
    shutil.copy(f, OUT_DIR+"dev/"+song_name+".wav")








