#%%
import os
import librosa
import datetime

for split in ["tst","dev"]:
    for instrument_name in ["Bassoon","Clarinet","Flute","Horn","Oboe","Saxophone","Trumpet","Trombone","Tuba"]:
        target_dir=f"../datasets/AIR/wav/{split}/{instrument_name}"

        # how many wav files are in directory
        n_files=len([filename for filename in os.listdir(target_dir) if filename.endswith(".wav")])

        # count how many files end with
          # how many wav files are in directory
        n_recordings=len([filename for filename in os.listdir(target_dir) if filename.endswith("_part0.wav")])

        #print(f"{instrument_name}, {split} : {n_files} files")
        s=0
        # compute the duration of a all audio files in a directory
        for filename in os.listdir(target_dir):
            if filename.endswith(".wav"):
                y,sr = librosa.load(f"{target_dir}/{filename}",sr=16000)
                print(f"instrument_name={instrument_name}, split={split}, filename={filename}, duration={len(y)/sr}")
                s+=len(y)/sr

        #print(f"summary > instrument_name={instrument_name}, split={split}, n_files={n_files}, n_recordings={n_recordings}, total_duration={s}, avg duration={s/n_files} ")
        # same but duration in h:m:s
        print(f"summary > instrument_name={instrument_name}, split={split}, n_files={n_files}, n_recordings={n_recordings}, total_duration={datetime.timedelta(seconds=int(s))}, avg duration={datetime.timedelta(seconds=int(s/n_files))} ")


# %%
