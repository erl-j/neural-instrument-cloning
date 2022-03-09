import librosa
import os
import glob

target_pattern="./datasets/musdb18hq_bass/**/*.wav"

fps=glob.glob(target_pattern, recursive=True)
# for all files in pattern check for errors
for f in fps:
    try:
        print("Checking file: ", f)
        y, sr = librosa.load(f,duration=1)
        #print("File has length: ", len(y))
        #print("File has sample rate: ", sr)
    except Exception as e:
        print("Error in file: ", f)
        #print(e)