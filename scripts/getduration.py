import os
import librosa
target_dir="datasets/AIR/wav/dev/Saxophone"

# how maby wav files are in directory
print(len([filename for filename in os.listdir(target_dir) if filename.endswith(".wav")]))
s=0
# compute the duration of a all audio files in a directory
for filename in os.listdir(target_dir):
    if filename.endswith(".wav"):
        y,sr = librosa.load(f"{target_dir}/{filename}",sr=16000)
        print(f"{filename} has duration {len(y)/sr}")
        s+=len(y)/sr


print(s)
