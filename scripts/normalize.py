import librosa
import glob
import numpy as np
import soundfile as sf

target_path="datasets/comparison_experiment/wav/**/*.wav"

# normalize all audio
for path in glob.glob(target_path):
    y, sr = librosa.load(path, sr=None)
    y=y/np.max(np.abs(y)+1e-10)
    sf.write(path.replace("/wav/","/normwav/"), y, sr)