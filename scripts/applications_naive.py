#%%
import librosa
import glob
import soundfile as sf
import numpy as np
fps = glob.glob("paper/applications_w_naive/**/*trn_data_duration=16*/training data.wav",recursive=True)

for fp in fps:
    print(fp)
    # load with librosa
    y, sr = librosa.load(fp,sr=None)

    # transpose by a fourth
    y_transposed = librosa.effects.pitch_shift(y, sr, n_steps=5)
    # save the result
    sf.write(fp.replace("training data.wav","transposed up a fourth naive.wav"), y_transposed, sr)
    
    # transpose down a fourth
    y_transposed = librosa.effects.pitch_shift(y, sr, n_steps=-5)
    # save the result
    sf.write(fp.replace("training data.wav","transposed down a fourth naive.wav"), y_transposed, sr)

    for key in "down 6 db", "up 6 db", "down 12 db", "up 12 db":
        # load corresponding file
        yl, sr = librosa.load(fp.replace("training data",f"loudness {key}"),sr=None)
  
        # adjust y to have the same rms as yl
        y_adjusted = y*np.sqrt(np.mean(yl**2)/np.mean(y**2))

        # print rms of y, yl and y adjusted
    
        # save the result
        sf.write(fp.replace("training data.wav",f"loudness {key} naive.wav"), y_adjusted, sr)


# %%
