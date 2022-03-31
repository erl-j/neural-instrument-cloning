
#%%
from constants import *
import matplotlib.pyplot as plt
import seaborn as sns

tst_data_provider=data.MultiTFRecordProvider(f"datasets/comparison_experiment/tfr/tst/*",sample_rate=SAMPLE_RATE)
tst_dataset=tst_data_provider.get_dataset(shuffle=False)

MAX_DISPLAY_SECONDS=32

tst_data_display=next(iter(tst_dataset.take(MAX_DISPLAY_SECONDS//CLIP_S).batch(MAX_DISPLAY_SECONDS//CLIP_S)))
#tst_data_display_wd=tf.data.Dataset.from_tensor_slices(join_and_window(tst_data_display,4,3)).batch(BATCH_SIZE)

# render f0 contours

#%%

print(tst_data_display.keys())

# extract contours
f0_contour=np.reshape(tst_data_display["f0_hz"],[-1])
loudness_contour=np.reshape(tst_data_display["loudness_db"],[-1])
f0confidence_contour=np.reshape(tst_data_display["f0_confidence"],[-1])

x=np.linspace(0,32,f0_contour.shape[0])



plt.figure(figsize=(20,10))

# change line colour


plt.subplot(3,1,1)
plt.plot(x,f0_contour,label="f0 contour (Hz)",color="black")
# label y axis
plt.ylabel("F0 (Hz)")
plt.xticks([])

plt.subplot(3,1,2)
plt.plot(x,f0confidence_contour,label="f0 confidence contour",color="black")
# label y axis
plt.ylabel("F0 confidence")
# hide x axis
plt.xticks([])

plt.subplot(3,1,3)
plt.plot(x,loudness_contour,label="loudness contour (db)",color="black")
# label y axis
plt.ylabel("Loudness (db)")


#label x axis
plt.xlabel("Time (seconds)")

# remove space between subplots
plt.subplots_adjust(hspace=0)





plt.show()

# %%
