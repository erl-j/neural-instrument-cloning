# %%
from constants import *

# %%
USE_NSYNTH=False
INSTRUMENT_FAMILY="**"
#INSTRUMENT_FAMILY="Saxophone"

#INSTRUMENT_FAMILY="Trombone"
# %%
if USE_NSYNTH:
    tfds.load("nsynth/gansynth_subset.f0_and_loudness",split="train", try_gcs=False,download=True) 
    trn_data_provider = data.CustomNSynthTfds(data_dir="/root/tensorflow_datasets/",split="train")
    tfds.load("nsynth/gansynth_subset.f0_and_loudness",split="valid", try_gcs=False,download=True) 
    val_data_provider = data.CustomNSynthTfds(data_dir="/root/tensorflow_datasets/",split="valid")
    def crepe_is_certain(x):
        is_playing = tf.cast(x["loudness_db"]>-100.0,dtype=tf.float32)
        average_certainty=tf.reduce_sum(x["f0_confidence"]*is_playing)/tf.reduce_sum(is_playing)
        return average_certainty
    def preprocess_dataset(dataset):
        if INSTRUMENT_FAMILY!="all":
            dataset=dataset.filter(lambda x: x["instrument_family"]==INSTRUMENT_FAMILY)
        return dataset
    trn_dataset = preprocess_dataset(trn_data_provider.get_dataset())
    val_dataset = preprocess_dataset(val_data_provider.get_dataset())

else:
    
    trn_path=f"datasets/AIR/tfr/dev/{INSTRUMENT_FAMILY}/*"
    val_path=f"datasets/AIR/tfr/tst/{INSTRUMENT_FAMILY}/*"
    
    if INSTRUMENT_FAMILY=="**_WHITHOUT_SAX":
        print("without_sax")
        trn_path=f"datasets/AIRnoSax/tfr/dev/**/*"
        val_path=f"datasets/AIRnoSax/tfr/tst/**/*"
    
    trn_data_provider=data.MultiTFRecordProvider(trn_path,sample_rate=SAMPLE_RATE,n_max_instruments=N_INSTRUMENTS)
    val_data_provider=data.MultiTFRecordProvider(val_path,sample_rate=SAMPLE_RATE,n_max_instruments=N_INSTRUMENTS)
    trn_dataset= trn_data_provider.get_dataset()
    val_dataset=val_data_provider.get_dataset(shuffle=False)
    
# remove some samples if number of recordings greater than model capacity


# %%
checkpoint_path=f"checkpoints/48k_{'bidir' if BIDIRECTIONAL else 'unidir'}_z{Z_SIZE}_conv_family_{INSTRUMENT_FAMILY}{'_f0c' if USE_F0_CONFIDENCE else ''}"
training_savedir=f"./artefacts/training/{INSTRUMENT_FAMILY}"

# %%
# ddsp style training


strategy =  tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())

with strategy.scope():
    model=model.get_model(SAMPLE_RATE,CLIP_S,FT_FRAME_RATE,Z_SIZE,N_INSTRUMENTS,IR_DURATION,BIDIRECTIONAL,USE_F0_CONFIDENCE,N_HARMONICS,N_NOISE_MAGNITUDES,losses=[spectral_loss])
    model.set_is_shared_trainable(True)
    trainer=ddsp.training.trainers.Trainer(
                model,
                strategy,
                checkpoints_to_keep=10,
                lr_decay_steps=10000,
                learning_rate=1e-4,
                lr_decay_rate=0.98,
                grad_clip_norm=100000.0)

ddsp.training.train_util.train(
        trn_data_provider,
        trainer,
        batch_size=6,
        num_steps=1000000,
        steps_per_summary=1000,
        steps_per_save=1000,
        save_dir=training_savedir,
        restore_dir=training_savedir,
        early_stop_loss_value=None,
        report_loss_to_hypertune=False)
