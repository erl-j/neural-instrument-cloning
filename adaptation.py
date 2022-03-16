# %%
from constants import *

# %%
def join_batch(batch):
    for key in batch.keys():
        assert len(batch[key].shape)<3
        if len(batch[key].shape)==2:
            batch[key]=tf.reshape(batch[key],(1,-1))
    return batch

def window_signal(a,window_len,hop_len):
     assert(a.shape[0]==1)
     windows=[]
     start_frame=0
     while True:
        windows.append(a[:,start_frame:start_frame+window_len,...])
        start_frame+=hop_len
        if start_frame > a.shape[1]-window_len:
            break
     return tf.concat(windows,axis=0)

def window_sample(instance,win_s,hop_s):
    instance["audio"]=window_signal(instance["audio"],win_s*SAMPLE_RATE,hop_s*SAMPLE_RATE)
    for key in ["f0_hz","loudness_db","f0_confidence"]:
        instance[key]=window_signal(instance[key],win_s*FT_FRAME_RATE,hop_s*FT_FRAME_RATE)
    instance["instrument"]=tf.repeat(instance["instrument"][0],(instance["audio"].shape[0]))
    instance["instrument_idx"]=tf.repeat(instance["instrument_idx"][0],(instance["audio"].shape[0]))
    #for key,item in instance.items():
    #    assert(len(item.shape)<2 or item.shape[0]>1)
    return instance

def join_and_window(instance,win_s=4,hop_s=1):
    instance=copy.deepcopy(instance)
    return window_sample(join_batch(instance),win_s,hop_s)

def rf2cf(row_form):
    return {k:[s[k] for s in row_form] for k in row_form[0].keys()}

def playback_and_save(x,fn,DEMO_PATH):
    print(fn)
    play(x)
    os.makedirs(DEMO_PATH,exist_ok=True)
    path=DEMO_PATH+f"{fn}.wav"
    soundfile.write(path,x,SAMPLE_RATE)

def stitch(audios):
  RENDER_OVERLAP_S=1
  out=np.zeros(N_SAMPLES*len(audios))
  tail_taper= np.concatenate([np.ones(RENDER_OVERLAP_S*SAMPLE_RATE),np.ones(N_SAMPLES-(RENDER_OVERLAP_S*SAMPLE_RATE*2)),np.linspace(1,0,RENDER_OVERLAP_S*SAMPLE_RATE)])
  head_taper= np.concatenate([np.linspace(0,1,RENDER_OVERLAP_S*SAMPLE_RATE),np.ones(N_SAMPLES-(RENDER_OVERLAP_S*SAMPLE_RATE*2)),np.ones(RENDER_OVERLAP_S*SAMPLE_RATE)])
  bi_taper = tail_taper*head_taper

  if len(audios)>2:
    out[:N_SAMPLES]=audios[0]*tail_taper
    for ai,a in enumerate(audios[1:-1]):
        out[(ai+1)*(N_SAMPLES-RENDER_OVERLAP_S*SAMPLE_RATE):(ai+2)*(N_SAMPLES-RENDER_OVERLAP_S*SAMPLE_RATE)+RENDER_OVERLAP_S*SAMPLE_RATE]+=a*bi_taper
    out[(ai+2)*(N_SAMPLES-RENDER_OVERLAP_S*SAMPLE_RATE):(ai+3)*(N_SAMPLES-RENDER_OVERLAP_S*SAMPLE_RATE)+RENDER_OVERLAP_S*SAMPLE_RATE]+=audios[-1]*head_taper
    out=out[:(ai+3)*(N_SAMPLES-RENDER_OVERLAP_S*SAMPLE_RATE)+RENDER_OVERLAP_S*SAMPLE_RATE]
  else:
      out[:N_SAMPLES]=audios[0]
  return out


def render_example(dataset,model, transform_key=None,transform_fn=lambda x:x):
    audio=[]
    for batch in dataset:
        if transform_key != None:
            batch=copy.deepcopy(batch)
            batch[transform_key]=transform_fn(batch[transform_key])
        output = model(batch)
        audio.append(output["audio_synth"])
    return stitch(audio)

# %%

# LOAD MODEL FOR FINETUNING

def get_finetuning_model(full_ir_duration,free_ir_duration,checkpoint_path):
    # load model
    test_model=shared_model.get_model(SAMPLE_RATE,CLIP_S,FT_FRAME_RATE,Z_SIZE,N_INSTRUMENTS,IR_DURATION,BIDIRECTIONAL,USE_F0_CONFIDENCE,N_HARMONICS,N_NOISE_MAGNITUDES,losses=[])
    # load model weights       

    DEMO_IR_SAMPLES=int(full_ir_duration*SAMPLE_RATE)

    test_model.set_is_shared_trainable(True)

    if checkpoint_path!=None:
        test_model.restore(checkpoint_path)
        #test_model.load_weights(checkpoint_path)

    test_model.instrument_weight_metadata["ir"]["initializer"]=lambda batch_size: tf.zeros([batch_size,int(full_ir_duration*SAMPLE_RATE)])

    if free_ir_duration<full_ir_duration:

        er_samples=int(free_ir_duration*SAMPLE_RATE)

        er_amp=np.ones((er_samples))
        er_amp[er_samples//2:er_samples]=np.linspace(1,0,er_samples//2)

        frame_rate=250
        n_filter_bands=100
        n_frames=int(frame_rate*DEMO_IR_DURATION)

        ir_fn=ddsp.synths.FilteredNoise(n_samples=DEMO_IR_SAMPLES,
                                           window_size=750,
                                           scale_fn=tf.nn.relu,
                                           initial_bias=1e-10)

        def processing_fn(batched_feature):

            batch_size=batched_feature.shape[0]
            er_ir = tf.nn.tanh(batched_feature[:,:er_samples])

            er_amp=np.ones(DEMO_IR_SAMPLES)
            er_amp[er_samples//2:er_samples]=np.linspace(1,0,er_samples//2)
            er_amp[er_samples:]=0

            er_amp = er_amp[None,:]
            fn_amp= 1-er_amp

            fn_mags=tf.reshape(batched_feature[:,er_samples:],[batch_size,n_frames,n_filter_bands])
            fn_ir=ir_fn(fn_mags)

            ir=fn_ir*fn_amp+tf.pad(er_ir,[[0,0],[0,int(full_ir_duration*SAMPLE_RATE)-er_samples]])*er_amp

            #ir = ddsp.core.fft_convolve( fn_ir,er_ir, padding='same', delay_compensation=0)
            return ir

        test_model.instrument_weight_metadata["ir"]["processing"]=processing_fn
        test_model.instrument_weight_metadata["ir"]["initializer"]=lambda batch_size: tf.zeros([batch_size,er_samples+n_frames*n_filter_bands])
        test_model.instrument_weight_metadata["wet_gain"]["initializer"]=lambda batch_size: tf.ones([batch_size,1])*0.5

    test_model.initialize_instrument_weights()
    test_model.set_is_shared_trainable(True)

    #TMP_CHECKPOINT_PATH="./artefacts/tmp_checkpoint"
    #test_model.save_weights(TMP_CHECKPOINT_PATH)
    
    #test_model.set_is_shared_trainable(True)
    #test_model.load_weights(TMP_CHECKPOINT_PATH)
    #test_model.initialize_instrument_weights()
    
    return test_model

# %%
# HP TUNING ON OTHER SET?

# constants

TRAIN_DATA_DURATIONS = [4*(2**i) for i in range(7)]
#TRAIN_DATA_DURATIONS=[16]
BATCH_SIZE=1
DEMO_IR_DURATION=1
MAX_DISPLAY_SECONDS=32

# comparison tst
trn_data_provider=data.MultiTFRecordProvider(f"datasets/comparison_experiment/tfr/dev/*",sample_rate=SAMPLE_RATE)
trn_dataset= trn_data_provider.get_dataset(shuffle=False)

tst_data_provider=data.MultiTFRecordProvider(f"datasets/comparison_experiment/tfr/tst/*",sample_rate=SAMPLE_RATE)
tst_dataset=tst_data_provider.get_dataset(shuffle=False)

tst_data_display=next(iter(tst_dataset.take(MAX_DISPLAY_SECONDS//CLIP_S).batch(MAX_DISPLAY_SECONDS//CLIP_S)))
tst_data_display_wd=tf.data.Dataset.from_tensor_slices(join_and_window(tst_data_display,4,3)).batch(BATCH_SIZE)

# check that no windowing is present
# trn_data_test=next(iter(trn_dataset.take(3).batch(3)))
# playback_and_save(tf.reshape(trn_data_test["audio"],[-1]),"trn data test","./comparison_experiments/")

# tst_data_test=next(iter(tst_dataset.take(3).batch(3)))
# playback_and_save(tf.reshape(tst_data_test["audio"],[-1]),"tst data test","./comparison_experiments/")


# set adaptation strategy
pretrained_checkpoint_path="./artefacts/training/Saxophone/ckpt-380000"
finetune_whole=False
free_ir_duration=0.2
ir_duration=1

for train_data_duration in TRAIN_DATA_DURATIONS:

    model=get_finetuning_model(ir_duration,free_ir_duration,pretrained_checkpoint_path)

    # load correct amount of training data and window it two ways
    trn_clips=train_data_duration//CLIP_S
    trn_data=next(iter(trn_dataset.take(trn_clips).batch(trn_clips)))

    trn_data_batched=tf.data.Dataset.from_tensor_slices(join_and_window(trn_data,4,1)).batch(BATCH_SIZE)
    n_batches=len(list(trn_data_batched))
    
    # set learning rate and n epochs based on adaptation strategy
    if pretrained_checkpoint_path!=None:
        model.set_is_shared_trainable(finetune_whole)
        if finetune_whole:
            lr=3e-5
            n_epochs=100
        if not finetune_whole:
            lr=2e-3
            n_epochs=100
    else:
        model.set_is_shared_trainable(True)
        lr=1e-4
        n_epochs=100

    OUTPUT_PATH=f"comparison_experiment/{pretrained_checkpoint_path}_trn_data_duration={train_data_duration}_finetunewhole={finetune_whole}_free_ir={free_ir_duration}/"

    trn_log_dir = OUTPUT_PATH + '/trn'
    tst_log_dir = OUTPUT_PATH + '/tst'
    trn_summary_writer = tf.summary.create_file_writer(trn_log_dir)
    tst_summary_writer = tf.summary.create_file_writer(tst_log_dir)

    trn_data_display=next(iter(trn_dataset.take(min(MAX_DISPLAY_SECONDS,train_data_duration)//CLIP_S).batch(min(MAX_DISPLAY_SECONDS,train_data_duration)//CLIP_S)))
    trn_data_display_wd=tf.data.Dataset.from_tensor_slices(join_and_window(trn_data_display,4,3)).batch(BATCH_SIZE)

    summary_interval = 10

    # batch tst data
    tst_data_batched=tst_dataset.batch(BATCH_SIZE)

    # set up optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    trn_losses=[]
    tst_losses=[]

    for epoch_count in tqdm.tqdm(range(n_epochs)):
        trn_data_batched=trn_data_batched.shuffle(100000)

        epoch_loss=0
        batch_counter=0

        for trn_batch in trn_data_batched:
            with tf.GradientTape() as tape:
                output = model(trn_batch)
                loss_value=spectral_loss(trn_batch["audio"],output["audio_synth"])
                epoch_loss+=loss_value.numpy()
                batch_counter+=1
                gradients = tape.gradient(loss_value, model.trainable_weights)
            optimizer.apply_gradients(zip(gradients, model.trainable_weights))

        trn_losses.append(epoch_loss/batch_counter)

        with trn_summary_writer.as_default():
            tf.summary.scalar('loss', epoch_loss/batch_counter, step=epoch_count)
     
        if epoch_count%summary_interval==0 or epoch_count==n_epochs-1:

            with trn_summary_writer.as_default():
                tf.summary.audio("target",tf.reshape(trn_batch["audio"],[-1])[None,...,None],SAMPLE_RATE,step=epoch_count)
                tf.summary.audio("estimate",tf.reshape(output["audio_synth"],[-1])[None,...,None],SAMPLE_RATE,step=epoch_count)

            tst_epoch_loss=0
            tst_batch_counter=0
            for tst_batch in tst_data_batched:
                tst_output=model(tst_batch)
                loss_value=spectral_loss(tst_batch["audio"],tst_output["audio_synth"])   
                tst_epoch_loss+=loss_value.numpy()
                tst_batch_counter+=1
            tst_losses.append(tst_epoch_loss/tst_batch_counter)
            with tst_summary_writer.as_default():
                tf.summary.scalar('loss', tst_epoch_loss/tst_batch_counter, step=epoch_count)
                tf.summary.audio("target",tf.reshape(tst_batch["audio"],[-1])[None,...,None],SAMPLE_RATE,step=epoch_count)
                tf.summary.audio("estimate",tf.reshape(tst_output["audio_synth"],[-1])[None,...,None],SAMPLE_RATE,step=epoch_count)
        epoch_count+=1

    # RENDER AUDIO EXAMPLES
    playback_and_save(tf.reshape(trn_data_display["audio"],[-1]),"training data",OUTPUT_PATH)
    
    # First trn data
    playback_and_save(render_example(trn_data_display_wd,model),"training estimate",OUTPUT_PATH)
    playback_and_save(render_example(trn_data_display_wd,model,"f0_hz",lambda x:x*(3/4)),"transposed down a fourth",OUTPUT_PATH)
    playback_and_save(render_example(trn_data_display_wd,model,"f0_hz",lambda x:x*(4/3)),"transposed up a fourth",OUTPUT_PATH)
    playback_and_save(render_example(trn_data_display_wd,model,"loudness_db",lambda x:x-12),"loudness down 12 db",OUTPUT_PATH)
    playback_and_save(render_example(trn_data_display_wd,model,"loudness_db",lambda x:x-6),"loudness down 6 db",OUTPUT_PATH)
    playback_and_save(render_example(trn_data_display_wd,model,"loudness_db",lambda x:x+6),"loudness up 6 db",OUTPUT_PATH)
    playback_and_save(render_example(trn_data_display_wd,model,"loudness_db",lambda x:x+12),"loudness up 12 db",OUTPUT_PATH)
    playback_and_save(render_example(trn_data_display_wd,model,"f0_confidence",lambda x:x*0.8),"pitch confidence * 0.8",OUTPUT_PATH)
    playback_and_save(render_example(trn_data_display_wd,model,"f0_confidence",lambda x:x*0.5),"pitch confidence * 0.5",OUTPUT_PATH)

    # save tst data
    playback_and_save(tf.reshape(tst_data_display["audio"],[-1]),"unseen target",OUTPUT_PATH)
     # transform data so that the clips overlap
    playback_and_save(render_example(tst_data_display_wd,model),"unseen estimate",OUTPUT_PATH)
    
asd













# %%
# render other examples

# %%
USE_NSYNTH=False
#INSTRUMENT_FAMILY="**_WHITHOUT_SAX"
INSTRUMENT_FAMILY="Saxophone"

# %%
N_FIT_ITERATIONS= 100 if TRAIN_SHARED else int(100*(16/N_FIT_SECONDS))
VAL_LR=3e-5 if TRAIN_SHARED else 2e-3
DEMO_IR_DURATION=1

BATCH_SIZE=1

# OUTPUT SETTINGS
VERSION=2
DEMO_PATH=f"artefacts/demos/{INSTRUMENT_FAMILY}_{VERSION}_{N_FIT_SECONDS}_{'train_shared' if TRAIN_SHARED else ''}/"

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
    
    trn_data_provider=data.MultiTFRecordProvider(trn_path,sample_rate=SAMPLE_RATE)
    val_data_provider=data.MultiTFRecordProvider(val_path,sample_rate=SAMPLE_RATE)
    trn_dataset= trn_data_provider.get_dataset()
    val_dataset=val_data_provider.get_dataset(shuffle=False)
    
# remove some samples if number of recordings greater than model capacity
trn_dataset = trn_dataset.filter(lambda x: int(x["instrument_idx"])<N_INSTRUMENTS)

# %%
# render demo audio 

# FINETUNING HPARAMS
TRAIN_SHARED=False
N_FIT_SECONDS = 16
FREE_IR_DURATION=0.2
n_fit_windows=int(N_FIT_SECONDS/CLIP_S)

# group by instrument id
val_dataset_by_instrument=pydash.collections.group_by(list(val_dataset),lambda x: str(x["instrument"].numpy()))
val_dataset_by_instrument = {k:v for k,v in val_dataset_by_instrument.items()}

for ii,instrument_set in enumerate(list(val_dataset_by_instrument.values())): 
    print(f"instrument nr {ii}")

    # first, we load up a fresh model
    test_model=get_finetuning_model(DEMO_IR_DURATION,FREE_IR_DURATION,checkpoint_path)

    #  first, separate the finetuning data from the test data
    fit_data_samples=instrument_set[:n_fit_windows]
    
    # Use second to last 4 windows (16 s) as test data
    test_data_samples=instrument_set[len(instrument_set)-5:-1]
    
    assert (len(instrument_set)-5>=n_fit_windows)

    # convert to column form
    fit_data = rf2cf(fit_data_samples)

    # get one batch for fitting
    fit_batch= next(iter(tf.data.Dataset.from_tensor_slices(fit_data).batch(len(list(fit_data)))))
    
    playback_and_save(tf.reshape(fit_data["audio"],[-1]),"training data",DEMO_PATH)

    # transform data so that the clips overlap
    fit_batch=join_and_window(fit_batch,4,1)
    fit_data=tf.data.Dataset.from_tensor_slices(fit_batch)
    fit_batched=fit_data.batch(BATCH_SIZE)

    # prepare test data
    test_data = rf2cf(test_data_samples)
    test_batched= tf.data.Dataset.from_tensor_slices(test_data).batch(BATCH_SIZE)

    fit_losses=[]
    tst_losses=[]

    # set up optimizer
    val_optimizer = tf.keras.optimizers.Adam(learning_rate=VAL_LR)

    for i in tqdm.tqdm(range(N_FIT_ITERATIONS)):
        fit_batched_shuffled=fit_batched.shuffle(100)
        epoch_loss=0
        batch_counter=0
        test_epoch_loss=0
        test_batch_counter=0

        for fit_batch in fit_batched_shuffled:
            with tf.GradientTape() as tape:
              test_model.set_is_shared_trainable(TRAIN_SHARED)
              output = test_model(fit_batch)
              loss_value=spectral_loss(fit_batch["audio"],output["audio_synth"])
              epoch_loss+=loss_value.numpy()
              batch_counter+=1
              gradients = tape.gradient(loss_value, test_model.trainable_weights)
            val_optimizer.apply_gradients(zip(gradients, test_model.trainable_weights))
        fit_losses.append(epoch_loss/batch_counter)

        for test_batch in test_batched:
            test_model.set_is_shared_trainable(False)
            test_output=test_model(test_batch)
            loss_value=spectral_loss(test_batch["audio"],test_output["audio_synth"])   
            test_epoch_loss+=loss_value.numpy()
            test_batch_counter+=1
        tst_losses.append(test_epoch_loss/test_batch_counter)

        if i%10==0:

            print("target")        
            play(tf.reshape(fit_batch["audio"],(-1)))

            print("estimate")     
            play(tf.reshape(output['audio_synth'],(-1)))
            # loss plot
            plt.plot(tst_losses,label="tst")
            plt.plot(fit_losses,label="trn")
            plt.yscale("log")
            plt.legend()
            plt.show()

            ir=output['ir'][0]

            plt.plot(ir)
            plt.show()

            play(tf.reshape(ir,(-1)))

            plt.imshow(ddsp.spectral_ops.compute_mel(ir).numpy().T,aspect="auto",origin="lower")
            plt.show()

            print(f"wet gain: {output['wet_gain']['controls']['gain_scaled']}")
            print(f"dry gain: {output['dry_gain']['controls']['gain_scaled']}")

    plt.plot(tst_losses,label="tst")
    plt.plot(fit_losses,label="trn")
    plt.yscale("log")
    plt.legend()
    plt.show()

    # RENDER AUDIO EXAMPLES
    
    # Transform fit data with 3 second skips instead
    fit_data = rf2cf(fit_data_samples)

    # get one batch for fitting
    fit_batch= next(iter(tf.data.Dataset.from_tensor_slices(fit_data).batch(len(list(fit_data)))))
    
    #playback_and_save(tf.reshape(fit_data["audio"],[-1]),"training data",DEMO_PATH)

    # transform data so that the clips overlap
    fit_batch=join_and_window(fit_batch,4,3)
    fit_data=tf.data.Dataset.from_tensor_slices(fit_batch)
    fit_batched=fit_data.batch(BATCH_SIZE)

    # First fit data
    
    playback_and_save(render_example(fit_batched,test_model),"training estimate",DEMO_PATH)
    playback_and_save(render_example(fit_batched,test_model,"f0_hz",lambda x:x*(3/4)),"transposed down a fourth",DEMO_PATH)
    playback_and_save(render_example(fit_batched,test_model,"f0_hz",lambda x:x*(4/3)),"transposed up a fourth",DEMO_PATH)
    playback_and_save(render_example(fit_batched,test_model,"loudness_db",lambda x:x-12),"loudness down 12 db",DEMO_PATH)
    playback_and_save(render_example(fit_batched,test_model,"loudness_db",lambda x:x-6),"loudness down 6 db",DEMO_PATH)
    playback_and_save(render_example(fit_batched,test_model,"loudness_db",lambda x:x+6),"loudness up 6 db",DEMO_PATH)
    playback_and_save(render_example(fit_batched,test_model,"loudness_db",lambda x:x+12),"loudness up 12 db",DEMO_PATH)
    playback_and_save(render_example(fit_batched,test_model,"f0_confidence",lambda x:x*0.8),"pitch confidence * 0.8",DEMO_PATH)
    playback_and_save(render_example(fit_batched,test_model,"f0_confidence",lambda x:x*0.5),"pitch confidence * 0.5",DEMO_PATH)

    # Next test data
    
    # we need to apply windowing to the signal before rendering
    
    test_data = rf2cf(test_data_samples)
    test_batch= next(iter(tf.data.Dataset.from_tensor_slices(test_data).batch(len(list(test_data)))))
    # save test data
    playback_and_save(tf.reshape(test_data["audio"],[-1]),"unseen data",DEMO_PATH)
    # transform data so that the clips overlap
    test_batch=join_and_window(test_batch,4,3)
    test_data=tf.data.Dataset.from_tensor_slices(test_batch)
    test_batched=test_data.batch(BATCH_SIZE)

    playback_and_save(render_example(test_batched,test_model),"unseen estimate",DEMO_PATH)


