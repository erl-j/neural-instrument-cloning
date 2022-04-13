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
# %%
# LOAD MODEL FOR FINETUNING

def get_finetuning_model(full_ir_duration,free_ir_duration,checkpoint_path,reset_parts=True,use_loss=False,use_f0_confidence=True,use_id_activation_on_free_reverb=False):
    # load model
    test_model=model.get_model(SAMPLE_RATE,CLIP_S,FT_FRAME_RATE,Z_SIZE,N_INSTRUMENTS,IR_DURATION,BIDIRECTIONAL,USE_F0_CONFIDENCE=use_f0_confidence,N_HARMONICS=N_HARMONICS,N_NOISE_MAGNITUDES=N_NOISE_MAGNITUDES,losses=[spectral_loss] if use_loss else [])
    # load model weights       

    DEMO_IR_SAMPLES=int(full_ir_duration*SAMPLE_RATE)

    test_model.set_is_shared_trainable(True)

    if checkpoint_path!=None:
        if "48k_bidir" in checkpoint_path:
            test_model.load_weights(checkpoint_path)
        else:
            test_model.restore(checkpoint_path)

    if reset_parts:

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


                if free_ir_duration>0.0:
                    ir=fn_ir*fn_amp+tf.pad(er_ir,[[0,0],[0,int(full_ir_duration*SAMPLE_RATE)-er_samples]])*er_amp
                else:
                    ir=fn_ir

                #ir = ddsp.core.fft_convolve( fn_ir,er_ir, padding='same', delay_compensation=0)
                return ir

            test_model.instrument_weight_metadata["ir"]["processing"]=processing_fn
            test_model.instrument_weight_metadata["ir"]["initializer"]=lambda batch_size: tf.zeros([batch_size,er_samples+n_frames*n_filter_bands])
            test_model.instrument_weight_metadata["wet_gain"]["initializer"]=lambda batch_size: tf.ones([batch_size,1])*0.5

        elif use_id_activation_on_free_reverb:
            print("USING ID ACTIVATION ON FREE REVERB")
            test_model.instrument_weight_metadata["ir"]["processing"]=lambda x: x
            test_model.instrument_weight_metadata["wet_gain"]["initializer"]=lambda batch_size: tf.ones([batch_size,1])*0.5

        test_model.initialize_instrument_weights()
        test_model.set_is_shared_trainable(True)

    
    
    return test_model

# %%

parser=argparse.ArgumentParser()
parser.add_argument('--pretrained_checkpoint_path',default=None)
parser.add_argument('--cloning_dataset_path',default=None)
parser.add_argument('--finetune_whole', action='store_true')
parser.add_argument('--id_activation_on_free_reverb', action='store_true')

args=parser.parse_args()

BATCH_SIZE=1
DEMO_IR_DURATION=1
MAX_DISPLAY_SECONDS=32

USE_F0_CONFIDENCE=True

# set adaptation strategy
#pretrained_checkpoint_path="./artefacts/training/**_WITHOUT_SAX/ckpt-558000" # None #
#pretrained_checkpoint_path="./artefacts/training/Saxophone/ckpt-380000"
 
pretrained_checkpoint_path=args.pretrained_checkpoint_path
finetune_whole=args.finetune_whole

free_ir_duration=0.2
ir_duration=1
summary_interval = 10

USE_ID_ACTIVATION_ON_FREE_REVERB=args.id_activation_on_free_reverb

USE_PRETRAINING_INSTRUMENTS=True
TRAIN_DATA_DURATIONS = [4] if USE_PRETRAINING_INSTRUMENTS else [64,128]
instrument_idxs=range(200) if USE_PRETRAINING_INSTRUMENTS else [0]

cloning_dataset_path=args.cloning_dataset_path

if cloning_dataset_path==None:
    # comparison tst
    trn_data_provider=data.MultiTFRecordProvider(f"datasets/comparison_experiment/tfr/dev/*",sample_rate=SAMPLE_RATE)
    trn_dataset= trn_data_provider.get_dataset(shuffle=False)
    tst_data_provider=data.MultiTFRecordProvider(f"datasets/comparison_experiment/tfr/tst/*",sample_rate=SAMPLE_RATE)
    tst_dataset=tst_data_provider.get_dataset(shuffle=False)
else:
    data_provider=data.MultiTFRecordProvider(cloning_dataset_path,sample_rate=SAMPLE_RATE)
    dataset=data_provider.get_dataset(shuffle=False)    
    print(len(list(dataset)))

    set_clips=32//CLIP_S
    trn_dataset=dataset.take(set_clips)
    tst_dataset=dataset.skip(set_clips).take(set_clips)

    transfer_data_provider=data.MultiTFRecordProvider(f"datasets/comparison_experiment/tfr/tst/*",sample_rate=SAMPLE_RATE)
    transfer_dataset=transfer_data_provider.get_dataset(shuffle=False)
    transfer_data_display=next(iter(transfer_dataset.take(MAX_DISPLAY_SECONDS//CLIP_S).batch(MAX_DISPLAY_SECONDS//CLIP_S)))
    transfer_data_display_wd=tf.data.Dataset.from_tensor_slices(join_and_window(transfer_data_display,4,3)).batch(BATCH_SIZE)


tst_data_display=next(iter(tst_dataset.take(MAX_DISPLAY_SECONDS//CLIP_S).batch(MAX_DISPLAY_SECONDS//CLIP_S)))
tst_data_display_wd=tf.data.Dataset.from_tensor_slices(join_and_window(tst_data_display,4,3)).batch(BATCH_SIZE)

for instrument_idx in instrument_idxs:
    def render_example(dataset,model, transform_key=None,transform_fn=lambda x:x):
        audio=[]
        for batch in dataset:
            if USE_PRETRAINING_INSTRUMENTS:
                batch["instrument_idx"]=batch["instrument_idx"]*0+instrument_idx
            if transform_key != None:
                if isinstance(transform_key, list):
                    for key_idx,key in enumerate(transform_key):
                        batch=copy.deepcopy(batch)
                        batch[key]=transform_fn[key_idx](batch[key])
                else:
                    batch=copy.deepcopy(batch)
                    batch[transform_key]=transform_fn(batch[transform_key])
           
            output = model(batch)
            audio.append(output["audio_synth"])
        return stitch(audio)
        
    for train_data_duration in TRAIN_DATA_DURATIONS:
        model=get_finetuning_model(ir_duration,free_ir_duration,pretrained_checkpoint_path,reset_parts=not USE_PRETRAINING_INSTRUMENTS,use_f0_confidence=USE_F0_CONFIDENCE,use_id_activation_on_free_reverb=USE_ID_ACTIVATION_ON_FREE_REVERB)

        # load correct amount of training data and window it two ways
        trn_clips=train_data_duration//CLIP_S
        trn_data=next(iter(trn_dataset.take(trn_clips).batch(trn_clips)))

        trn_data_batched=tf.data.Dataset.from_tensor_slices(join_and_window(trn_data,4,1)).batch(BATCH_SIZE)
        n_batches=len(list(trn_data_batched))
        
        # set learning rate and n epochs based on adaptation strategy

        if USE_PRETRAINING_INSTRUMENTS:
            lr=0
            n_epochs=1
        else:
            # starting from pretrained
            if pretrained_checkpoint_path!=None:
                model.set_is_shared_trainable(finetune_whole)
                if finetune_whole:
                    lr=3e-5
                    n_epochs=100
                if not finetune_whole:
                    lr=3e-3
                    n_epochs=300 if train_data_duration<32 else 100
            # no pretraining
            else:
                print("no pretraining")
                model.set_is_shared_trainable(True)
                lr=3e-5
                train_data_duration_2_epochs={4:2900,8:1000,16:1000,32:1000,64:1000,128:300,256:300}
                n_epochs=train_data_duration_2_epochs[train_data_duration]

        print(f" train duration = {train_data_duration} lr={lr} n_epochs={n_epochs}")
        OUTPUT_PATH=f"paper/comparison_experiment/nosaxnearest/{cloning_dataset_path}/use_f0_confidence={USE_F0_CONFIDENCE}_nr_{instrument_idx}_{pretrained_checkpoint_path}_trn_data_duration={train_data_duration}_finetunewhole={finetune_whole}_free_ir={free_ir_duration}_lr={lr}/"

        trn_log_dir = OUTPUT_PATH + '/trn'
        tst_log_dir = OUTPUT_PATH + '/tst'
        trn_summary_writer = tf.summary.create_file_writer(trn_log_dir)
        tst_summary_writer = tf.summary.create_file_writer(tst_log_dir)

        # load correct amount of training data and window it two
        trn_clips=train_data_duration//CLIP_S
        trn_data=next(iter(trn_dataset.take(trn_clips).batch(trn_clips)))

        trn_data_batched=tf.data.Dataset.from_tensor_slices(join_and_window(trn_data,4,1)).batch(BATCH_SIZE)
        n_batches=len(list(trn_data_batched))

        trn_data_display=next(iter(trn_dataset.take(min(MAX_DISPLAY_SECONDS,train_data_duration)//CLIP_S).batch(min(MAX_DISPLAY_SECONDS,train_data_duration)//CLIP_S)))
        trn_data_display_wd=tf.data.Dataset.from_tensor_slices(join_and_window(trn_data_display,4,3)).batch(BATCH_SIZE)

        # batch tst data
        tst_data_batched=tst_dataset.batch(BATCH_SIZE)

        # set up optimizer
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

        trn_losses=[]
        tst_losses=[]

        latest_test_loss=None
        for epoch_count in tqdm.tqdm(range(n_epochs)):
            trn_data_batched=trn_data_batched.shuffle(100000)

            epoch_loss=0
            batch_counter=0

            for trn_batch in trn_data_batched:
                with tf.GradientTape() as tape:
                    if USE_PRETRAINING_INSTRUMENTS:
                        trn_batch["instrument_idx"]=trn_batch["instrument_idx"]*0+instrument_idx
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
                    if USE_PRETRAINING_INSTRUMENTS:
                        tst_batch["instrument_idx"]=tst_batch["instrument_idx"]*0+instrument_idx
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

        if cloning_dataset_path is not None:

            # transform data so that the clips overlap
            playback_and_save(render_example(transfer_data_display_wd,model),"transfer",OUTPUT_PATH)

            # compute statistics over the transfer data
            f0c_min=np.min(transfer_data_display["f0_confidence"])
            f0c_max=np.max(transfer_data_display["f0_confidence"])
            loudness_min=np.min(transfer_data_display["loudness_db"])
            loudness_max=np.max(transfer_data_display["loudness_db"])

            # compute statistics over the training data
            f0c_min_trn=np.min(trn_data_display["f0_confidence"])
            f0c_max_trn=np.max(trn_data_display["f0_confidence"])
            loudness_min_trn=np.min(trn_data_display["loudness_db"])
            loudness_max_trn=np.max(trn_data_display["loudness_db"])

            # render adjusted transfer data 
            # first f0c
            playback_and_save(render_example(transfer_data_display_wd,model,"loudness_db",lambda x: x-loudness_max+loudness_max_trn),"transfer adjusted loudness",OUTPUT_PATH)
            playback_and_save(render_example(transfer_data_display_wd,model,"f0_confidence",lambda x: f0c_min_trn+((x-f0c_min)/(f0c_max-f0c_min))*(f0c_max_trn-f0c_min_trn)),"transfer adjusted f0c",OUTPUT_PATH)

            playback_and_save(render_example(transfer_data_display_wd,model,["f0_confidence","loudness_db"],[lambda x: f0c_min_trn+((x-f0c_min)/(f0c_max-f0c_min))*(f0c_max_trn-f0c_min_trn),lambda x: x-loudness_max+loudness_max_trn]),"transfer adjusted f0c & loudness",OUTPUT_PATH)
        










