from typing import Dict, Text
import ddsp
import tensorflow as tf


def get_model(SAMPLE_RATE,CLIP_S,FT_FRAME_RATE,Z_SIZE,N_INSTRUMENTS,IR_DURATION,BIDIRECTIONAL,USE_F0_CONFIDENCE,N_HARMONICS,N_NOISE_MAGNITUDES):


    class CustomRnnFcDecoder(ddsp.training.nn.OutputSplitsLayer):
        """RNN and FC stacks for f0 and loudness.
        Allows for bidirectionality
        """

        def __init__(self,
                    rnn_channels=512,
                    rnn_type='gru',
                    ch=512,
                    layers_per_stack=3,
                    input_keys=('ld_scaled', 'f0_scaled','z'),
                    output_splits=(('amps', 1), ('harmonic_distribution', 40)),
                    bidir=True,
                    **kwargs):
            super().__init__(
                input_keys=input_keys, output_splits=output_splits, **kwargs)
            stack = lambda: ddsp.training.nn.FcStack(ch, layers_per_stack)
            
            # z should be considered as input
            self.input_stacks = [stack() for k in self.input_keys]
            
            self.rnn = ddsp.training.nn.Rnn(rnn_channels, rnn_type,bidir=bidir)
            self.out_stack = stack()

        def compute_output(self, *inputs):
            # Initial processing.
                
            inputs = [stack(x) for stack, x in zip(self.input_stacks, inputs)]

            # Run an RNN over the latents.
            x = tf.concat(inputs, axis=-1)
            x = self.rnn(x)
            x = tf.concat(inputs + [x], axis=-1)

            # Final processing.
            return self.out_stack(x)
 
    class MultiInstrumentAutoencoder(ddsp.training.models.autoencoder.Autoencoder):
        def __init__(self,
                preprocessor=None,
                encoder=None,
                decoder=None,
                processor_group=None,
                losses=None,
                n_instruments=None,
                instrument_weight_metadata={},
                **kwargs):
            super().__init__(preprocessor,encoder,decoder,processor_group,losses,**kwargs)
            
            
            self.n_instruments=n_instruments
            self.instrument_weight_metadata=instrument_weight_metadata
            self.initialize_instrument_weights()
        
        def initialize_instrument_weights(self):
            self.instrument_weights={}
            for weight_name,weight_metadata in self.instrument_weight_metadata.items():
                self.instrument_weights[weight_name]=tf.Variable(weight_metadata["initializer"](self.n_instruments))
            
        def call(self, batch, train_shared):
            self.set_is_shared_trainable(train_shared)
            for weight_name,weights in self.instrument_weights.items():
                batch[weight_name]=tf.gather(weights,batch["instrument_idx"])
                if "processing" in self.instrument_weight_metadata[weight_name]:
                    batch[weight_name]=self.instrument_weight_metadata[weight_name]["processing"](batch[weight_name])
            
            # this should go in a preprocessor but I'm too lazy to write a custom preprocessor
            batch["f0_confidence"]=batch["f0_confidence"][...,None]
            
            return super().call(batch,training=False)
        
        def set_is_shared_trainable(self,train_shared):
            if self.encoder is not None:
                self.encoder.trainable=train_shared
            if self.decoder is not None:
                self.decoder.trainable=train_shared
            return
        


    class Gain(ddsp.processors.Processor):
        def __init__(self, name: Text = 'gain'):
            super().__init__(name=name)
        
        def get_signal(self, signal: tf.Tensor,
                    gain_scaled: tf.Tensor) -> tf.Tensor:
            return gain_scaled * signal
        
        def get_controls(self,signal: tf.Tensor,
                    gain: tf.Tensor) -> tf.Tensor:
            return {"signal":signal,"gain_scaled":tf.nn.relu(gain)}


    # some checkpoints have 1050 instead..
    # some have 200

  
    # 512 for single insturment, 1024 for multi
    
    IR_SIZE=int(SAMPLE_RATE*IR_DURATION)

    preprocessor=ddsp.training.preprocessing.F0LoudnessPreprocessor()

    decoder = CustomRnnFcDecoder(
                rnn_channels=512,
                rnn_type='gru',
                ch=512,
                layers_per_stack=3,
                input_keys=("ld_scaled", 'f0_scaled','z',) if not USE_F0_CONFIDENCE else ("ld_scaled", 'f0_scaled','f0_confidence','z'),
                output_splits=(('amps', 1), ('harmonic_distribution', N_HARMONICS),('magnitudes', N_NOISE_MAGNITUDES)),
                bidir=BIDIRECTIONAL
                )

    harmonic = ddsp.synths.Harmonic(
        n_samples=int(CLIP_S*SAMPLE_RATE), sample_rate=SAMPLE_RATE, name='harmonic')

    fn = ddsp.synths.FilteredNoise(
        n_samples=int(CLIP_S*SAMPLE_RATE), window_size=0, initial_bias=-5.0, name='fn')

    reverb = ddsp.effects.Reverb(name="reverb",reverb_length=IR_SIZE,add_dry=False, trainable=False)

    harmonic_plus_fn= ddsp.processors.Add(name='harmonic+fn')
    wet_gain_plus_dry_gain = ddsp.processors.Add(name='wet_gain+dry_gain')

    dry_gain = Gain(name='dry_gain')
    wet_gain = Gain(name='wet_gain')

    dag = [
    (harmonic, ['amps', 'harmonic_distribution', 'f0_hz']),
    (fn, ['magnitudes']),
    (harmonic_plus_fn, ['harmonic/signal', 'fn/signal']),
    (reverb, ["harmonic+fn/signal","ir"]),
    (wet_gain,["reverb/signal","wet_gain"]),
    (dry_gain,["harmonic+fn/signal","dry_gain"]),
    (wet_gain_plus_dry_gain,["wet_gain/signal","dry_gain/signal"])
    ]

    processor_group=ddsp.processors.ProcessorGroup(dag=dag)

    instrument_weight_metadata = {
        "z":
            {
            "initializer":lambda batch_size: tf.random.normal([batch_size,1,Z_SIZE]),
            "processing":lambda batched_feature: tf.tanh(tf.tile(batched_feature,[1,FT_FRAME_RATE*CLIP_S,1]))
            },
            "ir":
            {
                "initializer":lambda batch_size: tf.zeros([batch_size,IR_SIZE]),
                "processing":lambda batched_feature: tf.tanh(batched_feature)
                
            }
            ,
            "dry_gain":
                {
                "initializer":lambda batch_size : tf.math.sigmoid(tf.ones([batch_size,1])),
                "processing":lambda batched_feature: tf.nn.relu(batched_feature)
                },
        
            "wet_gain":
                {
                "initializer":lambda batch_size :  tf.math.sigmoid(tf.ones([batch_size,1])),
                "processing":lambda batched_feature: tf.nn.relu(batched_feature)
                }
    }

    ae = MultiInstrumentAutoencoder(
        preprocessor=preprocessor,
        decoder=decoder,
        processor_group=processor_group,
        n_instruments=N_INSTRUMENTS,
        instrument_weight_metadata=instrument_weight_metadata
    )
    return ae