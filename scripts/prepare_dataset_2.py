import glob
from absl import app
from absl import flags
import random
import tensorflow as tf
from ddsp.training.data_preparation.prepare_tfrecord_lib import prepare_tfrecord

SEED = 0
TST_SPLIT = 0.2
VAL_SPLIT = 0.2

WINDOW_S = 4
TRN_HOP_SIZE = 1
COARSE_CHUNK_S = 20.0

random.seed(SEED)

FLAGS = flags.FLAGS

flags.DEFINE_string('input_audio_pattern', None,
                    'Path to audio')
flags.DEFINE_integer('sample_rate', 48000,
                     'The sample rate to use for the audio.')
flags.DEFINE_integer(
    'frame_rate', 250,
    'The frame rate to use for f0 and loudness features. If set to 0, '
    'these features will not be computed.')
flags.DEFINE_string(
    'output_tfrecord_path', None,
    'The prefix path to the output TFRecord. Shard numbers will be added to '
    'actual path(s).')

def run():
    trn_paths = glob.glob(f"datasets/AIR/wav/dev/{FLAGS.input_audio_pattern}/*")
    val_paths = glob.glob(f"datasets/AIR/wav/val/{FLAGS.input_audio_pattern}/*")
    tst_paths=[]

    # delete this
    trn_paths=[]
    def path2filename(path):
            return path.split("/")[-1].split(".")[0]

    for p in trn_paths:
        prepare_tfrecord(
            [p],
            FLAGS.output_tfrecord_path+"/trn/"+FLAGS.input_audio_pattern+"/"+path2filename(p),
            num_shards=1,
            sample_rate=FLAGS.sample_rate,
            frame_rate=FLAGS.frame_rate,
            window_secs=WINDOW_S,
            hop_secs=TRN_HOP_SIZE,
            eval_split_fraction=0.0,
            coarse_chunk_secs=COARSE_CHUNK_S)

    for p in val_paths:
        prepare_tfrecord(
            [p],
            FLAGS.output_tfrecord_path+"/val/"+FLAGS.input_audio_pattern+"/"+path2filename(p),
            num_shards=1,
            sample_rate=FLAGS.sample_rate,
            frame_rate=FLAGS.frame_rate,
            window_secs=WINDOW_S,
            hop_secs=WINDOW_S,
            eval_split_fraction=0.0,
            coarse_chunk_secs=COARSE_CHUNK_S)

    for p in tst_paths:
        prepare_tfrecord(
            [p],
            FLAGS.output_tfrecord_path+"/tst/"+FLAGS.input_audio_pattern+"/"+path2filename(p),
            num_shards=1,
            sample_rate=FLAGS.sample_rate,
            frame_rate=FLAGS.frame_rate,
            window_secs=WINDOW_S,
            hop_secs=WINDOW_S,
            eval_split_fraction=0.0,
            coarse_chunk_secs=COARSE_CHUNK_S)


def main(unused_argv):
    """From command line."""
    run()


def console_entry_point():
    """From pip installed script."""
    app.run(main)


if __name__ == '__main__':
    console_entry_point()
