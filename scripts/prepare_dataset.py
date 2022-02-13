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
flags.DEFINE_integer('sample_rate', 16000,
                     'The sample rate to use for the audio.')
flags.DEFINE_integer(
    'frame_rate', 250,
    'The frame rate to use for f0 and loudness features. If set to 0, '
    'these features will not be computed.')
flags.DEFINE_string(
    'output_tfrecord_path', None,
    'The prefix path to the output TFRecord. Shard numbers will be added to '
    'actual path(s).')

fps = glob.glob("")


def run():
    input_audio_paths = glob.glob(FLAGS.input_audio_pattern)

    print(input_audio_paths)

    # split into trn, val and test
    random.shuffle(input_audio_paths)
    n = len(input_audio_paths)

    n_tst = int(n*TST_SPLIT)
    n_dev = n-n_tst

    dev_paths = input_audio_paths[:n_dev]
    tst_paths = input_audio_paths[n_dev:]

    n_val = int(0.2*n_dev)
    n_trn = n_dev-n_val

    trn_paths = dev_paths[:n_trn]
    val_paths = dev_paths[n_trn:]

    # for each audio file, we prepare a tfrecord

    def path2filename(path):
        return path.split("/")[-1].split(".")[0]

    for p in trn_paths:
        prepare_tfrecord(
            [p],
            FLAGS.output_tfrecord_path+"/trn/"+path2filename(p),
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
            FLAGS.output_tfrecord_path+"/val/"+path2filename(p),
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
            FLAGS.output_tfrecord_path+"/tst/"+path2filename(p),
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
