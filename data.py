import ddsp.training
import tensorflow as tf
_AUTOTUNE = tf.data.experimental.AUTOTUNE


class MultiTFRecordProvider():
    """Class for reading records and returning a dataset."""

    def __init__(self, file_pattern=None, example_secs=4, sample_rate=16000, frame_rate=250):
        """TFRecordProvider constructor."""
        self.file_pattern = file_pattern
        self.example_secs = example_secs
        self.sample_rate = sample_rate
        self.frame_rate = frame_rate

    def get_dataset(self, shuffle=False):
        """Read dataset.
        Args:
        shuffle: Whether to shuffle the files.
        Returns:
        dataset: A tf.dataset that reads from the TFRecord.
        """
        filenames = tf.data.Dataset.list_files(
            self.file_pattern, shuffle=shuffle)
        multi_dataset = None
        for fi, f in enumerate(filenames):
            instrument_dataset_provider = ddsp.training.data.TFRecordProvider(
                f, self.example_secs, self.sample_rate, self.frame_rate)
            instrument_dataset = instrument_dataset_provider.get_dataset()
            instrument_dataset = instrument_dataset.map(
                lambda x: {**x, "instrument": f, "instrument_idx": fi})
            if multi_dataset == None:
                multi_dataset = instrument_dataset
            multi_dataset = multi_dataset.concatenate(instrument_dataset)

        if shuffle:
            multi_dataset = multi_dataset.shuffle(1000)

        return multi_dataset

# test_dp=MultiTFRecordProvider("datasets/solos-violin-clean/tfr/val/*",4,16000,250)

# ds=test_dp.get_dataset()

# print(next(iter(ds))["source_filename"])

# TODO : Add instrument index


class CustomNSynthTfds(ddsp.training.data.TfdsProvider):
    """Parses features in the TFDS NSynth dataset.
    Unlike the default Nsynth data provider, this class keeps the the nsynth instrument metadata.

    If running on Cloud, it is recommended you set `data_dir` to
    'gs://tfds-data/datasets' to avoid unnecessary downloads.
    """

    def __init__(self,
                 name='nsynth/gansynth_subset.f0_and_loudness:2.3.3',
                 split='train',
                 data_dir='gs://tfds-data/datasets',
                 sample_rate=16000,
                 frame_rate=250,
                 include_note_labels=True):
        """TfdsProvider constructor.
        Args:
                        name: TFDS dataset name (with optional config and version).
                        split: Dataset split to use of the TFDS dataset.
                        data_dir: The directory to read the prepared NSynth dataset from. Defaults
                        to the public TFDS GCS bucket.
                        sample_rate: Sample rate of audio in the dataset.
                        frame_rate: Frame rate of features in the dataset.
                        include_note_labels: Return dataset without note-level labels
                        (pitch, instrument).
        """
        self._include_note_labels = include_note_labels
        super().__init__(name, split, data_dir, sample_rate, frame_rate)

    def get_dataset(self, shuffle=True):
        """Returns dataset with slight restructuring of feature dictionary."""

        def preprocess_ex(ex):
            ex_out = {
                'audio':
                ex['audio'],
                'f0_hz':
                ex['f0']['hz'],
                'f0_confidence':
                ex['f0']['confidence'],
                'loudness_db':
                ex['loudness']['db'],
            }
            if self._include_note_labels:
                ex_out.update({
                    'pitch':
                    ex['pitch'],
                    'velocity':
                    ex['velocity'],
                    'instrument_source':
                    ex['instrument']['source'],
                    'instrument_family':
                    ex['instrument']['family'],
                    'instrument':
                    ex['instrument']['label'],
                    'instrument_idx':
                    int(ex['instrument']['label'])
                })
                return ex_out

        dataset = super().get_dataset(shuffle)
        dataset = dataset.map(
            preprocess_ex, num_parallel_calls=_AUTOTUNE)
        return dataset
