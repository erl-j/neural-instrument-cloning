import ddsp.training
import tensorflow as tf

class MultiTFRecordProvider():
	"""Class for reading records and returning a dataset."""
	def __init__(self, file_pattern=None, example_secs=4, sample_rate=16000, frame_rate=250):
		"""TFRecordProvider constructor."""
		self.file_pattern=file_pattern
		self.example_secs=example_secs
		self.sample_rate=sample_rate
		self.frame_rate=frame_rate
	
	def get_dataset(self, shuffle=True):
		"""Read dataset.
		Args:
		shuffle: Whether to shuffle the files.
		Returns:
		dataset: A tf.dataset that reads from the TFRecord.
		"""
		filenames = tf.data.Dataset.list_files(self.file_pattern, shuffle=shuffle)
		multi_dataset=None
		for f in filenames:
			instrument_dataset_provider = ddsp.training.data.TFRecordProvider(f,self.example_secs,self.sample_rate,self.frame_rate)
			instrument_dataset = instrument_dataset_provider.get_dataset()
			instrument_dataset=instrument_dataset.map(lambda x: {**x,"instrument":f})
			if multi_dataset == None:
				multi_dataset=instrument_dataset
			multi_dataset=multi_dataset.concatenate(instrument_dataset)
		if shuffle:
			multi_dataset=multi_dataset.shuffle(1000)
		return multi_dataset

#test_dp=MultiTFRecordProvider("datasets/solos-violin-clean/tfr/val/*",4,16000,250)

#ds=test_dp.get_dataset()

#print(next(iter(ds))["source_filename"])







