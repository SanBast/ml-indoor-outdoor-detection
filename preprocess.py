import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from config import COL_NAMES
from sktime.datatypes._panel._convert import from_3d_numpy_to_nested


class DataCollector(object):
	def __init__(self, *arg, **kwargs):
		self.path = kwargs['path']
		self.paths_new_format = kwargs['paths_new_format']
		self.sensor = kwargs['sensor']
  
		#read all the ids and concatenate them
		self.df = pd.concat([
      		pd.read_csv(p).dropna(inplace=True)
        ] for p in self.path) if self.path is not None else pd.DataFrame([])
		self.df.reset_index(inplace=True, drop=True)

		assert self.path is not None or self.paths_new_format is not None
  
  		# for subjects beginning with 6xxx
		self.df_new_format = [
			pd.read_csv(p).dropna() for p in self.paths_new_format
		] if self.paths_new_format is not None else pd.DataFrame([])

	def __len__(self):
		return len(self.df)

	def __getitem__(self, idx):
		return self.df.iloc[idx] 

	def drop_columns(self):
		for c in self.df.columns:
			if not ('Mag' in c or c.startswith('T') or c=='Indoor Probability' or c=='Patient ID'):
				self.df.drop(columns=c, inplace=True)
		return self

	def add_norm(self):
		for sensor in COL_NAMES.keys():
			sens_cols = [col for col in self.df.columns if sensor in col]
			self.df[f'Mag{sensor}_Norm'] = np.linalg.norm(
				self.df[sens_cols].values,
				axis=1
			)
		for c in self.df.columns:
			for k in COL_NAMES.keys():
				if k in c:
					if 'Norm' in c:
						ax = c.split('_')[-1]
					else:
						ax = c.split('_')[-1].lower()[0]
			self.df.rename(columns={c:f'Mag{COL_NAMES[k]}_{ax}'}, inplace=True)
		return self

	def rename_cols(self):
		for c in self.df.columns:
			if 'Timestamp' in c:
				self.df.rename(columns={c:'Timestamp'}, inplace=True)
			elif 'Indoor' in c:
				self.df.rename(columns={c:'Indoor'}, inplace=True)
			elif 'Patient' in c:
				self.df.rename(columns={c:'Patient'}, inplace=True)
		self.df.reset_index(inplace=True, drop=True)
		return self 

	def save_data_space(self):
		for col in self.df.columns:
			if self.df[col].dtypes=='float64':
				self.df[col] = self.df[col].astype('float32')
			if self.df[col].dtype=='int64':
				self.df[col] = self.df[col].astype('int16')
		return self

	def convert_timestamps(self):
		self.df['Timestamp'] = pd.to_datetime(self.df['Timestamp'], unit='s')
		return self

	def preprocess(self):
		self.drop_columns()
		self.add_norm()
		self.rename_cols()
		self.save_data_space()
		self.convert_timestamps()
		return self.df

class DataframeToSeq(DataCollector):
	def __init__(self, *args, **kwargs):
		self.win_size = kwargs['win_size']
		super().__init__(*args, **kwargs)
		self.df = self.preprocess()
		self.df = pd.concat([self.df]+self.df_new_format)
		self.FEATURE_COLS = []
		self.sequences = []
		self.labels = []
  	
   	#------TODO------
    #def select_valids

	def prepare_for_slicing(self):
		not_cols = set(['Timestamp', 'Date', 'Patient'])
		all_cols = set(self.df.columns.to_list())

		self.FEATURE_COLS = sorted(list(set.difference(all_cols, not_cols)))

		df_preprocess = pd.DataFrame(columns=self.FEATURE_COLS)

		for k, group in self.df.groupby('Patient'):
			print('Processing patient: ', k)
			
			# 6001, 6002, 6003 ids are retrieved differently 
   			# from standards
			if k not in [6001, 6002, 6003]:
				for ts in group.Timestamp.unique():
					if group[group['Timestamp']==ts]['Timestamp'].value_counts().values in [12800, 12799, 12801]:
						data = group[group['Timestamp']==ts][self.FEATURE_COLS]
						df_train = pd.concat([df_train, data])
			else:
				data = group[~(group['Date'].isin(['13/04/2022 10:05:37','13/04/2022 18:34:28','14/06/2022 11:16:13', 
												'14/06/2022 14:20:03', '14/04/2022 07:58:00', '14/04/2022 16:22:01']))][self.FEATURE_COLS]
				df_preprocess = df_preprocess.append(data)

		df_preprocess['series_id'] = np.arange(len(df_preprocess)) // self.win_size + 1
		y = df_preprocess[['series_id', 'Indoor']]

		return df_preprocess, y

	@staticmethod
	def count_odd_slices(df):
		for t in df.series_id.value_counts().values:
			if t!=100:
				count_slices+=t
		return df.iloc[:-count_slices] if count_slices!=0 else df

	def return_sequences(self):
		df_preprocess, y = self.prepare_for_slicing()
		df_slices = self.count_odd_slices(df_preprocess)

		sequences = []
		labels = []
  
		for series_id, group in tqdm(df_slices.groupby('series_id'), position=0, leave=True):
			sequence_features = group[self.FEATURE_COLS[1:]].to_numpy()
			labels.append(y[y.series_id==series_id].iloc[0].Indoor)
			sequences.append(sequence_features)

		sequences = np.swapaxes(np.array(sequences),1,2)
		self.sequences, self.labels = sequences, labels
		return self.sequences, self.labels

	def return_sequences_rnn(self):
		df_preprocess, y = self.prepare_for_slicing()
		df_slices = self.count_odd_slices(df_preprocess)
  
		for series_id, group in tqdm(df_slices.groupby('series_id'), position=0, leave=True):
			sequence_features = group[self.FEATURE_COLS[1:]]
			label = y[y.series_id==series_id].iloc[0].Indoor
			self.sequences.append((sequence_features, label))
		return self.sequences

	def return_nested_df(self):
		sequences, labels = self.return_sequences()
		nested_df = from_3d_numpy_to_nested(
      					sequences, 
           				column_names=self.FEATURE_COLS[1:], 
               			cells_as_numpy=True
                  	)
		y_df = pd.DataFrame(labels)
		return nested_df,y_df
  
  		
