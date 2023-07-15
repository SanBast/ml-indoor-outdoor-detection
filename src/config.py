DATA_CONFIG = {
	'path': '/path/to/subjects.csv',
	'paths_new_format': ['...', '...'], # some data file are retrieved differently
	'sensor': 'LF',
	'win_size': 100
}

TRAIN_PARAMS = {
	'batch_size': 128,
	'n_epochs': 100,
	'ckpt_dir': 'ckpt_dir/',
	'accelerator': 'gpu'
}

EARLY_STOPPING_PARAMS = {
	'monitor': 'val_accuracy',
	'patience': 7,
	'verbose': False,
	'mode': 'min'
}

COL_NAMES = {
	'LowerBack': 'LB', 
	'LeftFoot': 'LF', 
	'RightFoot':'RF', 
	'Wrist':'WR'
}
