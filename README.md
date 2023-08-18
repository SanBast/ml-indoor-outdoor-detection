# Machine Learning models to discern indoor from outdoor environments by using triaxial magnetic digital sensors

Master's Thesis project by Vincenzo Marcianò (*DET - Polytechnic University of Turin* & *The University of Sheffield*).

Repo status table:
* ✅ Code fully executable 
* :warning: Code in maintainance (see instructions below)
* ❌ Code temporarly unexecutable

##  Repo status: :warning: 
### What does it mean?
Code is being checked for bug fixing, but you can still run the main notebook in "notebooks" folder to experiment, train or testing on your dataset.

### Instructions:
1) Run _datamaker.ipynb_ by selecting your raw data folder coming from _.mat_ files. You can find this notebook in the _/data_ folder. This will preprocess in a table csv fashion your raw signal files and save them in a chosen repository. For your convenience you can save them under _/data/All_".

2) Split your train/test subjects by moving them respectively into _/data(Train_ and _/data/Test_. 

3) Run _ml_model.ipynb_ under _/notebooks_. This will both fit the ML models and test them following the aforementioned data folder architecture. Please, tune the *CONFIG_DICT* at the beginning of the Jupyter notebook accordingly!

4) Enjoy your experiments! :smile:

## Reference paper
... Soon! :crossed_fingers: