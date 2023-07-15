import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew, iqr, trim_mean
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

class FeatExtractor():
    def __init__(self, sensors, type:str="all") -> None:
        self.sensors = sensors
        self.type = type
        self.preprocess_dict = {
            'time': [],
            'corr': [],
            'freq': []
        }

    @staticmethod
    def calc_corr(data,axis,sensor):
        if axis=='xy':
            corr_xy = np.corrcoef(data[f'Mag{sensor}_{axis[0]}'], data[f'Mag{sensor}_{axis[1]}'])
            return corr_xy[0][1]
        elif axis=='xz':
            corr_xz = np.corrcoef(data[f'Mag{sensor}_{axis[0]}'], data[f'Mag{sensor}_{axis[1]}'])
            return corr_xz[0][1]
        elif axis=='yz':
            corr_yz = np.corrcoef(data[f'Mag{sensor}_{axis[0]}'], data[f'Mag{sensor}_{axis[1]}'])
            return corr_yz[0][1]
    
    @staticmethod
    def extract_dominant(data, sampling_rate):
        fft_data = np.fft.fft(data)
        freqs = np.fft.fftfreq(len(data))
        peak_coefficient = np.argmax(np.abs(fft_data))
        peak_freq = freqs[peak_coefficient]
        
        return abs(peak_freq * sampling_rate)

    @staticmethod
    def extract_time_feat(df):
        return {
            'mean': np.array([np.array([np.mean(row[c]) for c in row.keys()]) for _,row in df.iterrows()]),
            'median': np.array([np.array([np.median(row[c]) for c in row.keys()]) for _,row in df.iterrows()]),
            'std': np.array([np.array([np.std(row[c]) for c in row.keys()]) for _,row in df.iterrows()]),
            'mad': np.array([np.array([np.mean(np.abs(row[c]-np.mean(row[c]))) for c in row.keys()]) for _,row in df.iterrows()]), 
            'rms': np.array([np.array([np.sqrt(np.mean(row[c]**2)) for c in row.keys()]) for _,row in df.iterrows()]), 
            'vars': np.array([np.array([np.var(row[c]) for c in row.keys()]) for _,row in df.iterrows()]), 
            'kurts': np.array([np.array([kurtosis(row[c]) for c in row.keys()]) for _,row in df.iterrows()]), 
            'skews': np.array([np.array([skew(row[c]) for c in row.keys()]) for _,row in df.iterrows()]),
            'p_1': np.array([np.array([np.percentile(row[c], 1) for c in row.keys()]) for _,row in df.iterrows()]),
            'p_10': np.array([np.array([np.percentile(row[c], 10) for c in row.keys()]) for _,row in df.iterrows()]),
            'p_25': np.array([np.array([np.percentile(row[c], 25) for c in row.keys()]) for _,row in df.iterrows()]),
            'p_50': np.array([np.array([np.percentile(row[c], 50) for c in row.keys()]) for _,row in df.iterrows()]),
            'p_75': np.array([np.array([np.percentile(row[c], 75) for c in row.keys()]) for _,row in df.iterrows()]),
            'p_99': np.array([np.array([np.percentile(row[c], 99) for c in row.keys()]) for _,row in df.iterrows()]),
            'iqr': np.array([np.array([iqr(row[c]) for c in row.keys()]) for _,row in df.iterrows()]),
            'trim_mean125': np.array([np.array([trim_mean(row[c], 0.125) for c in row.keys()]) for _,row in df.iterrows()])
        }
        
    @staticmethod
    def extract_corr_sensors(self, df):    
        return {
            'c_xy': np.array([np.array([self.calc_corr(row, 'xy', s) for s in self.sensors]) for _,row in df.iterrows()]), 
            'c_xz': np.array([np.array([self.calc_corr(row, 'xz', s) for s in self.sensors]) for _,row in df.iterrows()]), 
            'c_yz': np.array([np.array([self.calc_corr(row, 'yz', s) for s in self.sensors]) for _,row in df.iterrows()])
        }

    @staticmethod
    def extract_freq_feat(self,df):
        return {
            'f0': np.array([np.array([self.extract_dominant(row[c], 100) for c in row.keys()]) for _,row in df.iterrows()])
        }

    @staticmethod
    def create_feat_df(self,feat_dict, is_corr=True):
        num_feat = 4 if is_corr else 16
        return pd.concat([pd.DataFrame(feat_dict[tf], 
                                    columns=[f'{tf}_dim{i}' for i in range(16)]) for tf in feat_dict.keys()],
                        axis=1)        
    
    def wrap_all_features(self, df, return_single=False):
        time_feat = self.extract_time_feat(df), 
        corr_feat = self.extract_corr_sensors(df),
        freq_feat = self.extract_freq_feat(df)
        return  pd.concat([
                self.create_feat_df(time_feat, is_corr=False),
                self.create_feat_df(corr_feat),
                self.create_feat_df(freq_feat, is_corr=False)
            ], axis=1) if not return_single else (
                self.create_feat_df(time_feat, is_corr=False),
                self.create_feat_df(corr_feat),
                self.create_feat_df(freq_feat, is_corr=False)
            )
    
    def preprocess(self, df):
        if self.type == 'all':
            return self.wrap_all_features(df)
        feat_df_list = self.wrap_all_features(df, return_single=True)
        self.preprocess_dict = dict(zip(self.preprocess_dict.keys(), feat_df_list))
        return self.preprocess_dict[self.type]
    
    
class PCAExtractor():
    def __init__(self, n_components) -> None:
        self.n_components = n_components
        self.fitted_pca = []
        
    def auto_select_comp(self, thres_variance, X, plot=False):
        pca = PCA()
        pca.fit_transform(X)

        total = sum(pca.explained_variance_)
        k = 0
        current_variance = 0
        while current_variance/total < thres_variance:
            current_variance += pca.explained_variance_[k]
            k = k + 1    
            
        print(k, f"features explain around {thres_variance*100}% of the variance.")
        if plot:
            self.plot_results(X, k)
            
        self.n_components = k if self.n_components == "auto" else self.n_components
        self.fitted_pca = pca
        return self
        
    def preprocess(self, X, is_test=False):
        X_pca = self.fitted_pca.fit_transform(X) if not is_test else self.fitted_pca.transfor(X)
        return X_pca
    
    @staticmethod
    def plot_results(self, X, k):
        pca = PCA(n_components=k)
        _ = pca.fit_transform(X)

        var_exp = pca.explained_variance_ratio_.cumsum()
        var_exp = var_exp*100
        plt.bar(range(k), var_exp)
        plt.show()
        