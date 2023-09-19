import os
import pandas as pd
import scipy.io as spio

from utils import (check_indoor,
                   _check_keys,
                   print_mat_nested,
                   preprocess_subject,
                   filter_values,
                   print_subj_stats)

import gc

gc.enable()


class DataProcessor:
    def __init__(self, data_folder, folder_out, data_format, ts=997417, verbose=False):
        self.data_folder = data_folder
        self.folder_out = folder_out
        self.data_format = data_format
        self.ts = ts
        self.verbose = verbose

    def process_data(self):
        full_df = pd.DataFrame()

        for folder in os.listdir(self.data_folder):
            for i, patient in enumerate(os.listdir(os.path.join(self.data_folder, folder))):
                try:
                    print(f"Processing patient id: {patient}")
                    if os.path.isfile(os.path.join(self.folder_out, f'df_{patient}.csv')):
                        print("Existing file for ID: ", patient)
                    else:
                        # -------------LOAD DATA-------------
                        filemat = os.path.join(self.data_folder, folder, patient, 'Out of Lab', self.data_format)
                        matdata = self.loadmat(filemat)
                        recording = matdata['data']['TimeMeasure1']['Recording4']
                        if self.verbose:
                            print_mat_nested(recording)

                        # ---------CREATE RAW DATASET--------
                        df = pd.json_normalize(recording)

                        df_triaxial = pd.DataFrame()
                        for c in df.columns:
                            if 'Acc' in c and 'Fs' not in c:
                                df_triaxial[f'{c}_X (g)'] = pd.Series([v[0] for v in df[c].values[0]])
                                df_triaxial[f'{c}_Y (g)'] = pd.Series([v[1] for v in df[c].values[0]])
                                df_triaxial[f'{c}_Z (g)'] = pd.Series([v[2] for v in df[c].values[0]])
                            elif 'Gyr' in c and 'Fs' not in c:
                                df_triaxial[f'{c}_X (deg/s)'] = pd.Series([v[0] for v in df[c].values[0]])
                                df_triaxial[f'{c}_Y (deg/s)'] = pd.Series([v[1] for v in df[c].values[0]])
                                df_triaxial[f'{c}_Z (deg/s)'] = pd.Series([v[2] for v in df[c].values[0]])
                            elif 'Mag' in c and 'Fs' not in c:
                                df_triaxial[f'{c}_X (uT)'] = pd.Series([v[0] for v in df[c].values[0]])
                                df_triaxial[f'{c}_Y (uT)'] = pd.Series([v[1] for v in df[c].values[0]])
                                df_triaxial[f'{c}_Z (uT)'] = pd.Series([v[2] for v in df[c].values[0]])
                        df_triaxial['Timestamp (ms)'] = recording['SU_INDIP']['LowerBack']['Timestamp']

                        # ---------FILTER RAW DATASET--------
                        for ctxs in os.listdir(os.path.join(self.data_folder, folder, patient,
                                                            'Out of Lab/Contextual Factors')):
                            if ctxs.startswith('stay'):
                                staypts = pd.read_json(os.path.join(self.data_folder, folder, patient,
                                                                   'Out of Lab/Contextual Factors', ctxs))['data']
                            elif ctxs.startswith('per'):
                                ctx = pd.read_json(os.path.join(self.data_folder, folder, patient,
                                                               'Out of Lab/Contextual Factors', ctxs))['data'][0]
                            elif ctxs.startswith('path'):
                                _ = pd.read_json(os.path.join(self.data_folder, folder, patient,
                                                              'Out of Lab/Contextual Factors', ctxs))

                        filtered_ctx = filter_values(ctx, df_triaxial, self.ts)
                        df_triaxial['Indoor Probability'] = pd.Series(check_indoor(filtered_ctx, staypts))

                        df_triaxial['Patient ID'] = pd.Series([patient for _ in range(len(df_triaxial))])
                        df_triaxial['Disease'] = pd.Series([folder for _ in range(len(df_triaxial))])
                        print("Length of current dataset: ", len(df_triaxial))

                        if len(df_triaxial) > 0.5 * self.ts:
                            print_subj_stats(df_triaxial)
                            print('Saving patient CSV file...')
                            df_triaxial = preprocess_subject(df_triaxial)
                            df_triaxial.to_csv(os.path.join(self.folder_out, f'df_{patient}.csv'))
                            print('Saved!')
                        else:
                            print('Too few samples for ID: ', patient)
                except:
                    print(f'Corrupted patient file: {patient}')
                    continue
                print('--------------------------')
        return full_df
    
    def loadmat(filename):
        '''
        this function should be called instead of direct spio.loadmat
        as it cures the problem of not properly recovering python dictionaries
        from mat files. It calls the function check keys to cure all entries
        which are still mat-objects
        '''
        #print(filename)
        data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
        return _check_keys(data)
