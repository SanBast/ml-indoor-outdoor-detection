import scipy.io as spio
import os
import pandas as pd
import numpy as np
import math
import itertools


def preprocess_subject(df):
  for c in df.columns:
    if not ('Mag' in c or c.startswith('T') or c=='Indoor Probability' or c=='Patient ID'):
      df.drop(columns=c, inplace=True)

  col_names = {'LowerBack': 'LB', 'LeftFoot': 'LF', 'RightFoot':'RF', 'Wrist':'WR'}

  for sensor in col_names.keys():
    sens_cols = [col for col in df.columns if sensor in col]
    df[f'Mag{sensor}_Norm'] = np.linalg.norm(df[sens_cols].values,axis=1)

  for c in df.columns:
    for k in col_names.keys():
      if k in c:
        if 'Norm' in c:
          ax = c.split('_')[-1]
        else:
          ax = c.split('_')[-1].lower()[0]
        df.rename(columns={c:f'Mag{col_names[k]}_{ax}'}, inplace=True)

    if 'Timestamp' in c:
      df.rename(columns={c:'Timestamp'}, inplace=True)
    elif 'Indoor' in c:
      df.rename(columns={c:'Indoor'}, inplace=True)
    elif 'Patient' in c:
      df.rename(columns={c:'Patient'}, inplace=True)

  df.reset_index(inplace=True, drop=True)
  return df


def check_indoor(filtered_ctx, staypts):
  indoor=[]
  for j,el in enumerate(filtered_ctx):
    #stay_id.append(filtered_ctx[j][2])
    if filtered_ctx[j][1]!=50:
      if filtered_ctx[j][1]>50:
        indoor.append(1)
      else:
        indoor.append(0)
    else:
      indoor.append(0.5)
  return indoor


def filter_values(context_file, df, num_ts):
  context = context_file['contextValues']
  start = 0
  end = 0
  filtered_ctx = []
  for i,t in enumerate(context.keys()):
    if int(t)>=math.trunc(df['Timestamp (ms)'].iloc[0]) and int(t)<=math.trunc(df['Timestamp (ms)'].iloc[-1]):
      filtered_ctx.append([int(t), context[t][0], context[t][1], context[t][2], context[t][3]])

  filtered_ctx = list(itertools.chain.from_iterable(itertools.repeat(x, 100) for x in filtered_ctx))[:num_ts]
  return filtered_ctx


def _check_keys(dict):
    '''
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    '''
    for key in dict:
        if isinstance(dict[key], spio.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict

def _todict(matobj):
    '''
    A recursive function which constructs from matobjects nested dictionaries
    '''
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, spio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict

def print_mat_nested(d, indent=0, nkeys=0):
    # Subset dictionary to limit keys to print.  Only works on first level
    if nkeys>0:
        d = {k: d[k] for k in list(d.keys())[:nkeys]}  # Dictionary comprehension: limit to first nkeys keys.

    if isinstance(d, dict):
        for key, value in d.items():         # iteritems loops through key, value pairs
          print('\t' * indent + 'Key: ' + str(key))
          print_mat_nested(value, indent+1)

    if isinstance(d,np.ndarray) and d.dtype.names is not None:  # Note: and short-circuits by default
        for n in d.dtype.names:    # This means it's a struct, it's bit of a kludge test.
            print('\t' * indent + 'Field: ' + str(n))
            print_mat_nested(d[n], indent+1)
            

def print_subj_stats(df_triaxial):
    print('Total len: ', len(df_triaxial))
    print('Total null: ', df_triaxial[df_triaxial['Indoor Probability']==0.5]['Indoor Probability'].sum())
    print('Total indoor: ', df_triaxial[df_triaxial['Indoor Probability']==1]['Indoor Probability'].sum())
    print('Total null percentage: ', df_triaxial[df_triaxial['Indoor Probability']==0.5]['Indoor Probability'].sum()/len(df_triaxial)*100)
    print('Total indoor percentage: ', df_triaxial[df_triaxial['Indoor Probability']==1]['Indoor Probability'].sum()/len(df_triaxial)*100)