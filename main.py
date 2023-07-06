# RNN and GRU algorithms - Main Script
# J. A. Torres-Matallana
# since 2023-07-03 to 2023-07-04

import os
import pandas as pd
import time
import tensorflow as tf

from ann_main import ann_main
from ann3_nodate import my_ann3_nodate

#from ann_model import ann_model1
from ann_model import ann_model2
#from valida_2022 import valida_2022
from functions import lagged
from sklearn.preprocessing import MinMaxScaler
from functions import my_plot
from data1 import data1
from rnn1 import my_rnn_gru, my_rnn_gru_validation, my_rnn_pre
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler

folder_current = os.getcwd()
folder_base = os.path.dirname(folder_current)
folder_projects = os.path.dirname(folder_base)

folder_data = folder_base + '/ds_task'
folder_output = folder_current + '/output'

# execute data_2020_2021
# exec(open("data1.py").read())
df1, df1_nodate, df1_wet, df1_wet_nodate, df, y1 = data1(file0=folder_data + '/data1.csv',
                                                         name='data1',
                                                         folder_output=folder_output)

df1_nodate.columns
df1.columns


# =======================================
# Random Forest Dry/Wet Classification
# =======================================


# =======================================
# Training ANN
# =======================================
# devices = tf.config.list_physical_devices()
# print(devices)

# tf.debugging.set_log_device_placement(True)
# a=tf.random.normal([100,100])
# b=tf.random.normal([100,100])
# c = a*b


# case_ann = 'ann_model2'
case_ann = 'ann_lagged' # includes validation

# lagged
lags = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]


## validation
df2, df2_nodate, df2_wet, df2_wet_nodate, df2_wet_nodate_scaled, y2 = data1(file0=folder_data + '/data2.csv',
                                                                            name='data2',
                                                                            folder_output=folder_output)

t_start = time.time()
ann_model = ann_main(df, y1, df1_wet, case_ann, lags, df2_wet, folder_output)
t_end = time.time()
print(t_end - t_start)

# =======================================
# rnn gru algorithm
# =======================================
n_future=1*1 #24
n_past_par=24*1 #3
total_period_par=25*1 #4
learn_rate=0.005

case_par= 'gru1' # 'simple_gru2' # 'simple_gru1' # 'gru1' # 'gru2'

df.describe()
y1.describe()

t_start = time.time()
model_gru1, preds_gru1, y_train_gru1, y_new_gru1, y_gru1 = my_rnn_gru(df, y1, folder_output, learn_rate, n_future,
                                                                      n_past_par, total_period_par,
                                                                      validation_split_par=0.2, epochs_par=10*40,
                                                                      batch_size_par=24, case_par=case_par)
t_end = time.time()
print(t_end - t_start)


y1.columns
y_new_gru1.shape
#y1_train_gru1 = pd.concat([pd.DataFrame(y_new_gru1).iloc[:,0], pd.DataFrame(y1).iloc[:,0]], axis=1)
#y1_train_gru1.columns = ['y_new', 'Observed']

#my_plot(df_nodate=y1_train_gru1, cols=range(0,2), file_name=folder_output + '/' + 'prediction_y_vs_obs_data1_' + 'gru1')
#print(r2_score(y1_train_gru1.iloc[:,1], y1_train_gru1.iloc[:,0]))

df2_wet.columns
df2_wet.shape
df2_wet

scaler = MinMaxScaler()
df2_wet_scaled = pd.DataFrame(scaler.fit_transform(df2_wet.iloc[:,[1,2]]), columns=df2_wet.iloc[:,[1,2]].columns)
df2_wet_scaled

preds_gru1_val, y_new_gru1_val = my_rnn_gru_validation(model=model_gru1,
                                                       df=df2_wet_scaled,
                                                       y1=df2_wet,
                                                       n_future=n_future,
                                                       n_past_par=n_past_par,
                                                       total_period_par=total_period_par,
                                                       folder_output=folder_output,
                                                       name='data2_validation_' + case_par)

model_gru1.summary()

df2_wet.columns
df2_wet.shape
y_new_gru1_val.shape
y2_val_gru1 = pd.concat([pd.DataFrame(y_new_gru1_val).iloc[:,0], pd.DataFrame(df2_wet).iloc[:,2]], axis=1)
y2_val_gru1.columns = ['y2_new', 'Observed']

my_plot(df_nodate=y2_val_gru1, cols=range(0,2), file_name=folder_output + '/' + 'validation_y_vs_obs_data2_' + 'gru1')
print(r2_score(y2_val_gru1.iloc[:,1], y2_val_gru1.iloc[:,0]))

# =======================================
# Convert pdf figures in png format
# =======================================
from functions import my_pdf2png_2

folder_output = '/Users/torres/Documents/02_working/3-Production/02_projects/19_crake/20230706_progress_meeting/figs'
my_pdf2png_2(folder=folder_output)