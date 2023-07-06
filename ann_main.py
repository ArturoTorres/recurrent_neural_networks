# ANN algorithms
# J. A. Torres-Matallana
# since 2023-07-03 to 2023-07-04

def ann_main(df, y1, df1_wet, case_ann, lags, df2_wet, folder_output):

    import pandas as pd
    from ann3 import my_ann3
    from ann3_nodate import my_ann3_nodate
    from ann_model import ann_model2
    from functions import lagged
    from sklearn.preprocessing import MinMaxScaler
    import tensorflow as tf
    from functions import my_plot
    from sklearn.metrics import r2_score
    from validation import validation

    if (case_ann == 'ann_model2'):
        ann_data1 = my_ann3(df=df['FWD (C/N)'], # df1_wet_nodate_scaled['FWD (C/N)'],
                            y=y1['rain_intensity_rg'], # df1_wet_nodate_scaled['rain_intensity_rg'],
                            df_date=df1_wet[['timestamp_utc', 'FWD (C/N)']],
                            y_date=df1_wet[['timestamp_utc', 'rain_intensity_rg']],
                            ann_model=ann_model2,
                            par_epochs=10,
                            par_batch_size=24,
                            folder_output=folder_output,
                            name='ann_model2_prediction_data1')
        return [ann_data1]

    elif (case_ann == 'ann_lagged'):
        # lagged

        lags
        df1_wet.columns
        df1_x_wet_lagged = lagged(df=pd.DataFrame(df1_wet['FWD (C/N)']), par_lags=lags)
        df1_x_wet_lagged.columns

        df1_y_wet = df1_wet['rain_intensity_rg'].loc[max(lags):]
        df1_y_wet.reset_index(inplace=True, drop=True)  # reset index to start in 0

        ## MinMaxScaler
        scaler = MinMaxScaler()
        df1_x_wet_lagged.columns
        df_lagged = pd.DataFrame(scaler.fit_transform(df1_x_wet_lagged), columns=df1_x_wet_lagged.columns)
        print(df_lagged)

        ann_data1_lagged = my_ann3_nodate(df=df_lagged,
                                          y=df1_y_wet,
                                          ann_model=ann_model2,
                                          par_epochs=150*2,
                                          par_batch_size=12*2,
                                          par_learning_rate = 0.001,
                                          folder_output=folder_output,
                                          name='/ann_prediction_filtered_lagged_scaled_data1')

        df1_train = pd.concat([pd.DataFrame(ann_data1_lagged[3]), pd.DataFrame(ann_data1_lagged[4])], axis=1)
        df1_train.columns
        df1_train.rename(columns={0: 'Predicted'}, inplace=True)

        my_plot(df_nodate=df1_train, cols=range(0,2), file_name=folder_output + '/data1_simple_ann_lagged_prediction_train')

        # validation
        df2_wet.columns
        df2_x_wet_lagged = lagged(df=pd.DataFrame(df2_wet['FWD (C/N)']), par_lags=lags)
        df2_x_wet_lagged.columns

        df2_y_wet = df2_wet['rain_intensity_rg'].loc[max(lags):]
        df2_y_wet.reset_index(inplace=True, drop=True)  # reset index to start in 0

        ## MinMaxScaler
        df2_x_wet_lagged.columns
        df2_lagged = pd.DataFrame(scaler.fit_transform(df2_x_wet_lagged), columns=df2_x_wet_lagged.columns)
        print(df2_lagged)

        validation(model=ann_data1_lagged[0],
                   X=df2_lagged,
                   y=df2_y_wet,
                   folder_output=folder_output,
                   name='data2_ann_lagged_scaled')

        return [ann_data1_lagged]