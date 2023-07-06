# Data reading and preprocessing
# J. A. Torres-Matallana
# since 2023-07-03 to 2023-07-04

def data1(file0, name, folder_output):
    # file0 = folder_data + '/data1.csv'  # 2020-2021: train set
    # file1 = folder_data + '/data2.csv'  # 2021: test set

    import pandas as pd
    from functions import my_plot, my_plot_dt, my_subplot, my_scatter_plot, my_boxplots
    from functions import my_boxplots_individual

    # 2020-2021: train set
    df0 = pd.read_csv(file0)
    df0.columns
    df0.describe
    df0.describe()

    # combining
    frames = [df0]
    df10 = pd.concat(frames)
    df10.reset_index(inplace=True, drop=True) # reset index to start in 0

    df10.max()

    # plot LWS
    df10.columns
    df10['timestamp_utc'] = pd.to_datetime(df10['timestamp_utc'])
    my_plot_dt(df=df10, index='timestamp_utc', cols=range(1,3), file_name=folder_output + '/' + name + '_all')

    # identifying NAs
    df10.describe()
    df10[df10['FWD (C/N)'].isnull()].index.tolist()
    df10[df10['rain_intensity_rg'].isnull()].index.tolist()
    df1 = df10.dropna()
    df1.describe()
    df1.reset_index(inplace=True, drop=True) # reset index to start in 0
    df1.columns

    df1[df1['FWD (C/N)'].isnull()].index.tolist()
    df1[df1['rain_intensity_rg'].isnull()].index.tolist()

    my_plot_dt(df=df1, index='timestamp_utc', cols=range(1,3), file_name=folder_output + '/' + name + '_noNAs')


    # define and plot inputs
    my_subplot(df=df1, file_name=folder_output + '/' + name + '_inputs_noNAs')
    my_subplot(df=df10, file_name=folder_output + '/' + name + '_inputs_all')

    df10_nodate = df10.drop('timestamp_utc', axis=1)
    print(df10_nodate.head(5))
    print(df10_nodate.shape)
    my_plot(df_nodate=df10_nodate, cols=range(0,2), file_name=folder_output + '/' + name + '_inputs_all_nodate')

    df10_nodate.describe()
    my_boxplots_individual(df=df10_nodate,
                           xlabel='', ylabel='',
                           folder_output=folder_output,
                           name=name + '_all')

    # no NAs
    df1_nodate = df1.drop('timestamp_utc', axis=1)
    print(df1_nodate.head(5))
    print(df1_nodate.shape)
    my_plot(df_nodate=df1_nodate, cols=range(0,2), file_name=folder_output + '/' + name + 'inputs_noNAs_nodate')

    df1_nodate.describe()
    my_boxplots_individual(df=df1_nodate,
                           xlabel='', ylabel='',
                           folder_output=folder_output,
                           name=name + '_noNAs')

    # Creating the target series
    df1.columns
    y = df1["rain_intensity_rg"]
    y.reset_index(inplace = True, drop = True) # reset index to start in 0
    print(y.shape)
    my_plot(df_nodate=pd.DataFrame(y), cols=range(0), file_name=folder_output + '/' + name + '_noNAs_y')

    # # exporting csv files (site 2)
    # df10.columns
    # df10_site2.columns
    #
    # cols_site2 = ['Date', 'LWS-2-3', 'LWS-2-4', 'ASTA_LWS', 'LWS-2_Temp', 'ASTA_TAS-2m']
    # csv_site2 = pd.concat([df10[cols_site2], df10_site2['LWS-2-34_mean']], axis=1, ignore_index=True)
    # csv_site2.columns = ['Date', 'LWS-2-3', 'LWS-2-4', 'ASTA_LWS', 'LWS-2_Temp', 'ASTA_TAS-2m', 'LWS-2-34_mean']
    # csv_site2 = csv_site2[['Date', 'LWS-2-34_mean', 'ASTA_LWS', 'LWS-2_Temp', 'ASTA_TAS-2m']]
    #
    # csv_site2.to_csv(folder_output + '/LWS_Temp_Site2.csv', index=False)

    # scatter plots
    my_scatter_plot(y=df1[['FWD (C/N)']],
                    x=df1[['rain_intensity_rg']],
                    ylabel='FWD (C/N)',
                    xlabel='rain_intensity_rg',
                    folder_output=folder_output,
                    name=name + '_noNAs')

    ## df wet
    df1.columns
    df1_wet = df1[df1['rain_intensity_rg'] != 0.00000]
    df1_wet.reset_index(inplace=True, drop=True) # reset index to start in 0

    df1_wet_nodate = df1_wet.drop('timestamp_utc', axis=1)
    print(df1_wet_nodate.head(5))
    print(df1_wet_nodate.shape)
    my_plot(df_nodate=df1_wet_nodate, cols=range(0,2), file_name=folder_output + '/' + name + 'inputs_noNAs_dry_nodate')

    ## MinMaxScaler
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    df = pd.DataFrame(scaler.fit_transform(df1_wet_nodate[['FWD (C/N)']]), columns=['FWD (C/N)'])
    print(df)

    ## target
    y = df1_wet_nodate[['rain_intensity_rg']]
    print(y)

    my_scatter_plot(y=df,
                    x=y,
                    ylabel='FWD (C/N) -- scaled',
                    xlabel='rain_intensity_rg',
                    folder_output=folder_output,
                    name=name + '_noNAs_wet_scaled')

    print(name + ' loading  and preprocessing... done!')

    return [df1, df1_nodate, df1_wet, df1_wet_nodate, df, y]