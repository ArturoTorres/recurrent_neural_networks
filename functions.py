# Helper functions
# J. A. Torres-Matallana
# since 2023-07-03 to 2023-07-04

def xlsx2df(file):
    import pandas as pd

    xls0 = pd.ExcelFile(file)
    xls0_sname = xls0.sheet_names
    dict0 = xls0.parse(xls0_sname)
    df0 = dict0[xls0_sname[0]]
    print(df0.head(11))

    return df0

def my_plot_dt(df, index, cols, file_name):
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    plt.figure(figsize=(16*10, 9))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S+00:00'))
    # plt.gca().xaxis.set_major_locator(mdates.DayLocator())
    plt.plot(df.set_index(index), linewidth=0.5, label=df.columns[cols])
    plt.legend(loc='upper left')
    plt.gcf().autofmt_xdate()
    plt.savefig(file_name + '.pdf', dpi=175,
                bbox_inches='tight', pad_inches=0)

def my_plot(df_nodate, cols, file_name):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(16*10, 9))
    plt.plot(df_nodate, linewidth=0.5, label=df_nodate.columns[cols])
    plt.legend()
    plt.savefig(file_name + '.pdf', dpi=175,
                bbox_inches='tight', pad_inches=0)

def my_subplot(df, file_name):
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(16*10, 9*5))
    fig.suptitle('Ceutorhynchus pallidactylus -- First Migration -- Reuler', fontsize=120)

    def my_fonts(ax):
        ax.set_xlabel('Date', fontsize=120)
        ax.set_ylabel('Value', fontsize=120)
        ax.tick_params(axis='both', which='major', length=55, width=2, labelsize=100)
        ax.legend(fontsize=120)

    if df.shape[1] - 1 == 2:
        axis1 = fig.add_subplot(211)
        axis1.plot(df[df.columns[0]], df[df.columns[1]], linewidth=0.5, label=df.columns[1])
        my_fonts(axis1)

        axis2 = fig.add_subplot(212)
        axis2.plot(df[df.columns[0]], df[df.columns[2]], linewidth=0.5, label=df.columns[2])
        my_fonts(axis2)

    elif df.shape[1] - 1 == 3:
        axis1 = fig.add_subplot(311)
        axis1.plot(df[df.columns[0]], df[df.columns[1]], linewidth=0.5, label=df.columns[1])
        axis1.legend()

        axis2 = fig.add_subplot(312)
        axis2.plot(df[df.columns[0]], df[df.columns[2]], linewidth=0.5, label=df.columns[2])
        axis2.legend()

        axis3 = fig.add_subplot(313)
        axis3.plot(df[df.columns[0]], df[df.columns[3]], linewidth=0.5, label=df.columns[3])
        axis3.legend()

    elif df.shape[1]-1 == 8:
        axis1 = fig.add_subplot(811)
        axis1.plot(df[df.columns[0]], df[df.columns[1]], linewidth=0.5, label=df.columns[1])
        axis1.legend()

        axis2 = fig.add_subplot(812)
        axis2.plot(df[df.columns[0]], df[df.columns[2]], linewidth=0.5, label=df.columns[2])
        axis2.legend()

        axis3 = fig.add_subplot(813)
        axis3.plot(df[df.columns[0]], df[df.columns[3]], linewidth=0.5, label=df.columns[3])
        axis3.legend()

        axis4 = fig.add_subplot(814)
        axis4.plot(df[df.columns[0]], df[df.columns[4]], linewidth=0.5, label=df.columns[4])
        axis4.legend()

        axis5 = fig.add_subplot(815)
        axis5.plot(df[df.columns[0]], df[df.columns[5]], linewidth=0.5, label=df.columns[5])
        axis5.legend()

        axis6 = fig.add_subplot(816)
        axis6.plot(df[df.columns[0]], df[df.columns[6]], linewidth=0.5, label=df.columns[6])
        axis6.legend()

        axis7 = fig.add_subplot(817)
        axis7.plot(df[df.columns[0]], df[df.columns[7]], linewidth=0.5, label=df.columns[7])
        axis7.legend()

        axis8 = fig.add_subplot(818)
        axis8.plot(df[df.columns[0]], df[df.columns[8]], linewidth=0.5, label=df.columns[8])
        axis8.legend()

    fig.savefig(file_name + '.pdf', dpi=175,
                bbox_inches='tight', pad_inches=0)

# making values above threshold equal to value
def my_trunc(df, cols, threshold, value):
    for col in cols:
        for i in range(0, len(df.index), 1):
            if df.loc[i, col] > threshold:
                df.loc[i, col] = value

    return df

def my_dummy(x):
    axis8 = x+8
    return x, axis8

def my_scatter_plot(x, y, xlabel, ylabel, folder_output, name):
    from sklearn.metrics import r2_score
    import matplotlib.pyplot as plt

    r2 = r2_score(x, y) # x=true, y=predicted

    f = plt.figure()
    f.set_figwidth(11.7)
    f.set_figheight(11.7)
    plt.scatter(x, y)
    plt.xlabel(xlabel + '  (R2 = ' + str(round(r2, 3)) + ')', fontsize=22)
    plt.ylabel(ylabel, fontsize=22)
    plt.axline([0, 0], [1, 1], color='red', linewidth=2)
    plt.rcParams.update({'font.size': 22})
    plt.savefig(folder_output + '/' + name + '_scatter.pdf')
    plt.close()

def my_boxplots(df, cols2plot, labels, xlabel, ylabel, folder_output, name):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(16*3, 9))
    plt.boxplot(df.loc[:, cols2plot], labels=labels)
    plt.xlabel(xlabel, fontsize=22)
    plt.ylabel(ylabel, fontsize=22)
    plt.rcParams.update({'font.size': 22})
    plt.savefig(folder_output + '/' + name + '_boxplots.pdf',
                bbox_inches='tight', pad_inches=0)
    # plt.show()
    plt.close()

def my_boxplots_individual(df, xlabel, ylabel, folder_output, name):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 8)  # create figure and axes
    fig.set_figwidth(11.7*3)
    fig.set_figheight(11.7)

    for i, el in enumerate(list(df.columns.values)):
        a = df.boxplot(el, ax=axes.flatten()[i])
        a.grid(False)

    plt.xlabel(xlabel, fontsize=22)
    plt.ylabel(ylabel, fontsize=22)
    plt.rcParams.update({'font.size': 22})
    # fig.delaxes(axes[2, 4])  # remove empty subplot
    plt.savefig(folder_output + '/' + name + '_boxplots.pdf',
                bbox_inches='tight', pad_inches=0)
    # plt.show()
    plt.close()

def my_drop_datetime_duplicated(dt_ini, dt_end, df_1):
    import pandas as pd
    df_ref = pd.date_range(dt_ini, dt_end, freq='H')

    if len(df_1['Date']) != len(df_ref):
        # pd.DatetimeIndex.duplicated(df1['Date'], keep=False)
        df_1 = df_1.drop_duplicates(inplace=False, subset=['Date'])
        df_1 = df_1.reset_index(inplace=False, drop=True)  # reset index to start in 0

    return df_1

def my_validation_step1(file_asta, file_site):
    import pandas as pd

    df3_1 = pd.read_csv(file_asta, delimiter=";") # ASTA data
    df3_1['Date'] = df3_1["Tag"] + " " + df3_1["Stunde"]
    df3_1['Date'] = pd.Series(df3_1['Date'])
    df3_1['Date'] = pd.to_datetime(df3_1['Date'], format='%d.%m.%Y %H:%M')
    print(df3_1.head(-6))

    df3_2 = pd.read_csv(file_site, delimiter=",", header=1) # LWS data (station 2)
    print(df3_2.head(6))
    df3_2 = df3_2.loc[3:, ]
    df3_2.reset_index(inplace = True, drop = True) # reset index to start in 0
    df3_2["Date"] = pd.to_datetime(df3_2["TIMESTAMP"], format='%Y-%m-%d %H:%M:%S')
    print(df3_2.head(6))


    df3_1_ini = df3_1.loc[0, 'Date']
    df3_2_ini = df3_2.loc[0, 'Date']

    df3_1_end = df3_1.loc[df3_1.shape[0]-1, 'Date']
    df3_2_end = df3_2.loc[df3_2.shape[0]-1, 'Date']

    return [df3_1, df3_2, df3_1_ini, df3_2_ini, df3_1_end, df3_2_end]

def my_validation_step2(df3_1, df3_2, dt_ini, dt_end, folder_output, name):
    from functions import my_drop_datetime_duplicated
    from functions import my_subplot, my_plot_dt, my_scatter_plot, my_boxplots, my_trunc
    import pandas as pd

    df3_1 = df3_1[(df3_1['Date'] >= dt_ini) & (df3_1['Date'] <= dt_end)]
    df3_1.reset_index(inplace = True, drop = True) # reset index to start in 0
    df3_1.columns

    df3_2 = df3_2[(df3_2['Date'] >= dt_ini) & (df3_2['Date'] <= dt_end)]
    df3_2.reset_index(inplace = True, drop = True) # reset index to start in 0
    df3_2.columns

    df3_1 = my_drop_datetime_duplicated(dt_ini=dt_ini, dt_end=dt_end, df_1=df3_1)
    df3_2 = my_drop_datetime_duplicated(dt_ini=dt_ini, dt_end=dt_end, df_1=df3_2)

    df3 = pd.merge(df3_1, df3_2, on='Date', how='inner')
    nan_value = float("NaN")
    df3.replace("", nan_value, inplace=True)
    df3.columns
    df3.describe()

    for i in df3.columns[2:16]:
        df3[i] = [x.replace(',', '.') for x in df3[i]]
        df3[i] = df3[i].astype(float)

    df3['LWS_100_Avg'] = df3['LWS_100_Avg'].astype(float)
    df3['LWS_200_Avg'] = df3['LWS_200_Avg'].astype(float)
    df3['LWS_300_Avg'] = df3['LWS_300_Avg'].astype(float)
    df3['LWS_400_Avg'] = df3['LWS_400_Avg'].astype(float)
    df3['LWS_500_Avg'] = df3['LWS_500_Avg'].astype(float)

    # replace values greater than 100 by 100
    df3.max()
    cols2check100 =['AVG_LWET200',
                    'LWS_100_Avg', 'LWS_200_Avg',
                    'LWS_300_Avg', 'LWS_400_Avg',
                    'LWS_500_Avg',]

    df3 = my_trunc(df=df3, cols=cols2check100, threshold=100, value=100)
    df3.max()

    # x = df3.loc[:, "Date"]
    # f = plt.figure()
    # f.set_figwidth(11.69*8)
    # f.set_figheight(8.27*5)
    # plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y'))
    # plt.gca().xaxis.set_major_locator(mdates.DayLocator())
    # plt.plot(x, df3['AVG_LWET200'], linewidth=1, label='AVG_LWET200')
    # plt.plot(x, df3['LWS_300_Avg'], linewidth=0.5, label='LWS_300_Avg', color='red')
    # plt.plot(x, df3['LWS_400_Avg'], linewidth=0.5, label='LWS_400_Avg', color='green')
    # plt.gcf().autofmt_xdate()
    # plt.legend(loc='upper left')
    # plt.savefig(folder_output + '/target_site2_2022.pdf')
    # plt.close()

    df3.columns
    my_plot_dt(df=df3.loc[:, ['Date', 'AVG_LWET200', 'LWS_300_Avg', 'LWS_400_Avg']],
               index='Date', cols=list([1,2,3]),
               file_name=folder_output + '/target_' + name)

    # comparable variables
    # df1_set_inputs = df1[['Date', 'ASTA_LWS', 'ASTA_RH', 'ASTA_TAS-2m', 'ASTA_TAS-Bod',
    #                       'ASTA_WV', 'ASTA_Global', 'ASTA_Precip', 'ASTA_SD']]

    df3_set_inputs = df3[['Date', 'AVG_LWET200', 'AVG_RH200', 'AVG_TA200', 'AVG_TB005',
                          'AVG_WV1000', 'SUM_GS200', 'SUM_NN050', 'SUM_SSD']]

    my_subplot(df=df3_set_inputs, file_name=folder_output + '/set_inputs_2022')


    return [df3_set_inputs,
            df3.loc[:, ['Date', 'AVG_LWET200', 'LWS_300_Avg', 'LWS_400_Avg']]]

def my_pdf2png(folder):
    import os

    files = []
    for x in os.listdir(folder):
        if x.endswith(".pdf"):
            # Prints only text file present in My Folder
            files.append(x)

    for x in files:
        # x = files[0]
        basename = os.path.basename(x).split('.')[0]
        cmd2invoke = "convert -density 150 " + folder + "/" + x + " -quality 90 " + folder + "/" + basename + ".png"
        os.system(cmd2invoke)

def my_pdf2png_2(folder):
    from pdf2image import convert_from_path
    import os
    files = []
    for x in os.listdir(folder):
        if x.endswith(".pdf"):
            # Prints only text file present in My Folder
            files.append(x)

    for i in range(len(files)):
        # # Store Pdf with convert_from_path function
        basename = os.path.basename(files[i]).split('.')[0]
        image = convert_from_path(folder + "/" + files[i])

        # Save pages as images in the pdf
        image[0].save(folder + "/" + basename + '.png', 'PNG')


def lagged(df, par_lags):
    import pandas as pd

    lagged_x = []
    for lag in par_lags:
      lagged = df.shift(lag)
      lagged.columns = [x + '.lag' + str(lag) for x in lagged.columns]
      lagged_x.append(lagged)

    df = pd.concat(lagged_x, axis=1)
    df = df.iloc[max(par_lags):,:] # drop missing values due to lags
    df.columns

    return df


def my_wetdry(df1, col_rain='rain_intensity_rg', rain_thresh=0):
    import math
    import pandas as pd
    import numpy as np

    df_wetdry = (df1[col_rain]).apply(math.ceil)
    df_wetdry = pd.DataFrame(df_wetdry)
    df_wetdry.rename(columns={col_rain: 'wet'}, inplace=True)

    frames = [df1, pd.DataFrame(df_wetdry)]
    df1 = pd.concat(frames, axis=1)
    df1.reset_index(inplace=True, drop=True) # reset index to start in 0

    a = np.array(df1['wet'].values.tolist())
    print(a)

    df1['wet'] = np.where(a > rain_thresh, 1, 0).tolist()
    print(df1)
    df1.describe()
    print (df1)

    return df1