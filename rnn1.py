# RNN and GRU algorithms and helper functions
# J. A. Torres-Matallana
# since 2023-07-03 to 2023-07-04

# Preparing the sequence data
def my_rnn_pre(df, y1, n_future, n_past_par, total_period_par):
    import numpy as np

    xlist = list(df['FWD (C/N)'])
    ylist = list(y1['rain_intensity_rg'])

    n_past = n_past_par * n_future
    total_period = total_period_par * n_future

    idx_end = len(ylist)
    idx_start = idx_end - total_period
    X_new = []
    y_new = []

    while idx_start > 0:
        x_line = xlist[idx_start:idx_start + n_past]
        y_line = ylist[idx_start + n_past:idx_start + total_period]

        X_new.append(x_line)
        y_new.append(y_line)

        idx_start = idx_start - 1

    # converting list of lists to numpy array
    X_new = np.array(X_new)
    y_new = np.array(y_new)

    return [X_new, y_new]


# RNN using simpleRNN and GRU
def my_rnn_gru(df, y1, folder_output, learn_rate, n_future, n_past_par, total_period_par,
               validation_split_par=0.5, epochs_par=15, batch_size_par=24, case_par='gru1'):

    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    import random
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, SimpleRNN
    import keras.optimizers
    from sklearn.metrics import r2_score
    import matplotlib.pyplot as plt
    from tensorflow.keras.layers import GRU
    from functions import my_plot, my_scatter_plot

    # feature scaled: df
    df.columns
    df.describe()
    df.shape

    # target
    y1.columns
    y1.describe()
    y1.shape
    # ylist = list(df1_wet_nodate['rain_intensity_rg'])

    X_new, y_new = my_rnn_pre(df, y1, n_future, n_past_par, total_period_par)

    # Splitting into train and test
    X_train, X_test, y_train, y_test = train_test_split(X_new, y_new, test_size=0.33, random_state=42)

    # # Splitting into train and test (ordered samples)
    # split_pcent = 0.70
    # split_train = int(round(X_new.shape[0]*split_pcent, 0))
    # X_new.shape[0]
    #
    # X_train = X_new[:split_train, :] # if non PCA is used
    # # X_train = X_new.iloc[:split_train]  # if non PCA is used
    #
    # X_test = X_new[split_train + 1:]
    # y_train = y_new[:split_train]
    # y_test = y_new[split_train + 1:]

    # Reshape the data to be recognized by Keras
    n_samples = X_train.shape[0]
    n_timesteps = X_train.shape[1]
    n_steps = y_train.shape[1]
    n_features = 1
    X_train_rs = X_train.reshape(n_samples, n_timesteps, n_features )
    X_test_rs = X_test.reshape(X_test.shape[0], n_timesteps, n_features )

    if(case_par == 'simple_rnn1'):
        # A Simple SimpleRNN
        # Parameterize a small network with SimpleRNN
        random.seed(42)

        simple_model = Sequential([
            SimpleRNN(8, activation='tanh',input_shape=(n_timesteps, n_features)),
            Dense(y_train.shape[1]),
        ])

    elif (case_par == 'rnn1'):
        random.seed(42)

        simple_model = Sequential([
            SimpleRNN(32, activation='tanh',input_shape=(n_timesteps, n_features), return_sequences=True),
            SimpleRNN(32, activation='tanh', return_sequences = True),
            SimpleRNN(32, activation='tanh'),
            Dense(y_train.shape[1]),
        ])

    elif(case_par == 'simple_gru1'):
        # A simple architecture with one GRU layer
        random.seed(42)

        simple_model = Sequential([
            GRU(8, activation='tanh', input_shape=(n_timesteps, n_features)),
            Dense(y_train.shape[1]),
        ])

    elif (case_par == 'simple_gru2'):
        # A simple architecture with one GRU layer
        random.seed(42)

        simple_model = Sequential([
            GRU(24, activation='tanh', input_shape=(n_timesteps, n_features), return_sequences=True),
            GRU(24, activation='tanh', return_sequences=True),
            GRU(24, activation='tanh'),
            Dense(y_train.shape[1]),
        ])

    elif (case_par == 'gru1'):
        random.seed(42)

        simple_model = Sequential([
            GRU(24 * 4, activation='tanh', input_shape=(n_timesteps, n_features), return_sequences=True),
            GRU(24 * 3, activation='tanh', return_sequences=True),
            GRU(24 * 1, activation='tanh'),
            Dense(y_train.shape[1]),
        ])

    elif (case_par == 'gru2'):
        random.seed(42)

        simple_model = Sequential([
            GRU(24 * 4, activation='tanh', input_shape=(n_timesteps, n_features), return_sequences=True),
            GRU(24 * 5, activation='tanh', return_sequences=True),
            GRU(24 * 5, activation='tanh', return_sequences=True),
            GRU(24 * 5, activation='tanh', return_sequences=True),
            GRU(24 * 5, activation='tanh', return_sequences=True),
            GRU(24 * 5, activation='tanh', return_sequences=True),
            GRU(24 * 5, activation='tanh', return_sequences=True),
            GRU(24 * 5, activation='tanh', return_sequences=True),
            GRU(24 * 5, activation='tanh', return_sequences=True),
            GRU(24 * 5, activation='tanh', return_sequences=True),
            GRU(24 * 5, activation='tanh', return_sequences=True),
            GRU(24 * 5, activation='tanh', return_sequences=True),
            GRU(24 * 5, activation='tanh', return_sequences=True),
            GRU(24 * 2, activation='tanh', return_sequences=True),
            GRU(24 * 1, activation='tanh'),
            Dense(y_train.shape[1]),
        ])

    simple_model.summary()

    simple_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learn_rate),
        loss='mean_absolute_error',
        metrics=['mean_absolute_error'],
    )

    smod_history = simple_model.fit(X_train_rs, y_train,
                                    validation_split=validation_split_par,
                                    epochs=epochs_par,
                                    batch_size=batch_size_par,
                                    shuffle=True
                                    )

    preds_train = simple_model.predict(X_train_rs)
    print(r2_score(y_train, preds_train))

    preds = simple_model.predict(X_test_rs)
    print(r2_score(y_test, preds))

    name = 'prediction_scaled_data1_' + case_par
    f = plt.figure()
    f.set_figwidth(11.69)
    f.set_figheight(8.27)
    plt.plot(smod_history.history['loss'])
    plt.plot(smod_history.history['val_loss'])
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper right')
    # plt.show()
    plt.savefig(folder_output + '/' + name + '_training_history.pdf')
    plt.close()

    # training set plot
    df1_train_gru = pd.concat([pd.DataFrame(preds_train).iloc[:,0], pd.DataFrame(y_train).iloc[:,0]], axis=1)
    df1_train_gru.columns = ['Predicted', 'Observed']

    my_plot(df_nodate=df1_train_gru, cols=range(0,2), file_name=folder_output + '/' + name + '_training')
    print(r2_score(df1_train_gru.iloc[:,1], df1_train_gru.iloc[:,0]))

    my_scatter_plot(x=df1_train_gru.iloc[:,1], # obs
                    y=df1_train_gru.iloc[:,0], # pred
                    xlabel='Observed',
                    ylabel='Predicted',
                    folder_output=folder_output,
                    name=name + '_training')

    # testing set plot
    df1_test_gru = pd.concat([pd.DataFrame(preds).iloc[:, 0], pd.DataFrame(y_test).iloc[:, 0]], axis=1)
    df1_test_gru.columns = ['Predicted', 'Observed']

    my_plot(df_nodate=df1_test_gru, cols=range(0, 2), file_name=folder_output + '/' + name + '_testing')
    print(r2_score(df1_test_gru.iloc[:, 1], df1_test_gru.iloc[:, 0]))

    my_scatter_plot(x=df1_test_gru.iloc[:, 1],  # obs
                    y=df1_test_gru.iloc[:, 0],  # pred
                    xlabel='Observed',
                    ylabel='Predicted',
                    folder_output=folder_output,
                    name=name + '_testing')

    # save model
    simple_model.save(folder_output + '/' + name + '_model_save')

    return[simple_model, preds_train, y_train, y_new, y1]

def my_rnn_gru_validation(model, df, y1, n_future, n_past_par, total_period_par, folder_output, name):
    from functions import my_plot, my_scatter_plot
    from sklearn.metrics import r2_score
    import pandas as pd

    X_new, y_new = my_rnn_pre(df, y1, n_future, n_past_par, total_period_par)

    # Reshape the data to be recognized by Keras
    n_samples = X_new.shape[0]
    n_timesteps = X_new.shape[1]
    n_steps = y_new.shape[1]
    n_features = 1
    X_new_rs = X_new.reshape(n_samples, n_timesteps, n_features)

    preds = model.predict(X_new_rs)
    #print(r2_score(y1, preds))

    # plot and check
    #df_gru = pd.concat([pd.DataFrame(preds).iloc[:, 0], pd.DataFrame(y_new).iloc[:, 0]], axis=1)
    df_gru = pd.concat([pd.DataFrame(preds), pd.DataFrame(y_new)], axis=1)

    df_gru.columns = ['Predicted', 'Observed']

    my_plot(df_nodate=df_gru, cols=range(0, 2), file_name=folder_output + '/' + name)
    print(r2_score(df_gru.iloc[:, 1], df_gru.iloc[:, 0]))

    my_scatter_plot(x=df_gru.iloc[:, 1],  # obs
                    y=df_gru.iloc[:, 0],  # pred
                    xlabel='Observed',
                    ylabel='Predicted',
                    folder_output=folder_output,
                    name=name)
    return[preds, y_new]