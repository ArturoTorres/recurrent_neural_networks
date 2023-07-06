# ANN algorithms
# J. A. Torres-Matallana
# since 2023-07-03 to 2023-07-04

def my_ann3_nodate(df, y, ann_model, par_epochs, par_batch_size, par_learning_rate, folder_output, name):

    import matplotlib.pyplot as plt
    from functions import my_scatter_plot
    import matplotlib.dates as mdates

    # The Neural Network Using Keras
    # ==========================================================================================
    # Now that we have made sure that our data are correctly prepared, we can finally move on
    # to the actual neural network.
    # Building Neural Networks is a lot of work, and I want to find a good balance in
    # showing you the way to get started and to work on improving your network rather than
    # just showing a final performant network.
    # A general great first start is to start with a relatively simple network and work your
    # way up from there. In this case, let’s start with a network using two dense layers with 64
    # nodes.
    #
    # For the other hyperparameters, let’s take things that are a little bit standard:
    # • Optimizer: Adam
    # • Learning rate: 0.01
    # • Batch size: 32 (reduce this if you don’t have enough RAM)
    # • Epochs: 10
    #
    # Before starting, let’s do a train-test split:
    # Train-test split

    from sklearn.model_selection import train_test_split
    # X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.33, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.33, random_state=42)

    ## aux. variables for plotting
    # X_train_plt, X_test_plt, y_train_plt, y_test_plt = train_test_split(df_date, y_date, test_size=0.33, random_state=42)

    split_pcent = 0.50
    split_train = int(round(df.shape[0]*split_pcent, 0))
    df.shape[0]

    X_train_ordered = df.iloc[:split_train, :] # if non PCA is used
    #X_train_ordered = df.iloc[:split_train]  # if non PCA is used

    X_test_ordered = df[split_train + 1:]
    y_train_ordered = y[:split_train]
    y_test_ordered = y[split_train + 1:]

    # # aux. variables for plotting
    # X_train_ordered_plt = df_date.iloc[:split_train, :]
    # X_test_ordered_plt = df_date[split_train + 1:]
    # y_train_ordered_plt = y_date[:split_train]
    # y_test_ordered_plt = y_date[split_train + 1:]

    # Now you build the model using the keras library using the following code. First, you
    # specify the architecture. Keras is the go-to library for neural networks
    # in Python.

    # Specify the model and its architecture
    ## import keras
    import tensorflow as tf

    tf.config.list_physical_devices("GPU")
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense

    import random
    random.seed(42)

    simple_model = Sequential([
        Dense(64, activation='selu', input_shape=(X_train_ordered.shape[1],)),
        #Dense(32, activation='selu', input_shape=(1,)),
        Dense(64, activation='selu'),
        Dense(64, activation='selu'),
        Dense(32, activation='selu'),
        Dense(1),
    ])

    # You can obtain a summary to check if everything is alright using:
    # Obtain a summary of the model architecture
    simple_model.summary()

    # Then you compile the model. In the compilation part, you specify
    # the optimizer     and the learning rate. You also specify the loss, in our case the Mean
    # Absolute Error.

    # Compile the model
    # install: keras-applications; keras-base
    import keras.optimizers

    simple_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=par_learning_rate),
        loss='mean_absolute_error',
        # loss='mean_squared_error',
        # loss='MeanSquaredLogarithmicError',
        # loss='MeanAbsolutePercentageError',
        metrics=['mean_absolute_error'],
    )
    # And then you fit the model using Listing 16-11. At the fitting call, you specify the
    # epochs and the batch size. You can also specify a validation split so that you obtain a
    # train-validation-test scenario in which you still have the test set for a final check of the R2
    # score that is not biased by the model development process.

    # Fit the model
    #smod_history = simple_model.fit(X_train, y_train,
    #smod_history=simple_model.fit(df_tr, y,
    smod_history = simple_model.fit(X_train_ordered, y_train_ordered,
                                    validation_split=0.2,
                                    epochs=par_epochs,
                                    batch_size=par_batch_size,
                                    shuffle = True
                                    )

    # Be aware that fitting neural networks can take a lot of time. Running on GPU is
    # generally fast but not always possible depending on your computer hardware.

    # Now the important part here is to figure out whether or not this model has learned
    # something using those hyperparameters. There is a key graph that is going to help you
    # infinitely while building neural networks.
    # Be aware that you may get slightly different results. Setting the random seed is not
    # enough to force randomness to be the same in Keras. Although it is possible to force
    # exact reproducibility in Keras, it is quite complex, so I prefer to leave it out and accept
    # that results are not 100% reproducible. You can check out these instructions for more
    # information on how to fix the randomness in Keras: https://keras.io/getting_
    # started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-
    # development.
    #
    # # Plot the training history
    f = plt.figure()
    f.set_figwidth(11.69)
    f.set_figheight(8.27)
    plt.plot(smod_history.history['loss'])
    plt.plot(smod_history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper right')
    # plt.show()
    plt.savefig(folder_output + str('/') + name + '_training_history_simple.pdf')
    plt.close()

    from sklearn.metrics import r2_score
    preds_simple_train = simple_model.predict(X_train_ordered) # df) # simple_model.predict(X_train)
    preds_simple = simple_model.predict(X_test_ordered)
    # y_test_s = pd.Series(y_test[:100])

    # plt.plot(preds_simple[1:500])
    # plt.plot(y_test.values[1:500])
    # plt.legend(['preds simple', 'obs'], loc='upper left')
    # plt.show()
    r2_score_simple_train = r2_score(y_train_ordered, preds_simple_train) #y) # y_train)
    print('preds simple_train = ' + str(r2_score_simple_train))

    r2_score_simple = r2_score(y_test_ordered, preds_simple)
    print('preds simple = ' + str(r2_score_simple))

    my_scatter_plot(y=preds_simple_train,
                    x=y_train_ordered,
                    ylabel='Predicted',
                    xlabel='Observed',
                    folder_output=folder_output,
                    name=name+"_simple_train")

    #x = y_date.loc[1:8654, "timestamp_utc"]
    #x = y_date.loc[0:26223, "timestamp_utc"]
    # x = y_train _date.loc[0:76, "timestamp_utc"]
    #
    # f = plt.figure()
    # f.set_figwidth(11.69*40)
    # f.set_figheight(8.27)
    # # plt.plot(x.loc[0:((preds1_filtered.shape[0])),], preds1_filtered, linewidth=0.5, label='Predictions')
    # plt.plot(x, preds_simple_train, linewidth=0.5, label='Predictions')
    # plt.plot(x, y_train_ordered, linewidth=1, label='Observations', color='red')
    # plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S+00:00'))
    # # plt.gca().xaxis.set_major_locator(mdates.DayLocator())
    # plt.gcf().autofmt_xdate()
    # plt.legend(loc='upper left')
    # plt.savefig(folder_output + '/' + name + '_simple_test_dt.pdf', bbox_inches='tight', pad_inches=0)
    # plt.close()


    f = plt.figure()
    f.set_figwidth(11.69*40)
    f.set_figheight(8.27)
    plt.plot(preds_simple_train, linewidth=0.5, label='Predictions')
    plt.plot(y_train_ordered.values, linewidth=1, label='Observations', color='red')
    plt.legend(loc='upper left')
    plt.savefig(folder_output + '/' + name + '.pdf', bbox_inches='tight', pad_inches=0)
    plt.close()

    # save model
    simple_model.save(folder_output + '/' + name + '_model_ann2')

    return [simple_model, r2_score_simple_train, r2_score_simple,
            preds_simple_train, y_train_ordered]