# RandomFOrest algorithms
# J. A. Torres-Matallana
# since 2023-07-03 to 2023-07-04

def my_rforest(X, y):
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import r2_score
    from functions import my_scatter_plot

    # Create Train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=12345, shuffle=False)

    # my_rf = RandomForestRegressor()
    my_rf = RandomForestClassifier(max_depth=13)

    my_rf.fit(X_train, y_train)
    pred_train = my_rf.predict(X_train)
    pred_test = my_rf.predict(X_test)

    r2_score_train = r2_score(list(y_train), list(pred_train))
    r2_score_test = r2_score(list(y_test), list(pred_test))

    # scatter plots
    my_scatter_plot(y=pred_train,
                    x=y_train,
                    xlabel='FWD (C/N) -- training',
                    ylabel='rain_intensity_rg -- trainig',
                    folder_output=folder_output,
                    name='data1_wetdry_training')

    my_scatter_plot(y=pred_test,
                    x=y_test,
                    ylabel='FWD (C/N) -- test',
                    xlabel='rain_intensity_rg -- test',
                    folder_output=folder_output,
                    name='data1_wetdry_test')

    return [r2_score_train, r2_score_test]