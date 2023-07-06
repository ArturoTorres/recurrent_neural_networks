# ANN algorithms - Validation
# J. A. Torres-Matallana
# since 2023-07-03 to 2023-07-04

def validation(model, X, y, folder_output, name):
    from sklearn.metrics import r2_score
    from functions import my_scatter_plot

    preds = model.predict(X)

    r2_score_preds = r2_score(y, preds)

    print('preds validation = ' + str(r2_score_preds))


    my_scatter_plot(y=preds,
                    x=y,
                    ylabel='Predicted (Validation)',
                    xlabel='Observed',
                    folder_output=folder_output,
                    name=name+"_validation")