# RandomFOrest algorithms
# J. A. Torres-Matallana
# since 2023-07-03 to 2023-07-04

from randomForest import my_rforest
from functions import my_wetdry

df1_wetdry = my_wetdry(df1, col_rain='rain_intensity_rg', rain_thresh=0.00000)

df1_wetdry.columns
df1_X_lagged = lagged(df=pd.DataFrame(df1_wetdry['FWD (C/N)']), par_lags=lags)

rf1 = my_rforest(X = df1_X_lagged, y = df1_wetdry['wet'].loc[max(lags):])
rf1