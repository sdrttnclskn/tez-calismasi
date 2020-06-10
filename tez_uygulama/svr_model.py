#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 19:23:30 2020

@author: sdrttnclskn
"""

#### Kutuphane ####
from warnings import filterwarnings
filterwarnings('ignore')
import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, cross_val_score,GridSearchCV, cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score,mean_absolute_error
import matplotlib.pyplot as plt
import numpy as np
from sklearn import model_selection
from sklearn.preprocessing import scale 
from sklearn.preprocessing import StandardScaler


#### Veri #####
veriseti = pd.read_csv("/Users/sdrttnclskn/Desktop/tez/tez-calismasi/tez_uygulama/temiz_veri_seti.csv")
df = veriseti.copy()
#tanimsız olan boş kolon drop edildi.
df.drop(df.columns[[0]], axis=1, inplace=True)
df.info()
df.head


## SVR ile Model Gelistirme

### 1. Model

df_svr = df.copy()
X_s = df_svr.drop(['TARIH','FONTIP','FONTUR','FON','FONFIYAT'], axis = 1)
y_s = df_svr["FONFIYAT"]

X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_s, y_s, test_size=0.25, random_state=42)

X_s.head()


##veri seti standartlastırma islemi

scaler = StandardScaler()
scaler.fit(X_train_s)

X_train_scaled_s = scaler.transform(X_train_s)
X_test_scaled_s = scaler.transform(X_test_s)

svr_rbf = SVR("rbf").fit(X_train_scaled_s, y_train_s)

### 2. Tahmin

y_pred_s = svr_rbf.predict(X_train_scaled_s)

## eğitim seti ölçüm metrikleri

print("MSE : " + str(mean_squared_error(y_train_s, y_pred_s)))
print("RMSE : " + str(np.sqrt(mean_squared_error(y_train_s, y_pred_s))))
print("MAE : " + str(mean_absolute_error(y_train_s, y_pred_s)))
print("R2 : " + str(r2_score(y_train_s, y_pred_s)))

y_pred_s = svr_rbf.predict(X_test_scaled_s)

##test seti ölçüm metrikleri

print("MSE : " + str(mean_squared_error(y_test_s, y_pred_s)))
print("RMSE : " + str(np.sqrt(mean_squared_error(y_test_s, y_pred_s))))
print("MAE : " + str(mean_absolute_error(y_test_s, y_pred_s)))
print("R2 : " + str(r2_score(y_test_s, y_pred_s)))


### 3. Model Tuning

svr_rbf

#'C': [0.1, 1, 10, 100, 1000]
svr_params = {"C": np.arange(0.1,2,0.1)}
svr_cv_model = GridSearchCV(svr_rbf,svr_params, cv = 10)
svr_cv_model.fit(X_train_scaled_s, y_train_s)

pd.Series(svr_cv_model.best_params_)[0]

svr_tuned = SVR("rbf", C = pd.Series(svr_cv_model.best_params_)[0]).fit(X_train_scaled_s, y_train_s)

y_pred_s = svr_tuned.predict(X_test_scaled_s)


##tuning ölçüm metrikleri

print("MSE : " + str(mean_squared_error(y_test_s, y_pred_s)))
print("RMSE : " + str(np.sqrt(mean_squared_error(y_test_s, y_pred_s))))
print("MAE : " + str(mean_absolute_error(y_test_s, y_pred_s)))
print("R2 : " + str(r2_score(y_test_s, y_pred_s)))






