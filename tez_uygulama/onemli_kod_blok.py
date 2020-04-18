#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 13:50:22 2020

@author: sdrttnclskn
"""
import matplotlib.pyplot as plt
import warnings
#Sonuçları değerlendirmek için gerekli fonksiyonlar
from sklearn.metrics import mean_squared_error, accuracy_score

# =============================================================================
# # Aşağıdaki iki satır uyarı mesajlarını kapatmaya yarıyor
# 
# =============================================================================

warnings.simplefilter("ignore")

# =============================================================================
#   
# Örüntüyü görmek için bir serpme (scatter) grafiği kullanalım.
#  
# =============================================================================

plt.figure(figsize=(8,8))
plt.scatter(X_train,y_train)
plt.show()


# =============================================================================
# mavi noktalar öğrenme verisini, kırmızı noktalar sınama verisini ve yeşil çizgi de elde ettiğimiz
# modelin çıktısını temsil ediyor.
# =============================================================================


#0 ile 10 arasındaki değerler için modelin sonuçlarını elde edelim.
line_X = np.arange(0,11, 0.1)
line_y = lr.predict(line_X.reshape(-1,1))

plt.figure(figsize=(8,8))
plt.scatter(X_train,y_train, alpha = 0.5, label = 'Öğrenme')
plt.scatter(X_test, y_test, c='red', alpha = 0.7, label = 'Sınama')
plt.plot(line_X, line_y, c='green', linewidth=3, alpha=0.6, label = 'Doğrusal Bağlanım')
plt.legend(loc = 4)
plt.show()