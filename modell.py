import pickle
import numpy as np
import pandas as pd
#from sklearn.externals import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import KFold, cross_validate

data = pd.read_csv('blood.data', sep=',')
data.head()

# AMBIL LABEL SETIAP BARIS
label = data.iloc[:,-1:].values.ravel()
label

# AMBIL FEATURE DATA
feature = data.iloc[:,:-1]
feature.values

# PEMBAGIAN DATA LATIH DAN DATA UJI SEBANYAK 80% UNTUK DATA LATIH 20% UNTUK DATA TEST
X_train, X_test, y_train, y_test = train_test_split(feature, label ,test_size=0.3)

print("Jumlah Data Latih:", len(X_train))
print("Jumlah Data Uji:", len(X_test))

# PEMODELAN MENGGUNAKAN KNN
clf = KNeighborsClassifier(n_neighbors=3)

# FITTING DATA TRAINING
clf.fit(X_train, y_train)

pred = clf.predict(X_test)
pred

cols = {
    'Y True' : y_test,
    'Predict' : pred,
}

print("Prediksi Yang Dihasilkan")
pd.DataFrame(cols)

# FOLD = 10
kfold = KFold(n_splits=10)
scores = ['accuracy', 'precision', 'recall']
results = cross_validate(clf, X_test, y_test, cv=kfold, scoring=scores, return_train_score=True)
results_data = pd.DataFrame(results)
results_data

print("PERFORMANCE MODEL KNN")

accuracy = results_data.test_accuracy.mean() * 100
precision = results_data.test_precision.mean() * 100
recall = results_data.test_recall.mean() * 100

print("Accuracy : %0.2f" % accuracy, "%")
print("Precision : %0.2f " % precision, "%")
print("Recall : %0.2f " % recall, "%")

pickle.dump(clf, open('model_new2.pkl','wb'))