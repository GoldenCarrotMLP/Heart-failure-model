import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils import shuffle

df = pd.read_csv('project/heart_failure_clinical_records_dataset.csv')




#----------------------- Pre-processing------------------------------------------------------------------------


X = df.values[:, 0:11]
y = df.values[:, 12]

#print('Any columns with null values')
#print(df.isna().sum())
#print(df.dtypes)

#print('Column \'Class\' Values Count:')


#print('X')
#print(X)
#print('Data (X) Size:',X.shape)

#print('Label (y) Size:',y.shape)




from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)
    

#------------------------------ Model: LogisticRegression------------------------------------------------------

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
#model = KNeighborsClassifier(n_neighbors=1)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print('y_pred: ', y_pred)
print('y_test')
print(y_test)

from sklearn.metrics import classification_report
print('Classification Report')
print(classification_report(y_test, y_pred))

"""
Classification Report
              precision    recall  f1-score   support

         0.0       0.75      0.97      0.85        40
         1.0       0.88      0.35      0.50        20

    accuracy                           0.77        60
   macro avg       0.81      0.66      0.67        60
weighted avg       0.79      0.77      0.73        60
"""
