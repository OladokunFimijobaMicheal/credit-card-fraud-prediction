import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
import math

# loading the data
credit = pd.read_csv('creditcard.csv')
# variables creation
X = scale(credit.iloc[0:])
y = credit.iloc[:, -1]

classes = ['fraud', 'no fraud']

# traintest splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# model creation
model = KMeans(n_clusters=2, random_state=0)
# training model
model.fit(X_train)
# predictions
predictions = model.predict(X_test)

# label creation
labels = model.labels_
clustercount = np.bincount(labels)


for i in range(len(predictions)):
    print(classes[predictions[i]])



print('labels:', labels)
print('predictions:', predictions)
print('accuracy:', accuracy_score(y_test, predictions))
print('actual:', y_test)




















