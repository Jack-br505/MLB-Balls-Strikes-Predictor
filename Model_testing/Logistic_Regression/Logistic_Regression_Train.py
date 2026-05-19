import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression


player_train_features = pd.read_csv('Processed_Data/train_player_features.csv')
player_test_features = pd.read_csv('Processed_Data/test_player_features.csv')

no_player_train_features = pd.read_csv('Processed_Data/train_no_player_features.csv')
no_player_test_features = pd.read_csv('Processed_Data/test_no_player_features.csv')

train_labels = pd.read_csv('Processed_Data/train_labels.csv')
test_labels = pd.read_csv('Processed_Data/test_labels.csv')

# Flatten labels to 1D (required by sklearn)

train_labels = train_labels.values.ravel()  
test_labels = test_labels.values.ravel()    

#Player Model

model = LogisticRegression(max_iter = 1000, class_weight= 'balanced')

results = model.fit(player_train_features, train_labels)

from sklearn.metrics import accuracy_score, classification_report

y_pred = results.predict(player_test_features)

print("Accuracy:", accuracy_score(test_labels, y_pred))

print("Classification report \n", classification_report(test_labels, y_pred))