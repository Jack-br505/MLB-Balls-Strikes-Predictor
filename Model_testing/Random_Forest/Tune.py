from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import pandas as pd
import numpy as np
#Load in the processed data
player_train_features = pd.read_csv('Processed_Data/train_player_features.csv')
player_test_features = pd.read_csv('Processed_Data/test_player_features.csv')

no_player_train_features = pd.read_csv('Processed_Data/train_no_player_features.csv')
no_player_test_features = pd.read_csv('Processed_Data/test_no_player_features.csv')

train_labels = pd.read_csv('Processed_Data/train_labels.csv')
test_labels = pd.read_csv('Processed_Data/test_labels.csv')


# Flatten labels to 1D (required by sklearn)
train_labels = train_labels.values.ravel()  
test_labels = test_labels.values.ravel()    

#Tune for the player model

#Create param grid for hyperparameter tuning
param_grid = {
    'n_estimators' : [200, 300, 400, 500],
    'max_depth': [None, 5, 7, 10, 15, 20, 25],
    'min_samples_split' : [4, 5, 6, 7, 8, 9],
    'min_samples_leaf' : [1, 2, 3, 4],
    'bootstrap' : [True, False],
    'criterion' : ['gini', 'entropy'],
    'class_weight' : ['balanced']

}

# Instantiate the Random Forest classifier
rf = RandomForestClassifier(random_state=42)

# Create RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_grid,
    n_iter=10,  # Number of parameter settings sampled
    cv=5,  # 5-fold cross-validation
    scoring='balanced_accuracy',  # Use F1 score for evaluation (good for binary classification)
    n_jobs=-1,  # Use all available cores
    random_state=42,
    verbose=1
)
#Fit model
random_search.fit(player_train_features, train_labels)

#Print best params
print("Best Params", random_search.best_params_)

best_model = random_search.best_estimator_
y_pred_best = best_model.predict(player_test_features)

#Print accuracy of best fit
print("Best fit Accuracy", accuracy_score(test_labels, y_pred_best))

#Save model if better than a previously found one using joblib
import joblib

curr_model = joblib.load('Model_testing/Random_Forest/final_model.pkl')
y_pred_curr = curr_model.predict(player_test_features)

curr_accuracy = accuracy_score(test_labels, y_pred_curr)
new_accuracy = accuracy_score(test_labels, y_pred_best)

if new_accuracy > curr_accuracy:
    joblib.dump(best_model, 'Model_testing/Random_Forest/final_model.pkl')
    print("New params saved to final_model.pkl")
else:
    print("Final model is not changed; \n Final model Accuracy is:", curr_accuracy)

print(len(y_pred_curr[y_pred_curr == 0]) / len(y_pred_curr))

print(y_pred_best)