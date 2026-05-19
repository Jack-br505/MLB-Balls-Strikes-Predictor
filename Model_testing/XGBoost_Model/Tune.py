from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import pandas as pd
import xgboost as xgb
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

# Parameter grid for XGBoost GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 200, 300, 400],
    'max_depth': [3, 5, 7, 9],
    'learning_rate': [0.01, 0.05, 0.1, 0.15],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
    'min_child_weight': [1, 3, 5],
    'gamma': [0, 1, 5],
    'scale_pos_weight': [0.694, 1] #0.694 is balanced weights
}

classifier = xgb.XGBClassifier(
     objective='binary:logistic',
    random_state=42
)
#Tune for the player model
random_search = RandomizedSearchCV(
    estimator=classifier,
    param_distributions=param_grid,
    n_iter=10,  # Number of parameter settings sampled
    cv=5,  # 5-fold cross-validation
    scoring='f1',  # Use F1 score for evaluation (good for binary classification)
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

curr_model = joblib.load('Model_testing/XGBoost_Model/final_model.pkl')
y_pred_curr = curr_model.predict(player_test_features)

curr_accuracy = accuracy_score(test_labels, y_pred_curr)
new_accuracy = accuracy_score(test_labels, y_pred_best)

if new_accuracy > curr_accuracy:
    joblib.dump(best_model, 'Model_testing/XGBoost_Model/final_model.pkl')
    print("New params saved to final_model.pkl")
else:
    print("Final model is not changed; \n Final model Accuracy is:", curr_accuracy)

print("Mean y_pred", np.mean(y_pred_best))