from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import pandas as pd
import joblib
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
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'l1_ratio' : [0],
    'solver': ['lbfgs', 'liblinear', 'saga'],
    'max_iter': [200, 500, 1000],
    'class_weight': [None, 'balanced']
}

# Instantiate the Logistic Regression classifier
lr = LogisticRegression(random_state=42)

# Create GridSearchCV
grid_search = GridSearchCV(
    estimator=lr,
    param_grid=param_grid,
    cv=5,  # 5-fold cross-validation
    scoring='f1',  # Use F1 score for evaluation (good for binary classification)
    n_jobs=-1,  # Use all available cores
    verbose=1
)

#Fit model
grid_search.fit(player_train_features, train_labels)

#Print best params
print("Best Params", grid_search.best_params_)

best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(player_test_features)

#Print accuracy of best fit
print("Best fit Accuracy", accuracy_score(test_labels, y_pred_best))

#Save model if better than a previously found one using joblib
try:
    curr_model = joblib.load('Model_testing/Logistic_Regression/final_model.pkl')
    y_pred_curr = curr_model.predict(player_test_features)

    curr_accuracy = accuracy_score(test_labels, y_pred_curr)
    new_accuracy = accuracy_score(test_labels, y_pred_best)

    if new_accuracy > curr_accuracy:
        joblib.dump(best_model, 'Model_testing/Logistic_Regression/final_model.pkl')
        print("New params saved to final_model.pkl")
    else:
        print("Final model is not changed; \n Final model Accuracy is:", curr_accuracy)
except:
    joblib.dump(best_model, 'Model_testing/Logistic_Regression/final_model.pkl')
    print("First model saved to final_model.pkl")

print("Mean of y_pred", np.mean(y_pred_curr))