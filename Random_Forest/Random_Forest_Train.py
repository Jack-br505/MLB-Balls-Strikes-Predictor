from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import pandas as pd

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

#Create weighting because more strikes in dataset
custom_weights = {0: 1.15, 1:1}

#Player Model
print("Player data included")

#Create model with some basic values
player_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=5,
    max_features="sqrt",
    class_weight= "balanced", #Balanced is better than custom weights
    random_state=42,
    n_jobs=-1
)
#Fit model
player_model.fit(player_train_features, train_labels) 

#Find predictions
test_pred = player_model.predict(player_test_features)


print(f"Accuracy Score: {accuracy_score(test_labels, test_pred)}")
print("\n", classification_report(test_labels, test_pred))
print("ROC-AUC Score: ", roc_auc_score(test_labels, test_pred))

# Feature importance for player model
player_importances = player_model.feature_importances_
player_feature_names = player_train_features.columns
player_importance_df = pd.DataFrame({
    'Feature': player_feature_names,
    'Importance': player_importances
}).sort_values(by='Importance', ascending=False)
print("\nPlayer Model Feature Importances:")
print(player_importance_df)

#Switch to the no player data model
print("\n \n No Player data included")

#Create model with some basic values
base_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=5,
    max_features="sqrt",
    class_weight= "balanced", #Balanced is better than custom weights
    random_state=42,
    n_jobs=-1
)
#Fit model
base_model.fit(no_player_train_features, train_labels) 

#Find predictions
test_pred = base_model.predict(no_player_test_features)


print(f"Accuracy Score: {accuracy_score(test_labels, test_pred)}")
print("\n", classification_report(test_labels, test_pred))
print("ROC-AUC Score: ", roc_auc_score(test_labels, test_pred))

# Feature importance for base model
base_importances = base_model.feature_importances_
base_feature_names = no_player_train_features.columns
base_importance_df = pd.DataFrame({
    'Feature': base_feature_names,
    'Importance': base_importances
}).sort_values(by='Importance', ascending=False)
print("\nBase Model Feature Importances:")
print(base_importance_df)