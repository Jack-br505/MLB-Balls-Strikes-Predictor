import pandas as pd
import xgboost as xgb


player_train_features = pd.read_csv('Processed_Data/train_player_features.csv')
player_test_features = pd.read_csv('Processed_Data/test_player_features.csv')

no_player_train_features = pd.read_csv('Processed_Data/train_no_player_features.csv')
no_player_test_features = pd.read_csv('Processed_Data/test_no_player_features.csv')

train_labels = pd.read_csv('Processed_Data/train_labels.csv')
test_labels = pd.read_csv('Processed_Data/test_labels.csv')

player_model = xgb.XGBClassifier(
    objective='binary:logistic',
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=0.694,  # To balance classes (3663 zeros / 5278 ones)
    random_state=42
)

# Flatten labels to 1D (required by sklearn/XGBoost)
train_labels = train_labels.values.ravel()
test_labels = test_labels.values.ravel()

# Fit model
player_model.fit(player_train_features, train_labels)

# Find predictions
test_pred = player_model.predict(player_test_features)

# Evaluate
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
print(f"Accuracy Score: {accuracy_score(test_labels, test_pred)}")
print("\n", classification_report(test_labels, test_pred))
print("ROC-AUC Score: ", roc_auc_score(test_labels, test_pred))

# Feature importance
importances = player_model.feature_importances_
feature_names = player_train_features.columns
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)
print("\nFeature Importances:")
print(importance_df)