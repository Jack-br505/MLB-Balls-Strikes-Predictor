# Clase
Machine Learning project aimed at predicting whether a pitch will be a ball or strike by using MLB data about the conditions at the time of the pitch.

Uses Pitch Data from the 2025 postseason, taken from Kaggle: https://www.kaggle.com/datasets/janus137/mlb-postseason-2025-pitch-by-pitch-data to get conditions of the at-bat such as outs, score, and baserunners. Takes 2025 regular season player data to determine different batting and pitching statistics of the players, from Baseball Savant. 

Looks at two different models, a model including player statistics and one without player's information

Mostly a learning project

- Jack Brach, Student at Michigan State University

## Model testing

Looked at 3 different models: random forest, adaboost and logistic regression to determine if the best model for this data

## Data Exploration

Jupyter notebook looking at various impacts of certain features by creating figures and such. Also looks at some information important for preprocessing the data. 

## Data

Raw data of player metrics and 2025 pitch data. Taken from google statcast.

## Processed data

Csv files of the data processed for the models use.

## Frontend

Very basic frontend using streamlit to live track MLB games using the statsapi package. Can make a custom scenario or tracj live games by inputting data and getting the pitch prediction.  The final model in the frontend uses the RandomForest Model as that was most accurate in testing. 