import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
#load data
pitches = pd.read_csv("Data/Data_MLB_2025_StatcastPostseason_PitchByPitch_20251102a.csv")

#Convert outcome string to a numeric value
pitch_outcome_num = []
for i in range(pitches.shape[0]):
    if pitches.loc[i, 'description'] in ['called_strike', 'foul', 'swinging_strike', 'foul_tip']: #All of these are counting as strikes
        pitch_outcome_num.append(1)
    elif pitches.loc[i, 'description'] == 'ball':
        pitch_outcome_num.append(0)
    else:
        pitch_outcome_num.append(2)
#Add numeric outcome to data
pitches['outcome'] = pitch_outcome_num


#Replace baserunner ID, with a 1 or 0, for if a runners on or not
#Turn all NA to 0
for col in ['on_1b', 'on_2b', 'on_3b']:
    pitches[col] = pd.to_numeric(pitches[col], errors='coerce').fillna(0).astype(int)
for i in range(pitches.shape[0]):
    if pitches.loc[i, 'on_3b'] != 0:
        pitches.loc[i, 'on_3b'] = 1.0
    if pitches.loc[i, 'on_2b'] != 0:
        pitches.loc[i, 'on_2b'] = 1.0
    if pitches.loc[i, 'on_1b'] != 0:
        pitches.loc[i, 'on_1b'] = 1.0


#Feature engineering
#Make the score a differential instead of straight score
pitches['score_diff'] = pitches['bat_score'] - pitches['fld_score']

#Create variable for the advantage of pitcher / batter
pitches['count_advantage'] = pitches['balls'] - pitches['strikes']

#Or can make count seperate strings
pitches['count'] = ''
for i in range(len(pitches)):
    count = str(pitches.loc[i, 'balls']) + ', ' + str(pitches.loc[i, 'strikes'])
    pitches.loc[i, 'count'] = count
#Create variable for handedness matchup, 0 for same, 1 for different
pitches["Handedness"] = np.where(pitches['stand'] == pitches['p_throws'], 0, 1)
#Create variable for runners in scoring positon (2nd or 3rd)
pitches['RISP'] = 0 
for i in range(len(pitches['RISP'])):
    if pitches.loc[i, 'on_3b'] == 1 or pitches.loc[i, 'on_2b'] == 1:
        pitches.loc[i, 'RISP'] = 1
#Create variable for if there is a double play oppurtunity(could influence outcome)
pitches['DP_chance'] = np.where((pitches['on_1b'] == 1) & (pitches['outs_when_up'] != 2), 1, 0)
#Create variable for whether bases are loaded, pitchers can't walk batters
pitches['bases_loaded'] = np.where((pitches['on_1b'] == 1) & (pitches['on_2b'] == 1) & (pitches['on_3b'] == 1), 1, 0)

#Select wanted columns
feature_cols = pitches[['pitch_number', 'count_advantage', 'inning', 'RISP', 'DP_chance', 'outs_when_up', 'bases_loaded',
                        'score_diff', 'pitcher', 'batter', 'outcome']]


feature_cols = feature_cols.fillna(0)

#Remove hits from dataset so its only ball and strike outcomes
feature_cols = feature_cols[feature_cols['outcome'] != 2]
#Create seperate unmerged data frame for no player model
unmerged = feature_cols


#Load in pitching and batting data
pitchers = pd.read_csv('Data/Pitcher_Data.csv')
batters = pd.read_csv('Data/batter_data.csv')

#turn ball and called strike to per_pitch instead of sums
pitchers['called_strike%'] = pitchers['p_called_strike'] / pitchers['pitch_count'] * 100
pitchers['ball%'] = pitchers['p_ball'] / pitchers['pitch_count'] * 100 
#Not needed if using in_zone_percent as aggeragate

#Pick featured columns from pitchers set
pitcher_cols = pitchers[['player_id', 'in_zone_percent', 'whiff_percent' ]]

#Rename 'player_id' to 'pitcher_id' in pitcher_cols for consistent merging, and rename stats to have pitcher label to avoid confusion
pitcher_cols.rename(columns = {'player_id' : 'pitcher_id', 'in_zone_percent' : 'p_in_zone_percent', 'whiff_percent': 'p_whiff_percent' }, inplace= True)
# Rename 'pitcher' to 'pitcher_id' in feature_cols for consistent merging
feature_cols.rename(columns={'pitcher': 'pitcher_id'}, inplace=True)

# Merge feature_cols with pitcher_cols on player_id, keeping all rows from feature_cols
feature_cols = pd.merge(feature_cols, pitcher_cols, on='pitcher_id', how='left')

#Merge batter stats
#pick feature cols
batter_cols = batters[['player_id', 'in_zone_percent', 'oz_swing_miss_percent']]
#Rename to batters
batter_cols.rename(columns = {'player_id' : 'batter_id', 'in_zone_percent' : 'b_in_zone_percent', 'oz_swing_miss_percent': 'b_oz_swing_miss_percent' }, inplace= True)

feature_cols.rename(columns={'batter': 'batter_id'}, inplace=True)

#Merge feature cols and batter cols
feature_cols = pd.merge(feature_cols, batter_cols, on='batter_id', how='left')

#Test merge
#print(feature_cols.head())
#print(feature_cols.tail())

#Save into two datasets, for two different models, player info and no player info
player_features = feature_cols.drop(columns = ['pitcher_id', 'batter_id', 'outcome'])

no_player_features = unmerged.drop(columns = ['pitcher_id', 'batter', 'outcome'])

labels = feature_cols['outcome']

for i in range(len(player_features)):
    #Change NaN values to be median values

    #Go through 4 cols that have NaN values
    if pd.isna(player_features.loc[i, 'p_in_zone_percent']):
        player_features.loc[i, 'p_in_zone_percent'] = player_features['p_in_zone_percent'].median()
    
    if pd.isna(player_features.loc[i, 'p_whiff_percent']):
        player_features.loc[i, 'p_whiff_percent'] = player_features['p_whiff_percent'].median()
    
    if pd.isna(player_features.loc[i, 'b_in_zone_percent']):
        player_features.loc[i, 'b_in_zone_percent'] = player_features['b_in_zone_percent'].median()

    if pd.isna(player_features.loc[i, 'b_oz_swing_miss_percent']):
        player_features.loc[i, 'b_oz_swing_miss_percent'] = player_features['b_oz_swing_miss_percent'].median()

train_player_features, test_player_features, train_labels, test_labels = train_test_split(player_features, labels, test_size=0.2, random_state=20)
train_no_player_features, test_no_player_features, train_labels, test_labels = train_test_split(no_player_features, labels, test_size=0.2, random_state=20)

# Save player-based features
train_player_features.to_csv('Processed_Data/train_player_features.csv', index=False)
test_player_features.to_csv('Processed_Data/test_player_features.csv', index=False)

# Save non-player features
train_no_player_features.to_csv('Processed_Data/train_no_player_features.csv', index=False)
test_no_player_features.to_csv('Processed_Data/test_no_player_features.csv', index=False)

# Save labels
train_labels.to_csv('Processed_Data/train_labels.csv', index=False)
test_labels.to_csv('Processed_Data/test_labels.csv', index=False)