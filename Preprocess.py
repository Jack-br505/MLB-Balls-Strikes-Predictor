import pandas as pd
from sklearn.model_selection import train_test_split

pitches = pd.read_csv("Data/Data_MLB_2025_StatcastPostseason_PitchByPitch_20251102a.csv")

pitch_outcome_num = []
for i in range(pitches.shape[0]):
    if pitches.loc[i, 'description'] in ['called_strike', 'foul', 'swinging_strike', 'foul_tip']:
        pitch_outcome_num.append(1)
    elif pitches.loc[i, 'description'] == 'ball':
        pitch_outcome_num.append(0)
    else:
        pitch_outcome_num.append(2)

pitches['outcome'] = pitch_outcome_num

feature_cols = pitches[['balls', 'strikes', 'inning', 'pitch_number', 'on_3b', 'on_2b', 'on_1b', 'outs_when_up',
              'bat_score', 'pitcher', 'batter', 'outcome']]


feature_cols = feature_cols.fillna(0)

for i in range(feature_cols.shape[0]):
    if feature_cols.loc[i, 'on_3b'] != 0.0:
        feature_cols.loc[i, 'on_3b'] = 1.0
    if feature_cols.loc[i, 'on_2b'] != 0.0:
        feature_cols.loc[i, 'on_2b'] = 1.0
    if feature_cols.loc[i, 'on_1b'] != 0.0:
        feature_cols.loc[i, 'on_1b'] = 1.0


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

train_player_features, test_player_features, train_labels, test_labels = train_test_split(player_features, labels, test_size=0.2)
train_no_player_features, test_no_player_features, train_labels, test_labels = train_test_split(player_features, labels, test_size=0.2)