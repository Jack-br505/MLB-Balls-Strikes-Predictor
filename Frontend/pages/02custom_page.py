
#Same as 01 model page but team names are a text input 
import streamlit as st
import statsapi
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(
    page_title= "MLB Pitch Predictor (Model Use Custom Scenario)",
    page_icon='⚾️',
    layout = 'wide'
)
st.title("MLB Pitch Predictor")
st.caption("Enter the specific values for the pitch and get a prediction")
#Load in game details
game = st.session_state['Game']


st.subheader(f"{st.session_state['Away_Team']} @ {st.session_state['Home_Team']}")

#Get deafult values or known values 
balls = "0"
strikes = "0"

#Start here
#Get the rosters for the teams for switching pitchers / batters
away_id = statsapi.lookup_team(st.session_state['Away_Team'])[0]['id']
home_id = statsapi.lookup_team(st.session_state['Home_Team'])[0]['id']

away_roster = away_roster = statsapi.roster(away_id)
home_roster = statsapi.roster(home_id)

#Convert string output of roster function to a list
a_table = [line.split("  ") for line in away_roster.split("\n")]
h_table = [line.split("  ") for line in home_roster.split("\n")]

#Remove last element because that is an empty list
a_table.pop(-1)
h_table.pop(-1)

#had issue with spaces in names like <Emmanuel De  Jesus>
#Merge 3rd and 4th elements if there was a 4th elemnt in list caused by a spaced out name
a_table = [
    lst[:2] + [lst[2] + ' ' + lst[3]] if len(lst) == 4 else lst 
    for lst in a_table
]
h_table = [
    lst[:2] + [lst[2] + ' ' + lst[3]] if len(lst) == 4 else lst 
    for lst in h_table
]

#Remove leading spaces from player name
for list in a_table:
    list[2] = list[2].lstrip()
for list in h_table:
    list[2] = list[2].lstrip()

away_roster_df = pd.DataFrame(a_table, columns=['Number', 'Position', 'Name'])
home_roster_df = pd.DataFrame(h_table, columns=['Number', 'Position', 'Name'])

#Find starting pitchers to use as default
if st.session_state['Inning_state'] == 'Top' or st.session_state['Inning_state'] == "End":
    s_pitcher = game.get('home_probable_pitcher') #Home team pitcher if home team is pitching
else:
    s_pitcher = game.get('away_probable_pitcher')




col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    with st.container(border=True):
        st.subheader("Count")
        # The Dropdown
        balls = st.selectbox( #Save to balls variable
            "Balls",
            ["0", "1", "2", "3"],
            key="balls_box"
        )
        # The Dropdown
        strikes = st.selectbox( #Save to strikes variable
            "Strikes",
            ["0", "1", "2"],
            key="strikes_box"
        )
#Set deafult values for the score variables
st.session_state['away_score'] = "0"
st.session_state['home_score'] = "0"

with col2:
    with st.container(border=True):
        st.subheader("Score:")
        st.text_input(f"{st.session_state['Away_Team']}", value = st.session_state['away_score'], placeholder="0")
        st.text_input(f"{st.session_state['Home_Team']}", value = st.session_state['home_score'], placeholder="0")


with col3:
    with st.container(border=True):
        st.subheader("Inning")
        #st.write(st.session_state['inning_state'], str(inning))
        state = st.selectbox(
            "State",
            ["Top", "Bottom"],
            key="inning_state_box"
        )
        inning = st.selectbox(
            "Number",
            ["1", "2", "3", "4", "5", "6", "7", "8", "9"],
            key = 'inning_box'
        )
        st.subheader("Outs")
        outs = st.selectbox('Outs', ["0", "1", "2"], key = 'out_box')

with col4:
    with st.container(border=True):
        st.subheader("Baserunners")
        first = st.checkbox("1st Base")
        second = st.checkbox("2nd Base")
        third = st.checkbox("3rd Base")

with col5:
    with st.container(border=True):
        st.subheader("Players")
        #Get the rosters of the pitching team to change pitcher
        if game.get("inning_state") == "Top" or game.get("inning_state" == "End"):
            pitchers = home_roster_df[home_roster_df['Position'] == 'P']
            hitters = away_roster_df[away_roster_df['Position'] != 'P']
        else:
            pitchers = away_roster_df[away_roster_df['Position'] == 'P']
            hitters = home_roster_df[home_roster_df['Position'] != 'P']
        #Create selectbox for pitcher with starter as default
        pitcher = st.selectbox(
            "Pitcher",
            [s_pitcher] + pitchers['Name'].tolist(),
            key="pitcher_box"
        )
        #Create box for batter (no default)
        batter = st.selectbox(
            "Batter",
            hitters['Name'].tolist(),
            key="batter_box1"
        )

#Load in model
model = joblib.load('Model_testing/Random_Forest/final_model.pkl')

#set y_pred to 2 by deafult to get rid of error on startup
y_pred = 2

#Create button to predict pitch and move count ahead
if st.button("Predict Pitch", use_container_width=True, key = "predict"):
    #needed columns are  'pitch_number', 'count_advantage', 'inning',  'RISP', 'DP_chance', 'outs_when_up', 'bases_loaded','score_diff', player metrics
    #Create empty dataframe to keep data
    col_names = ['pitch_number', 'count_advantage', 'inning',  'RISP', 'DP_chance', 'outs_when_up', 'bases_loaded','score_diff',
                 'p_in_zone_percent', 'p_whiff_percent', 'b_in_zone_percent', 'b_oz_swing_miss_percent']
    temp_df = pd.DataFrame([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],columns = col_names) #Initialize one empty row
    #Get all the data
    temp_df['pitch_number'] = int(balls) + int(strikes)
    temp_df['count_advantage'] = int(balls) - int(strikes)
    #Runners in scoring position
    if second or third:
        temp_df['RISP'] = 1
    else:
        temp_df['RISP'] = 0
    
    if first:
        temp_df['DP_chance'] = 1
    else:
        temp_df['DP_chance'] = 0
    
    temp_df['outs_when_up'] = int(outs)

    if first and second and third:
        temp_df['bases_loaded'] = 1
    else:
        temp_df['bases_loaded'] = 0
    
    #Get score data
    if st.session_state['inning_state'] == "Top":
        temp_df['score_diff'] = int(st.session_state['away_score']) - int(st.session_state['home_score'])
    else:
        temp_df['score_diff'] = -1 * st.session_state['away_score'] +  st.session_state['home_score']

    #Use 2025 data for player metrics, if no data (ex: rookie) then use median val in 2025 data
    #Metrics are Pitcher: 'in_zone_percent', 'whiff_percent' Batter: 'in_zone_percent', 'oz_swing_miss_percent'
    pitcher_data = st.session_state['pitcher_metrics']
    #Find if current pitcher is in list of names
    P_df = pitcher_data[pitcher_data['full_name'] == pitcher]

    if len(P_df) == 1: #Make sure the pitcher is in 2025 data, and also no duplicate names (Would need API to have statcast ID to deal with duplicate names)
        temp_df['p_in_zone_percent'] = P_df.iloc[0, 6] #Get vals for needed data
        whiff_percent = P_df.iloc[0, 7]
    else:
        temp_df['p_in_zone_percent'] = pitcher_data['in_zone_percent'].median()
        temp_df['p_whiff_percent'] = pitcher_data['whiff_percent'].median()
        #st.write("Error: Pitcher Data not Found")

    #Do same for hitters
    batter_data = st.session_state['batter_metrics']
    #Mask for current hitter
    B_df = batter_data[batter_data['full_name'] == batter]
    if len(B_df) == 1: #Make sure the hitter is in 2025 data, and also no duplicate names (Would need API to have statcast ID to deal with duplicate names)
        temp_df['b_in_zone_percent'] = B_df.iloc[0, 4] #Get vals for needed data
        temp_df['b_oz_swing_miss_percent'] = B_df.iloc[0, 3]
    else:
        temp_df['b_in_zone_percent'] = batter_data['in_zone_percent'].median()
        temp_df['b_oz_swing_miss_percent'] = batter_data['oz_swing_miss_percent'].median()
        #st.write("Error: Batter Data not Found")
    
    y_pred = model.predict(temp_df)

    st.session_state['last_prediction'] = y_pred[0] #Save val
#Write out prediction
#Write out prediction
if y_pred == 1:
    p ="Strike"
elif y_pred == 0:
    p = "Ball"
else:
    p = ''

st.write(f"Prediction: {p}")
#Create section where user can enter the actual outcome, advancing numbers and tracking accuracy
st.subheader("Acutal Outcome")

s_col, b_col, f_col, o_col, h_col = st.columns(5)

#Function for increasing the value of one of the select boxes
def increment_selectbox(box_key, options_list):
    """
    Increases the index of the select box by one

    inputs
    ------

    box_key str: the name of the key of the selectbox you want to increment

    options_list list: a list the possible options for the box
    """
    # Get the current string value from the selectbox key
    current_value = st.session_state[box_key]
    
    # Find its current index in the list
    current_index = options_list.index(current_value)
    
    # Calculate the next index (with safety check so it wraps around)
    next_index = (current_index + 1) % len(options_list)
    
    # Update the selectbox key directly with the NEW string value
    st.session_state[box_key] = options_list[next_index]

#Create sperate cases for incrementing the different things
def reset_count():
    """Resets the count to 0, 0 """
    st.session_state['balls_box'] = "0"
    st.session_state['strikes_box'] = "0"

def increment_out(inning = inning):
    increment_selectbox('out_box', out_options)

    reset_count()

    #Change inning if 3 outs
    if st.session_state['out_box'] == "0":
        #Change inning state
        increment_selectbox('inning_state_box', ['Top', 'Bottom']) #Change state

        if (int(inning) == 9 and state == 'Bottom') or (int(inning) == 9 and st.session_state['home_score'] > st.session_state['away_score']):
            inning = "Game Finished"
            st.session_state['inning_state'] = ""

        elif state == 'Bottom': #If new inning switch inning num
            increment_selectbox('inning_box', ["1", "2", "3", "4", "5", "6", "7", "8", "9"]) #Increase inning if bottom of inning

def increment_strike():
    increment_selectbox('strikes_box', strike_options)

    #Add condition if there is a strikeout
    if st.session_state['strikes_box'] == '0':
        increment_out()
    
    #Change the counters
    if st.session_state['last_prediction'] == 1:
        st.session_state['correct'] = str( int(st.session_state['correct']) + 1)
    else:
        st.session_state['incorrect'] = str( int(st.session_state['incorrect']) + 1)

    
def increment_foul():
    if st.session_state['strikes_box'] != '2':
        increment_strike()
        #Dont increment if 2 strikes
    
    #Change the counters
    if st.session_state['last_prediction'] == 1:
        st.session_state['correct'] = str(int(st.session_state['correct']) + 1)
    else:
        st.session_state['incorrect'] = str( int(st.session_state['incorrect']) + 1)

def increment_ball():
    increment_selectbox('balls_box', ball_options)
    if st.session_state['balls_box'] == '0':
        reset_count()

    #Change the counters
    if st.session_state['last_prediction'] == 0:
        st.session_state['correct'] = str( int(st.session_state['correct']) + 1)
    else:
        st.session_state['incorrect'] = str( int(st.session_state['incorrect']) + 1)
#Add options_lists for the possible boxes
strike_options = ['0', '1', '2']
ball_options = ['0', '1', '2', '3']
out_options = ['0', '1', '2']

  
#Make the buttons
with s_col:
    st.button("Strike", key = "strike_button",  use_container_width=True, on_click = increment_strike)#If user says actual pitch was a strike

with b_col:
    st.button("Ball", key = 'ball_button', use_container_width=True, on_click= increment_ball)

with f_col:
    st.button("Foul Ball", key = 'foul_button', use_container_width=True, on_click= increment_foul)

with o_col:
    st.button("Put in play (Out)", key = 'out_button',  use_container_width=True, on_click= increment_out)
with h_col:
    st.button("Hit", key = 'hit_button', use_container_width=True,  on_click= reset_count)

st.subheader("Overall Accuracy")
right, wrong = st.columns(2)

with right:
    st.subheader("Correct")
    st.write(f"### {st.session_state['correct']}")
with wrong:
    st.subheader("Incorrect")
    
    st.write(f"### {st.session_state['incorrect']}")