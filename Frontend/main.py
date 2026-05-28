import streamlit as st
import statsapi
import pandas as pd
from datetime import datetime
import pytz

st.set_page_config(
    page_title= "MLB Pitch Predictor (Game Selector)",
    page_icon='⚾️',
    layout = 'wide'
)

st.title("MLB Pitch Predictor")
st.caption("Today's Games")

#Get the current date
from datetime import date

today = date.today()
print(today)  #

#Find schedule for games on current date
schedule = statsapi.schedule(date = today)

game_num = len(schedule)

#Load in player data to get player metrics for the model
st.session_state['pitcher_metrics'] = pd.read_csv('Data/Pitcher_Data.csv')
st.session_state['batter_metrics'] = pd.read_csv('Data/batter_data.csv')


#Set deafult values for the score variables to track overall accuracy
st.session_state['correct'] = "0"
st.session_state['incorrect'] = "0"

#Load in names of all the mlb teams for custom game selection
all_teams = ['Baltimore Orioles', 'Boston Red Sox', 'New York Yankees', 'Tampa Bay Rays', 'Toronto Blue Jays',
'Chicago White Sox', 'Cleveland Guardians', 'Detroit Tigers', 'Kansas City Royals', 'Minnesota Twins',
'Houston Astros', 'Los Angeles Angels', 'Oakland Athletics', 'Seattle Mariners', 'Texas Rangers', 
'Atlanta Braves', 'Miami Marlins', 'New York Mets', 'Philadelphia Phillies', 'Washington Nationals',
'Chicago Cubs', 'Cincinnati Reds', 'Milwaukee Brewers', 'Pittsburgh Pirates', 'St. Louis Cardinals',
'Arizona Diamondbacks', 'Colorado Rockies','Los Angeles Dodgers', 'San Diego Padres', 'San Francisco Giants']


# Create a Streamlit tab for each scheduled game
if game_num == 0:
    st.info("No games scheduled for today.")
    #Create custom game option
    st.subheader("Custom game option: ")
    st.write("Input team names")

    a, at, h = st.columns(3) #Create columns to select teams

    with a:
        st.session_state['Away_Team'] = st.selectbox('Away Team', all_teams, key = 'away_box')
    with at:
        st.subheader("@")
    with h:
        st.session_state['Home_Team'] = st.selectbox('Home Team', all_teams, key = 'home_box')
    
    if st.button("Use Model in Custom Scenario", use_container_width=True, key=f"Custom_no_games"):
        st.switch_page("pages/02custom_page.py")
else:
    def _get_team_name(field):
        if not field:
            return "Unknown"
        if isinstance(field, str):
            return field
        if isinstance(field, dict):
            return field.get('team_name') or field.get('name') or field.get('full_name') or field.get('alias') or 'Unknown'
        return str(field)

    tab_labels = []
    for g in schedule:
        away = _get_team_name(g.get('away_name') or g.get('away'))
        home = _get_team_name(g.get('home_name') or g.get('home'))
        tab_labels.append(f"{away} @ {home}")

    tab_labels.append("Custom Scenario")

    tabs = st.tabs(tab_labels)

    for i, tab in enumerate(tabs):
        with tab:
            if tab_labels[i] == "Custom Scenario": 
                #Different code if the user selects a custom scenario
                #Create boxes to pick teams
                a, at, h = st.columns(3) #Create columns to select teams
                with a:
                    st.session_state['Away_Team'] = st.selectbox('Away Team', all_teams, key = 'away_box')
                with at:
                    st.subheader("@")
                with h:
                    st.session_state['Home_Team'] = st.selectbox('Home Team', all_teams, key = 'home_box')
                if st.button("Use Model in Custom Scenario", use_container_width=True, key=f"Custom_games"):
                    #Set inning to top 1 by default
                    st.session_state['Inning_state'] = 'Top'
                    st.session_state['inning'] = "1"
                    st.switch_page("pages/02custom_page.py") #Switch to the page for custom use of model


            else:
                #If its a scheduled game use Statsapi to get data on teams and game
                game = schedule[i]
                away = _get_team_name(game.get('away_name') or game.get('away'))
                home = _get_team_name(game.get('home_name') or game.get('home'))
                st.subheader(f"{away} @ {home}")

                # Display basic game info (fall back to raw dict if keys differ)
                time =  game.get('game_datetime')
                if time:
                    # Convert UTC to EST
                    utc_dt = datetime.fromisoformat(time.replace('Z', '+00:00'))
                    est_tz = pytz.timezone('US/Eastern')
                    est_dt = utc_dt.astimezone(est_tz)
                    time = est_dt.strftime('%I:%M %p') + " EST" # Format as HH:MM AM/PM
            
                venue = game.get('venue') or game.get('venue_name') or game.get('location')
                inning = game.get('inning_state') + " " + str(game.get("current_inning"))

                if time:
                    st.write("Time:", time)
                if venue:
                    st.write("Venue:", venue)
                if inning != " ":
                    st.write("Inning:", inning)
                    #Save to session state
                    st.session_state['inning'] = inning
                    st.session_state['inning_state'] = game.get('inning_state')

                # Dump the raw schedule entry for debugging/extension
                #st.write(game)
            
                st.session_state['Game'] = game
            

                if st.button("Use Model", use_container_width=True, key=f"Game{i}"):
                    st.switch_page("pages/01model_page.py")
