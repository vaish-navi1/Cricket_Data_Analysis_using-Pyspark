import pandas as pd
import streamlit as st

# Load CSV data into pandas DataFrame
def load_data(file_path):
    return pd.read_csv(file_path)

def show_analysis_page():
    st.title("Cricket Data Analysis - Analysis Page")

    # Analysis options
    analysis_options = [
        "Select Analysis",
        "Total and Average Runs",
        "High Impact Balls",
        "Win Margin Category",
        "Toss Match Winner"
    ]

    # User selection for analysis type
    selected_analysis = st.selectbox("Choose an analysis type:", analysis_options)

    if selected_analysis == "Total and Average Runs":
        # Load and filter the "Ball by Ball" dataset
        ball_by_ball_df = load_data("C:/Users/vaish/Downloads/cricket/Ball_By_Ball.csv")
        filtered_ball_by_ball_df = ball_by_ball_df[(ball_by_ball_df['Wides'] == 0) & (ball_by_ball_df['Noballs'] == 0)]

        # Calculate total and average runs
        total_and_avg_runs = filtered_ball_by_ball_df.groupby(['MatcH_id', 'Innings_No']).agg(
            total_runs=pd.NamedAgg(column='Runs_Scored', aggfunc='sum'),
            average_runs=pd.NamedAgg(column='Runs_Scored', aggfunc='mean')
        ).reset_index()

        # Display the total and average runs data
        st.write("### Total and Average Runs:")
        st.dataframe(total_and_avg_runs)

    elif selected_analysis == "High Impact Balls":
        # Load and filter the "Ball by Ball" dataset
        ball_by_ball_df = load_data("C:/Users/vaish/Downloads/cricket/Ball_By_Ball.csv")
        filtered_ball_by_ball_df = ball_by_ball_df[(ball_by_ball_df['Wides'] == 0) & (ball_by_ball_df['Noballs'] == 0)]

        # Calculate high impact balls based on runs and wickets
        filtered_ball_by_ball_df['high_impact'] = ((filtered_ball_by_ball_df['Runs_Scored'] + filtered_ball_by_ball_df['Extra_runs']) > 6) | (filtered_ball_by_ball_df['Bowler_Wicket'] == True)

        # Display high impact balls
        st.write("### High Impact Balls:")
        st.dataframe(filtered_ball_by_ball_df[filtered_ball_by_ball_df['high_impact']])

    elif selected_analysis == "Win Margin Category":
        # Load the "Match" dataset and categorize win margins
        match_df = load_data("C:/Users/vaish/Downloads/cricket/Match.csv")
        match_df['win_margin_category'] = pd.cut(
            match_df['Win_Margin'], 
            bins=[-1, 49, 99, float('inf')], 
            labels=['Low', 'Medium', 'High']
        )

        # Display win margin categories
        st.write("### Win Margin Categories:")
        st.dataframe(match_df[['match_id','Win_Margin', 'win_margin_category']])

    elif selected_analysis == "Toss Match Winner":
        # Load the "Match" dataset for toss and winner comparison
        match_df = load_data("C:/Users/vaish/Downloads/cricket/Match.csv")
        match_df['toss_match_winner'] = match_df.apply(
            lambda row: 'Correct' if row['Toss_Winner'] == row['match_winner'] else 'Incorrect', axis=1
        )

        # Display toss match winner results
        st.write("### Toss Match Winner Results:")
        st.dataframe(match_df[['match_id', 'Toss_Winner', 'match_winner', 'toss_match_winner']])

# Call the function to show the analysis page in a Streamlit app
if __name__ == "__main__":
    show_analysis_page()