import streamlit as st
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, BooleanType, DateType
from pyspark.sql.functions import col, sum, avg, when
from pyspark.sql import Window
import matplotlib.pyplot as plt
import seaborn as sns

spark = SparkSession.builder \
    .appName("Cricket Data Analysis") \
    .getOrCreate()

ball_by_ball_schema = StructType([
    StructField("match_id", IntegerType(), True),
    StructField("over_id", IntegerType(), True),
    StructField("ball_id", IntegerType(), True),
    StructField("innings_no", IntegerType(), True),
    StructField("team_batting", StringType(), True),
    StructField("team_bowling", StringType(), True),
    StructField("striker_batting_position", IntegerType(), True),
    StructField("extra_type", StringType(), True),
    StructField("runs_scored", IntegerType(), True),
    StructField("extra_runs", IntegerType(), True),
    StructField("wides", IntegerType(), True),
    StructField("legbyes", IntegerType(), True),
    StructField("byes", IntegerType(), True),
    StructField("noballs", IntegerType(), True),
    StructField("penalty", IntegerType(), True),
    StructField("bowler_extras", IntegerType(), True),
    StructField("out_type", StringType(), True),
    StructField("caught", BooleanType(), True),
    StructField("bowled", BooleanType(), True),
    StructField("run_out", BooleanType(), True),
    StructField("lbw", BooleanType(), True),
    StructField("retired_hurt", BooleanType(), True),
    StructField("stumped", BooleanType(), True),
    StructField("caught_and_bowled", BooleanType(), True),
    StructField("hit_wicket", BooleanType(), True),
    StructField("obstructingfeild", BooleanType(), True),
    StructField("bowler_wicket", BooleanType(), True),
    StructField("match_date", DateType(), True),
    StructField("season", IntegerType(), True),
    StructField("striker", IntegerType(), True),
    StructField("non_striker", IntegerType(), True),
    StructField("bowler", IntegerType(), True),
    StructField("player_out", IntegerType(), True),
    StructField("fielders", IntegerType(), True),
    StructField("striker_match_sk", IntegerType(), True),
    StructField("strikersk", IntegerType(), True),
    StructField("nonstriker_match_sk", IntegerType(), True),
    StructField("nonstriker_sk", IntegerType(), True),
    StructField("fielder_match_sk", IntegerType(), True),
    StructField("fielder_sk", IntegerType(), True),
    StructField("bowler_match_sk", IntegerType(), True),
    StructField("bowler_sk", IntegerType(), True),
    StructField("playerout_match_sk", IntegerType(), True),
    StructField("battingteam_sk", IntegerType(), True),
    StructField("bowlingteam_sk", IntegerType(), True),
    StructField("keeper_catch", BooleanType(), True),
    StructField("player_out_sk", IntegerType(), True),
    StructField("matchdatesk", DateType(), True)
])

def load_data(file_name):
    if file_name == "Ball by Ball Data":
        return spark.read.schema(ball_by_ball_schema).format("csv").option("header", "true").load("C:/Users/vaish/Downloads/cricket/Ball_By_Ball.csv")
    elif file_name == "Match Data":
        return spark.read.format("csv").option("header", "true").load("C:/Users/vaish/Downloads/cricket/Match.csv")
    elif file_name == "Player Data":
        return spark.read.format("csv").option("header", "true").load("C:/Users/vaish/Downloads/cricket/Player.csv")
    elif file_name == "Player Match Data":
        return spark.read.format("csv").option("header", "true").load("C:/Users/vaish/Downloads/cricket/Player_Match.csv")
    elif file_name == "Team Data":
        return spark.read.format("csv").option("header", "true").load("C:/Users/vaish/Downloads/cricket/Team.csv")

if 'page' not in st.session_state:
    st.session_state.page = 'home'

def show_home_page():
    st.title("Cricket Data Analysis")
    st.write("Select a CSV file to display its data:")

    file_options = [
        "Ball by Ball Data",
        "Match Data",
        "Player Data",
        "Player Match Data",
        "Team Data"
    ]
    selected_file = st.selectbox("Choose a file:", file_options)

    data_df = load_data(selected_file)

    # Convert to Pandas DataFrame for Streamlit
    pandas_df = data_df.toPandas()

    # Display the DataFrame as a table
    st.write(f"### Data from {selected_file}")
    st.dataframe(pandas_df)

    # Optionally, display a sample of the DataFrame
    if st.checkbox("Show sample data"):
        st.write(pandas_df.sample(10))  

    if st.button("Go to Analysis"):
        st.session_state.page = "analysis"

    if st.button("Show Analysis Dashboard"):
        st.session_state.page = "analysis_dashboard"
    
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report, confusion_matrix

    ball_by_ball = pd.read_csv(r"C:\Users\vaish\Downloads\cricket\Ball_By_Ball.csv", encoding='ISO-8859-1')
    player = pd.read_csv(r"C:\Users\vaish\Downloads\cricket\Player.csv", encoding='ISO-8859-1')
    match = pd.read_csv(r"C:\Users\vaish\Downloads\cricket\Match.csv", encoding='ISO-8859-1')
    player_match = pd.read_csv(r"C:\Users\vaish\Downloads\cricket\Player_Match.csv", encoding='ISO-8859-1')
    team = pd.read_csv(r"C:\Users\vaish\Downloads\cricket\Team.csv", encoding='ISO-8859-1')

    print("player_match columns:", player_match.columns)
    print("match columns:", match.columns)
    print("ball_by_ball columns:", ball_by_ball.columns)

    data = player_match.merge(match, left_on='Match_Id', right_on='match_id', how='inner')
    data = data.merge(ball_by_ball, left_on='Match_Id', right_on='MatcH_id', suffixes=('', '_bb'), how='inner')

    # Check for NaN values in the target variable
    print("NaN values in Win_Margin:", data['Win_Margin'].isnull().sum())
    nan_count = data['Win_Margin'].isnull().sum()
    st.write("NaN values in Win_Margin:", nan_count)
    # Option 1: Drop rows with NaN values in target
    data = data.dropna(subset=['Win_Margin'])

    features = data[['Over_id', 'Ball_id', 'Innings_No', 'BowlingTeam_SK']]
    target = data['Win_Margin']

    # Handle categorical data if necessary (e.g., one-hot encoding)
    features = pd.get_dummies(features, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
     
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))
    conf_matrix = confusion_matrix(y_test, predictions)
    
    st.write("Confusion Matrix:")
    st.dataframe(conf_matrix)
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))
    class_report = classification_report(y_test, predictions, output_dict=True)
   
    st.write("Classification Report:")
    st.dataframe(class_report)

def show_analysis_page():
    st.title("Cricket Data Analysis - Analysis Page")

    analysis_options = [
        "Select Analysis",
        "Total and Average Runs",
        "High Impact Balls",
        "Win Margin Category",
        "Toss Match Winner"
    ]

    selected_analysis = st.selectbox("Choose an analysis type:", analysis_options)

    if selected_analysis == "Total and Average Runs":
        filtered_ball_by_ball_df = load_data("Ball by Ball Data").filter((col("wides") == 0) & (col("noballs") == 0))

        # Aggregation: Calculate total and average runs scored
        total_and_avg_runs = filtered_ball_by_ball_df.groupBy("match_id", "innings_no").agg(
            sum("runs_scored").alias("total_runs"),
            avg("runs_scored").alias("average_runs")
        )
        st.write("### Total and Average Runs:")
        st.dataframe(total_and_avg_runs.toPandas())  

    elif selected_analysis == "High Impact Balls":
        filtered_ball_by_ball_df = load_data("Ball by Ball Data").filter((col("wides") == 0) & (col("noballs") == 0))
        windowSpec = Window.partitionBy("match_id", "innings_no").orderBy("over_id")
        filtered_ball_by_ball_df = filtered_ball_by_ball_df.withColumn(
            "running_total_runs",
            sum("runs_scored").over(windowSpec)
        )
        filtered_ball_by_ball_df = filtered_ball_by_ball_df.withColumn(
            "high_impact",
            when((col("runs_scored") + col("extra_runs") > 6) | (col("bowler_wicket") == True), True).otherwise(False)
        )
        st.write("### High Impact Balls:")
        st.dataframe(filtered_ball_by_ball_df.filter(col("high_impact") == True).toPandas())

    elif selected_analysis == "Win Margin Category":
        match_df = load_data("Match Data").withColumn(
            "win_margin_category",
            when(col("win_margin") >= 100, "High")
            .when((col("win_margin") >= 50) & (col("win_margin") < 100), "Medium")
            .otherwise("Low")
        )
        st.write("### Win Margin Categories:")
        st.dataframe(match_df.select("match_id", "win_margin", "win_margin_category").toPandas())

    elif selected_analysis == "Toss Match Winner":
        match_df = load_data("Match Data").withColumn(
            "toss_match_winner",
            when(col("toss_winner") == col("match_winner"), "Correct")
            .otherwise("Incorrect")
        )
        st.write("### Toss Match Winner Results:")
        st.dataframe(match_df.select("match_id", "toss_winner", "match_winner", "toss_match_winner").toPandas())

   
    if st.button("Go Back to Home"):
        st.session_state.page = "home"

import seaborn as sns

def show_analysis_dashboard():
    st.title("Cricket Analysis Dashboard")
    st.write("Welcome to the analysis dashboard! Here you can view various insights and metrics.")

    ball_by_ball_df = load_data("Ball by Ball Data")
    player_match_df = load_data("Player Match Data")
    player_df = load_data("Player Data")
    match_df = load_data("Match Data")  

    # Register DataFrames as temporary views for SQL queries
    ball_by_ball_df.createOrReplaceTempView("ball_by_ball")
    player_match_df.createOrReplaceTempView("player_match")
    player_df.createOrReplaceTempView("player")
    match_df.createOrReplaceTempView("match")  # Register match data view

    # Perform SQL query for economical bowlers in the powerplay
    economical_bowlers_powerplay = spark.sql("""
        SELECT 
            p.player_name, 
            AVG(b.runs_scored) AS avg_runs_per_ball, 
            COUNT(b.bowler_wicket) AS total_wickets
        FROM ball_by_ball b
        JOIN player_match pm ON b.match_id = pm.match_id AND b.bowler = pm.player_id
        JOIN player p ON pm.player_id = p.player_id
        WHERE b.over_id <= 6
        GROUP BY p.player_name
        HAVING COUNT(*) >= 1
        ORDER BY avg_runs_per_ball, total_wickets DESC
    """)

    economical_bowlers_pd = economical_bowlers_powerplay.toPandas()
    st.title("Economic bowlers in powerplay")
    st.dataframe(economical_bowlers_powerplay.toPandas())
    plt.figure(figsize=(12, 8))
    top_economical_bowlers = economical_bowlers_pd.nsmallest(10, 'avg_runs_per_ball')
    plt.bar(top_economical_bowlers['player_name'], top_economical_bowlers['avg_runs_per_ball'], color='skyblue')
    plt.xlabel('Bowler Name')
    plt.ylabel('Average Runs per Ball')
    plt.title('Most Economical Bowlers in Powerplay Overs (Top 10)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(plt)

    # Toss impact on match outcome analysis
    toss_impact_individual_matches = spark.sql("""
        SELECT m.match_id, m.toss_winner, m.toss_name, m.match_winner,
               CASE WHEN m.toss_winner = m.match_winner THEN 'Won' ELSE 'Lost' END AS match_outcome
        FROM match m
        WHERE m.toss_name IS NOT NULL
        ORDER BY m.match_id
    """)

    toss_impact_pd = toss_impact_individual_matches.toPandas()
    st.title("Toss impact on matches")
    st.dataframe(toss_impact_individual_matches.toPandas())
    # Creating a countplot to show win/loss after winning toss
    plt.figure(figsize=(10, 6))
    sns.countplot(x='toss_winner', hue='match_outcome', data=toss_impact_pd)
    plt.title('Impact of Winning Toss on Match Outcomes')
    plt.xlabel('Toss Winner')
    plt.ylabel('Number of Matches')
    plt.legend(title='Match Outcome')
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(plt)
    
    # average runs
    average_runs_in_wins = spark.sql("""
        SELECT p.player_name, AVG(b.runs_scored) AS avg_runs_in_wins, COUNT(*) AS innings_played
        FROM ball_by_ball b
        JOIN player_match pm ON b.match_id = pm.match_id AND b.striker = pm.player_id
        JOIN player p ON pm.player_id = p.player_id
        JOIN match m ON pm.match_id = m.match_id
        WHERE m.match_winner = pm.player_team
        GROUP BY p.player_name
        ORDER BY avg_runs_in_wins ASC
    """)
    st.title("average runs in wins")
    st.dataframe(average_runs_in_wins.toPandas())
    
    average_runs_pd = average_runs_in_wins.toPandas()

    # Using seaborn to plot average runs in winning matches
    plt.figure(figsize=(12, 8))
    top_scorers = average_runs_pd.nlargest(10, 'avg_runs_in_wins')
    sns.barplot(x='player_name', y='avg_runs_in_wins', data=top_scorers)
    plt.title('Average Runs Scored by Batsmen in Winning Matches (Top 10 Scorers)')
    plt.xlabel('Player Name')
    plt.ylabel('Average Runs in Wins')
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(plt)

    scores_by_venue = spark.sql("""
        SELECT venue_name, AVG(total_runs) AS average_score, MAX(total_runs) AS highest_score
        FROM (
            SELECT ball_by_ball.match_id, match.venue_name, SUM(runs_scored) AS total_runs
            FROM ball_by_ball
            JOIN match ON ball_by_ball.match_id = match.match_id
            GROUP BY ball_by_ball.match_id, match.venue_name
            )
        GROUP BY venue_name
        ORDER BY average_score DESC
    """)

    st.title("Scores by Venue")
    st.dataframe(scores_by_venue.toPandas())
    scores_by_venue_pd = scores_by_venue.toPandas()
    plt.figure(figsize=(14, 8))
    sns.barplot(x='average_score', y='venue_name', data=scores_by_venue_pd)
    plt.title('Distribution of Scores by Venue')
    plt.xlabel('Average Score')
    plt.ylabel('Venue')
    st.pyplot(plt)

    dismissal_types = spark.sql("""
        SELECT out_type, COUNT(*) AS frequency
        FROM ball_by_ball
        WHERE out_type IS NOT NULL
        GROUP BY out_type
        ORDER BY frequency DESC
    """)
    st.title("Dissimal Types")
    st.dataframe(dismissal_types.toPandas())
    dismissal_types_pd = dismissal_types.toPandas()

    plt.figure(figsize=(12, 6))
    sns.barplot(x='frequency', y='out_type', data=dismissal_types_pd, palette='pastel')
    plt.title('Most Frequent Dismissal Types')
    plt.xlabel('Frequency')
    plt.ylabel('Dismissal Type')
    st.pyplot(plt)

    team_toss_win_performance = spark.sql("""
    SELECT team1, COUNT(*) AS matches_played, SUM(CASE WHEN toss_winner = match_winner THEN 1 ELSE 0 END) AS wins_after_toss
    FROM match
    WHERE toss_winner = team1
    GROUP BY team1
    ORDER BY wins_after_toss DESC
    """)
    
    st.title("Toss winning performance")
    st.dataframe(team_toss_win_performance.toPandas())
    team_toss_win_pd = team_toss_win_performance.toPandas()

    plt.figure(figsize=(12, 8))
    sns.barplot(x='wins_after_toss', y='team1', data=team_toss_win_pd)
    plt.title('Team Performance After Winning Toss')
    plt.xlabel('Wins After Winning Toss')
    plt.ylabel('Team')
    st.pyplot(plt)

    top_scoring_batsmen_per_season = spark.sql("""
    SELECT 
    p.player_name,
    m.season_year,
    SUM(b.runs_scored) AS total_runs 
    FROM ball_by_ball b
    JOIN match m ON b.match_id = m.match_id   
    JOIN player_match pm ON m.match_id = pm.match_id AND b.striker = pm.player_id     
    JOIN player p ON p.player_id = pm.player_id
    GROUP BY p.player_name, m.season_year
    ORDER BY m.season_year, total_runs DESC
    """)
    top_scoring_batsmen_per_season_pd = top_scoring_batsmen_per_season.toPandas()
    st.title("Top scoring batsman per season")
    st.dataframe(top_scoring_batsmen_per_season.toPandas())
    

    plt.figure(figsize=(12, 6))
    sns.barplot(x='season_year', y='total_runs', data=top_scoring_batsmen_per_season_pd)
    plt.title('Top scoring batsman')
    plt.xlabel('season')
    plt.ylabel('Runs')
    st.pyplot(plt)

    if st.button("Go Back to Home"):
        st.session_state.page = "home"


if st.session_state.page == 'home':
    show_home_page()
elif st.session_state.page == 'analysis':
    show_analysis_page()
elif st.session_state.page == 'analysis_dashboard':
    show_analysis_dashboard()
