import streamlit as st  # For building the web app interface
import pandas as pd  # Equivalent to Spark DataFrames for data manipulation
import numpy as np  # For numerical operations if needed

# DataFrame manipulation and functions are handled by pandas instead of pyspark.sql
from datetime import datetime  # For handling date types
from sklearn.preprocessing import LabelEncoder  # For encoding categorical data

# Plotting libraries remain the same
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import pandas as pd
import streamlit as st


# Function to load data based on the selected file
def load_data(file_name):
    if file_name == "Ball by Ball Data":
        return pd.read_csv("C:/Users/vaish/Downloads/cricket/Ball_By_Ball.csv", encoding='ISO-8859-1')
    elif file_name == "Match Data":
        return pd.read_csv("C:/Users/vaish/Downloads/cricket/Match.csv", encoding='ISO-8859-1')
    elif file_name == "Player Data":
        return pd.read_csv("C:/Users/vaish/Downloads/cricket/Player.csv", encoding='ISO-8859-1')
    elif file_name == "Player Match Data":
        return pd.read_csv("C:/Users/vaish/Downloads/cricket/Player_Match.csv", encoding='ISO-8859-1')
    elif file_name == "Team Data":
        return pd.read_csv("C:/Users/vaish/Downloads/cricket/Team.csv", encoding='ISO-8859-1')

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
   
    # Display the DataFrame as a table
    st.write(f"### Data from {selected_file}")
    st.dataframe(data_df)
    

    # Optionally, display a sample of the DataFrame
    if st.checkbox("Show sample data"):
        st.write(data_df.sample(10))  

    if st.button("Go to Analysis"):
        st.session_state.page = "analysis"

    if st.button("Classifier"):
        st.session_state.page = "classifier"
    
# Load CSV data into pandas DataFrame
def loade_data(file_path):
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
        ball_by_ball_df = loade_data("C:/Users/vaish/Downloads/cricket/Ball_By_Ball.csv")
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
        ball_by_ball_df = loade_data("C:/Users/vaish/Downloads/cricket/Ball_By_Ball.csv")
        filtered_ball_by_ball_df = ball_by_ball_df[(ball_by_ball_df['Wides'] == 0) & (ball_by_ball_df['Noballs'] == 0)]

        # Calculate high impact balls based on runs and wickets
        filtered_ball_by_ball_df['high_impact'] = ((filtered_ball_by_ball_df['Runs_Scored'] + filtered_ball_by_ball_df['Extra_runs']) > 6) | (filtered_ball_by_ball_df['Bowler_Wicket'] == True)

        # Display high impact balls
        st.write("### High Impact Balls:")
        st.dataframe(filtered_ball_by_ball_df[filtered_ball_by_ball_df['high_impact']])

    elif selected_analysis == "Win Margin Category":
        # Load the "Match" dataset and categorize win margins
        match_df = loade_data("C:/Users/vaish/Downloads/cricket/Match.csv")
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
        match_df = loade_data("C:/Users/vaish/Downloads/cricket/Match.csv")
        match_df['toss_match_winner'] = match_df.apply(
            lambda row: 'Correct' if row['Toss_Winner'] == row['match_winner'] else 'Incorrect', axis=1
        )

        # Display toss match winner results
        st.write("### Toss Match Winner Results:")
        st.dataframe(match_df[['match_id', 'Toss_Winner', 'match_winner', 'toss_match_winner']])

    if st.button("Go Back to Home"):
        st.session_state.page = "home"
def show_classifier():
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from xgboost import XGBClassifier
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import LabelEncoder
    import streamlit as st

    # Load datasets
    ball_by_ball = pd.read_csv(r"C:\Users\vaish\Downloads\cricket\Ball_By_Ball.csv", encoding='ISO-8859-1')
    player = pd.read_csv(r"C:\Users\vaish\Downloads\cricket\Player.csv", encoding='ISO-8859-1')
    match = pd.read_csv(r"C:\Users\vaish\Downloads\cricket\Match.csv", encoding='ISO-8859-1')
    player_match = pd.read_csv(r"C:\Users\vaish\Downloads\cricket\Player_Match.csv", encoding='ISO-8859-1')
    team = pd.read_csv(r"C:\Users\vaish\Downloads\cricket\Team.csv", encoding='ISO-8859-1')

    # Merge datasets
    data = player_match.merge(match, left_on='Match_Id', right_on='match_id', how='inner')
    data = data.merge(ball_by_ball, left_on='Match_Id', right_on='MatcH_id', suffixes=('', '_bb'), how='inner')

    # Check for NaN values in the target variable
    st.write("NaN values in Win_Margin:", data['Win_Margin'].isnull().sum())

    # Drop rows with NaN values in the target variable
    data = data.dropna(subset=['Win_Margin'])

    # Reduce dataset size for faster testing (optional)
    data = data.sample(frac=0.1, random_state=42)  # Use 10% of the data

    # Features and target variable
    features = data[['Over_id', 'Ball_id', 'Innings_No', 'BowlingTeam_SK']]
    target = data['Win_Margin']

    # One-hot encode categorical features
    features = pd.get_dummies(features, drop_first=True)

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    # RandomForestClassifier
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_predictions = rf_model.predict(X_test)
    rf_accuracy = accuracy_score(y_test, rf_predictions)

    st.title("### Random Forest Classifier")
    st.write("Confusion Matrix:")
    st.write(confusion_matrix(y_test, rf_predictions))

    class_report = classification_report(y_test, rf_predictions, output_dict=True)
    
    st.write("Classification Report:")
    st.dataframe(class_report)
    st.write(f"Accuracy: {rf_accuracy}")

    # Logistic Regression
    logistic_model = LogisticRegression(max_iter=100, random_state=42)
    logistic_model.fit(X_train, y_train)
    logistic_predictions = logistic_model.predict(X_test)
    logistic_accuracy = accuracy_score(y_test, logistic_predictions)

    st.title("### Logistic Regression")
    st.write("Confusion Matrix:")
    st.write(confusion_matrix(y_test, logistic_predictions))

    classi_report = classification_report(y_test, logistic_predictions, output_dict=True)
    
    st.write("Classification Report:")
    st.dataframe(classi_report)
    st.write(f"Accuracy: {logistic_accuracy}")

    # GradientBoostingClassifier
    gb_model = GradientBoostingClassifier(n_estimators=50, random_state=42)
    gb_model.fit(X_train, y_train)
    gb_predictions = gb_model.predict(X_test)
    gb_accuracy = accuracy_score(y_test, gb_predictions)

    st.title("### Gradient Boosting Classifier")
    st.write("Confusion Matrix:")
    st.write(confusion_matrix(y_test, gb_predictions))

    classi_report = classification_report(y_test, gb_predictions, output_dict=True)
    
    st.write("Classification Report:")
    st.dataframe(classi_report)
    st.write(f"Accuracy: {gb_accuracy}")

    # XGBoostingClassifier
    # Label encode the target variable (Win_Margin)
    label_encoder = LabelEncoder()
    data['Win_Margin_Encoded'] = label_encoder.fit_transform(data['Win_Margin'])

    # Ensure that the target variable is integer
    data['Win_Margin_Encoded'] = data['Win_Margin_Encoded'].astype(int)

    # Features and target variable
    features = data[['Over_id', 'Ball_id', 'Innings_No', 'BowlingTeam_SK']]
    target = data['Win_Margin_Encoded']

    # One-hot encode categorical features (if needed)
    features = pd.get_dummies(features, drop_first=True)

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    # Initialize XGBoost model
    xgb_model = XGBClassifier(random_state=42)
    xgb_model.fit(X_train, y_train)

    # Make predictions
    xgb_predictions = xgb_model.predict(X_test)
    xgb_accuracy = accuracy_score(y_test, xgb_predictions)

    st.title("### XGB Classifier")
    st.write("Confusion Matrix:")
    st.write(confusion_matrix(y_test, xgb_predictions))

    classif_report = classification_report(y_test, xgb_predictions, output_dict=True)
    st.write("Classification Report:")
    st.dataframe(classif_report)
    st.write(f"Accuracy: {xgb_accuracy}")

    # Create DataFrame for accuracy comparison
    accuracy_data = {
        'Model': ['Random Forest', 'Logistic Regression', 'Gradient Boosting','XGB Classifier'],
        'Accuracy': [rf_accuracy, logistic_accuracy, gb_accuracy,xgb_accuracy]
    }

    accuracy_df = pd.DataFrame(accuracy_data)

    # Plot box plot of accuracy
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='Model', y='Accuracy', data=accuracy_df, palette="Set2")
    plt.title('Accuracy Comparison of Classifiers')
    plt.ylim(0, 0.8)  # Adjust the accuracy range based on results
    plt.ylabel('Accuracy')
    plt.xlabel('Classifier')

    st.pyplot(plt)
     
    if st.button("Go Back to Home"):
        st.session_state.page = "home"

if st.session_state.page == 'home':
        show_home_page()
elif st.session_state.page == 'analysis':
        show_analysis_page()
elif st.session_state.page == 'classifier':
        show_classifier()
