import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

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
print("NaN values in Win_Margin:", data['Win_Margin'].isnull().sum())

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

# Initialize and fit the Logistic Regression model
logistic_model = LogisticRegression(max_iter=10, random_state=42)  # Set max_iter to 100 for quicker fitting
logistic_model.fit(X_train, y_train)

# Make predictions
logistic_predictions = logistic_model.predict(X_test)

# Print confusion matrix and classification report for Logistic Regression
print("Logistic Regression Confusion Matrix:")
print(confusion_matrix(y_test, logistic_predictions))

print("\nLogistic Regression Classification Report:")
print(classification_report(y_test, logistic_predictions))
