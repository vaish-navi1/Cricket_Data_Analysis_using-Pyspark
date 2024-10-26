import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Load data
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

# Drop rows with NaN values in target
data = data.dropna(subset=['Win_Margin'])

features = data[['Over_id', 'Ball_id', 'Innings_No', 'BowlingTeam_SK']]
target = data['Win_Margin']

# Handle categorical data (one-hot encoding)
features = pd.get_dummies(features, drop_first=True)

# Optional: Sample a smaller subset for quicker testing
# features, target = features.sample(n=5000, random_state=42), target.sample(n=5000, random_state=42)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Train the Gradient Boosting Classifier
gb_model = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)  # Adjusted parameters
gb_model.fit(X_train, y_train)
gb_predictions = gb_model.predict(X_test)

# Output results
print("Gradient Boosting Confusion Matrix:")
print(confusion_matrix(y_test, gb_predictions))

print("\nGradient Boosting Classification Report:")
print(classification_report(y_test, gb_predictions))
