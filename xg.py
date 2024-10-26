from xgboost import XGBClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder


ball_by_ball = pd.read_csv(r"C:\Users\vaish\Downloads\cricket\Ball_By_Ball.csv", encoding='ISO-8859-1')
player = pd.read_csv(r"C:\Users\vaish\Downloads\cricket\Player.csv", encoding='ISO-8859-1')
match = pd.read_csv(r"C:\Users\vaish\Downloads\cricket\Match.csv", encoding='ISO-8859-1')
player_match = pd.read_csv(r"C:\Users\vaish\Downloads\cricket\Player_Match.csv", encoding='ISO-8859-1')
team = pd.read_csv(r"C:\Users\vaish\Downloads\cricket\Team.csv", encoding='ISO-8859-1')

data = player_match.merge(match, left_on='Match_Id', right_on='match_id', how='inner')
data = data.merge(ball_by_ball, left_on='Match_Id', right_on='MatcH_id', suffixes=('', '_bb'), how='inner')


print("NaN values in Win_Margin:", data['Win_Margin'].isnull().sum())


data = data.dropna(subset=['Win_Margin'])


label_encoder = LabelEncoder()
data['Win_Margin_Encoded'] = label_encoder.fit_transform(data['Win_Margin'])


features = data[['Over_id', 'Ball_id', 'Innings_No', 'BowlingTeam_SK']]
target = data['Win_Margin_Encoded']


features = pd.get_dummies(features, drop_first=True)


X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)


xgb_model = XGBClassifier(random_state=42)
xgb_model.fit(X_train, y_train)


xgb_predictions = xgb_model.predict(X_test)


print("XGBoost Confusion Matrix:")
print(confusion_matrix(y_test, xgb_predictions))

print("\nXGBoost Classification Report:")   
print(classification_report(y_test, xgb_predictions))
