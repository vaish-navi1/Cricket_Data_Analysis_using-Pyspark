from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import RandomForestClassifier, LogisticRegression, GBTClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# Initialize Spark session
spark = SparkSession.builder.appName("CricketClassification").getOrCreate()

# Load datasets
ball_by_ball = spark.read.csv(r"C:\Users\vaish\Downloads\cricket\Ball_By_Ball.csv", header=True, inferSchema=True)
player = spark.read.csv(r"C:\Users\vaish\Downloads\cricket\Player.csv", header=True, inferSchema=True)
match = spark.read.csv(r"C:\Users\vaish\Downloads\cricket\Match.csv", header=True, inferSchema=True)
player_match = spark.read.csv(r"C:\Users\vaish\Downloads\cricket\Player_Match.csv", header=True, inferSchema=True)

# Merge datasets
data = player_match.join(match, player_match.Match_Id == match.match_id, "inner")
data = data.join(ball_by_ball, player_match.Match_Id == ball_by_ball.MatcH_id, "inner")

# Drop rows with NaN values in the target variable
data = data.na.drop(subset=["Win_Margin"])

# Features and target variable
assembler = VectorAssembler(inputCols=['Over_id', 'Ball_id', 'Innings_No', 'BowlingTeam_SK'], outputCol="features")
indexer = StringIndexer(inputCol="Win_Margin", outputCol="label")

# Prepare data for ML
data = assembler.transform(data)
data = indexer.fit(data).transform(data)

# Split the dataset
train_data, test_data = data.randomSplit([0.8, 0.2], seed=42)

# RandomForestClassifier
rf = RandomForestClassifier(featuresCol="features", labelCol="label", numTrees=100, seed=42)
rf_model = rf.fit(train_data)
rf_predictions = rf_model.transform(test_data)

# LogisticRegression
lr = LogisticRegression(featuresCol="features", labelCol="label", maxIter=100)
lr_model = lr.fit(train_data)
lr_predictions = lr_model.transform(test_data)

# GradientBoostingClassifier
gb = GBTClassifier(featuresCol="features", labelCol="label", maxIter=50, seed=42)
gb_model = gb.fit(train_data)
gb_predictions = gb_model.transform(test_data)

# XGBoost Classifier
# Convert Spark DataFrame to Pandas to work with XGBoost
X_train = train_data.select("features").rdd.map(lambda row: row[0]).collect()
y_train = train_data.select("label").rdd.map(lambda row: row[0]).collect()

X_test = test_data.select("features").rdd.map(lambda row: row[0]).collect()
y_test = test_data.select("label").rdd.map(lambda row: row[0]).collect()

# Initialize and fit XGBoost model
xgb = XGBClassifier(random_state=42)
xgb.fit(X_train, y_train)

# Make predictions
xgb_predictions = xgb.predict(X_test)

# XGBoost accuracy using sklearn accuracy_score
xgb_accuracy = accuracy_score(y_test, xgb_predictions)

# Evaluator for accuracy (for other classifiers)
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")

rf_accuracy = evaluator.evaluate(rf_predictions)
lr_accuracy = evaluator.evaluate(lr_predictions)
gb_accuracy = evaluator.evaluate(gb_predictions)

# Print accuracy
print("Random Forest Accuracy:", rf_accuracy)
print("Logistic Regression Accuracy:", lr_accuracy)
print("Gradient Boosting Accuracy:", gb_accuracy)
print("XGBoost Accuracy:", xgb_accuracy)

# Create DataFrame for accuracy comparison
accuracy_data = [(rf_accuracy, "Random Forest"), (lr_accuracy, "Logistic Regression"), 
                 (gb_accuracy, "Gradient Boosting"), (xgb_accuracy, "XGBoost")]
accuracy_df = spark.createDataFrame(accuracy_data, ["Accuracy", "Model"])

# Show the accuracy comparison
accuracy_df.show()

# Stop the Spark session
spark.stop()
