import streamlit as st
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, BooleanType, DateType
from pyspark.sql.functions import col, sum, avg, when
from pyspark.sql import Window
import matplotlib.pyplot as plt
import seaborn as sns

spark = SparkSession.builder \
    .appName("Cricket Analysis") \
    .master("local[*]") \
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

def show_classifier():
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder
    from pyspark.sql import SparkSession
    from pyspark.ml.feature import VectorAssembler, StringIndexer
    from pyspark.ml.classification import RandomForestClassifier, LogisticRegression, GBTClassifier
    from pyspark.ml.evaluation import MulticlassClassificationEvaluator
    from pyspark.ml import Pipeline
    from xgboost import XGBClassifier
    from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
    from pyspark.sql import SparkSession
    from pyspark.ml.feature import VectorAssembler, StringIndexer
    from pyspark.ml.classification import RandomForestClassifier, LogisticRegression, GBTClassifier
    from pyspark.ml.evaluation import MulticlassClassificationEvaluator
    from pyspark.ml import Pipeline
    from xgboost import XGBClassifier

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

    accuracy_data = {
        'Model': ['Random Forest', 'Logistic Regression', 'Gradient Boosting','XGB Classifier'],
        'Accuracy': [rf_accuracy, lr_accuracy,gb_accuracy,xgb_accuracy]
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
elif st.session_state.page == 'analysis_dashboard':
    show_classifier()
