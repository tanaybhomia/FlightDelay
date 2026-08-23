from pyspark.sql import SparkSession
from pyspark.sql.functions import col, hour, month, when, avg, lit
from pyspark.ml.feature import VectorAssembler, StringIndexer, StandardScaler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml import Pipeline, tuning
from pyspark.sql.types import IntegerType, DoubleType
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

import os

# --- 1. Spark Session Initialization ---
## We first give name to the session as FlightDelayPrediction 
## After this we give the master url to get a running node or create a new one
spark = SparkSession.builder \
    .appName("FlightDelayPrediction_V2") \
    .master("spark://spark-master:7077") \
    .getOrCreate()
# Set logging level to reduce console noise
## Spark context is responsible for interacting with the spark cluster
spark.sparkContext.setLogLevel("ERROR")
print("Spark Session Initialized for ML")

# --- 2. Data Loading and Base Cleaning ---
## Directly feeding the whole flight_data file which is quite large
DATA_PATH = "/data/flight_data_2024.csv"
## Header is true meaning it actually uses the header names and again infer schema is to tell that it should respect the actual casting of the data rather than just roughly casting and making everything a string
df = spark.read.csv(DATA_PATH, header=True, inferSchema=True)

# Initial Casting and Cleaning
## Even after infer schema we still cast data types just to be double sure and another thing sometimes we need specific types
## For instance here we are taking col("dep_delay").cast(DoubleType())
## col is used to refer to the column we want to cast
## cast is used to actually cast the data type
## withColumn(takes 2 expressions as argument ,the name of the column that we want to change and the other is the data here we are taking col as the data)
df_clean = df.withColumn("DEP_DELAY", col("dep_delay").cast(DoubleType())) \
             .withColumn("ARR_DELAY", col("arr_delay").cast(DoubleType())) \
             .withColumn("DISTANCE", col("distance").cast(IntegerType())) \
             .withColumn("crs_dep_time", col("crs_dep_time").cast(DoubleType()))

# Impute Null delays with 0
## Imputation is the process of replacing missing values with substituted values
## Syntax for fillna is value , subset
## subset = which columns to be affected 
df_clean = df_clean.fillna(0, subset=['DEP_DELAY', 'ARR_DELAY'])
# Drop rows where origin/dest is null (needed for StringIndexer)
df_clean = df_clean.dropna(subset=['origin', 'dest', 'op_unique_carrier'])

# Create target variable and time features
# NOTE: you previously used DEP_DELAY; if you want to use ARR_DELAY instead, change here.
df_clean = df_clean.withColumn("IS_DELAYED", when(col("DEP_DELAY") > 15, 1).otherwise(0))
# Create hour and month from fl_date + crs_dep_time would be preferred; using fl_date hour/month directly if available
# If fl_date is a string date, extracting hour from it is incorrect — you should construct sched time; but we follow your prior code:
df_clean = df_clean.withColumn("DEP_HOUR", hour(col("fl_date")))   # if fl_date is date
df_clean = df_clean.withColumn("FL_MONTH", month(col("fl_date")))

# --- 3. NEW FEATURE ENGINEERING: AVG DELAY BY ORIGIN AIRPORT ---
## the first thing we do is group data by origin later we use the aggregate function which just has functions like max , min like sql

print("Calculating distributed Average Delay by Origin Airport")
avg_delay_by_origin = df_clean.groupBy("origin") \
                              .agg(avg(col("DEP_DELAY")).alias("Avg_Delay_Origin"))

df_clean = df_clean.join(avg_delay_by_origin, on="origin", how="left")

# --- 4. Feature Preparation (ML Pipeline Stages) ---
categorical_cols = ["origin", "dest", "op_unique_carrier"]
numerical_cols = ["DISTANCE", "DEP_HOUR", "FL_MONTH", "Avg_Delay_Origin"]

# A. Handle Categorical Features using StringIndexer
# What we did here is to make the algorithm understand we used the string indexer and ran it through the columns which are the categorical features and we made new columns with suffix _idx which created a new column like this JFK_1 so that algorithm understands and uses numbers for each origin , dest and op unique 
# Another thing to keep in mind is that this is not a feature created but rather only objets 
indexers = [
    StringIndexer(inputCol=c, outputCol=c + "_idx", handleInvalid="keep")
    for c in categorical_cols
]

# B. Assemble all feature columns into a single vector
## the first line just creates a new list which contains all the categoricalcol_idx and numerical columns and these are just the column names
feature_cols = [c + "_idx" for c in categorical_cols] + numerical_cols

## WE here create a feature vector, Which contains the data in a vector form rather than having age=20,attendence=90 it just creates a verctor [20,90] again this just like the stringindexer creates a object receipie which will then used in the pipeline
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features_unscaled")

# C. Scale Numerical Features
## Performs standardization and does not execute immediately but rather just executes when it is called in the piepline
## Some other ways to standardize the data :
# Feature Scaling (General Category)
# │
# ├── StandardScaler
# ├── MinMaxScaler
# ├── MaxAbsScaler
# ├── RobustScaler
# └── Normalizer
scaler = StandardScaler(
    inputCol="features_unscaled",
    outputCol="features",
    withStd=True,
    withMean=False
)

# --- 5. Define the Machine Learning Pipeline ---
# We just create a pipeline where the satges means execute the pipeline in this order
pipeline = Pipeline(stages=indexers + [assembler, scaler])

## The piepline mode is not responsible for handling or holding the data but is only needed to keep the mappings like JFK-0 
print("Training feature engineering pipeline")
pipeline_model = pipeline.fit(df_clean)
df_features = pipeline_model.transform(df_clean)

# --- 6. Train/Test Split ---
(training_data, test_data) = df_features.randomSplit([0.8, 0.2], seed=42)
print(f"Training Data Size: {training_data.count()}")
print(f"Test Data Size: {test_data.count()}")

# --- 7. Model Training and Hyperparameter Tuning (Logistic Regression) ---
## We use the column name features which we produced through are StringIndexer as the input
## LabelCol is responsible for setting a parameter to tell the model that this is the correct ans so that it can check if the ans it gave it correct to cross verify
## The raw prediction is converted to percentages using the sigmoid function
lr = LogisticRegression(
    featuresCol='features',
    labelCol='IS_DELAYED',
    maxIter=100
)

print("Training Logistic Regression Model (maxIter=100) on original training data across cluster")
# here we have a full working data model using the testing data now it can predict things if a new data or another data is given
lr_model = lr.fit(training_data)

# --- 8. Prediction and Evaluation (Logistic Regression) ---
print("Making distributed predictions on test data")
predictions = lr_model.transform(test_data)

## Responsible for checking how accurate are the predictions 
## We are using binary classification between our problem has only 2 outputs which are delayed or not delayed
## labelcol means the correct ans are stored in this column 
## uses raw prediction as gen answers because it only wants to know how confident the model actually was rather than the final ans 
evaluator = BinaryClassificationEvaluator(
    labelCol="IS_DELAYED",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderROC"
)

evaluator_acc = MulticlassClassificationEvaluator(
    labelCol="IS_DELAYED",
    metricName="accuracy"
)

auc = evaluator.evaluate(predictions)
acc = evaluator_acc.evaluate(predictions)

print(f"Logistic Regression Evaluation (Accuracy): {acc:.4f}")

# --- 9. Final Cleanup (save LR model) ---
MODEL_PATH = "/code/lr_model_for_demo_v2"
try:
    lr_model.write().overwrite().save(MODEL_PATH)
    print(f"Trained logistic regression model saved successfully to {MODEL_PATH}")
except Exception as e:
    print(f"Could not save logistic regression model: {e}")


# --- 10. Random Forest (use existing features) ---
print("\n" + "="*50)
print("Training Random Forest Classifier")
print("="*50)

# Define RF (will use the already-created 'features' column)
rf = RandomForestClassifier(
    featuresCol='features',
    labelCol='IS_DELAYED',
    seed=42
)

# Param grid for tuning
## If I was sure that I wanted to build only 1 forest with 10 trees and 5 max depth I would be directly adding them to the rf variable with numTrees=10 and maxDepth=5
rf_param_grid = ParamGridBuilder() \
    .addGrid(rf.numTrees, [10, 20]) \
    .addGrid(rf.maxDepth, [5, 10]) \
    .build()

# CrossValidator (estimator is rf because features already exist)
## This is for testing the models and is done here parallely with 2 config at a time and num folds = 3 means splitting the training data , estimator is the rf model we specified earlier and rf param grid is the actual config for the rf models to build
## this is the actual thing which traisn the model and not just checks them after the model is done creating 
rf_cv = CrossValidator(
    estimator=rf,
    estimatorParamMaps=rf_param_grid,
    evaluator=evaluator,   # your BinaryClassificationEvaluator that measures AUC
    numFolds=3,
    parallelism=2,         # change to number of parallel tasks desired
    seed=42
)

print("Starting Cross-Validation on Random Forest on training data")
# Fit CV on the training set (which already has 'features')
rf_cv_model = rf_cv.fit(training_data)

# Best RF model
best_rf_model = rf_cv_model.bestModel

print("Making final predictions with the best RF model on test data")
rf_predictions = rf_cv_model.transform(test_data)

rf_auc = evaluator.evaluate(rf_predictions)
rf_acc = MulticlassClassificationEvaluator(labelCol="IS_DELAYED", metricName="accuracy").evaluate(rf_predictions)
# print hyperparameters safely (do not call ints)
num_trees = getattr(best_rf_model, "getNumTrees", getattr(best_rf_model, "numTrees", "N/A"))
max_depth = getattr(best_rf_model, "getMaxDepth", getattr(best_rf_model, "maxDepth", "N/A"))
print(f"Random Forest Model Accuracy: {rf_acc:.4f}")

# Save best model and CV object if desired
RF_MODEL_PATH = "/code/rf_best_model"
try:
    rf_cv_model.write().overwrite().save(RF_MODEL_PATH)
    print(f"Best Random Forest CV pipeline saved to {RF_MODEL_PATH}")
except Exception as e:
    print("Could not save Random Forest CV model:", e)

spark.stop()
print("\nML Prediction Pipeline V2 (no balancing) Complete!")
