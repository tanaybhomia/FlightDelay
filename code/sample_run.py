from pyspark.sql import SparkSession
from pyspark.sql.functions import col, hour, month, when, avg
from pyspark.ml.feature import VectorAssembler, StringIndexer, StandardScaler
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline

spark = SparkSession.builder.appName("SampleRun").master("spark://spark-master:7077").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

DATA_PATH = "/data/flight_data_2024.csv"
# Sample 200,000 rows to make it fast but representative
df = spark.read.csv(DATA_PATH, header=True, inferSchema=True).limit(200000)

df_clean = df.withColumn("DEP_DELAY", col("dep_delay").cast("double")) \
             .withColumn("ARR_DELAY", col("arr_delay").cast("double")) \
             .withColumn("DISTANCE", col("distance").cast("int")) \
             .fillna(0, subset=['DEP_DELAY', 'ARR_DELAY']) \
             .dropna(subset=['origin', 'dest', 'op_unique_carrier']) \
             .withColumn("IS_DELAYED", when(col("DEP_DELAY") > 15, 1).otherwise(0)) \
             .withColumn("DEP_HOUR", hour(col("fl_date"))) \
             .withColumn("FL_MONTH", month(col("fl_date")))

avg_delay = df_clean.groupBy("origin").agg(avg(col("DEP_DELAY")).alias("Avg_Delay_Origin"))
df_clean = df_clean.join(avg_delay, on="origin", how="left")

categorical_cols = ["origin", "dest", "op_unique_carrier"]
numerical_cols = ["DISTANCE", "DEP_HOUR", "FL_MONTH", "Avg_Delay_Origin"]

indexers = [StringIndexer(inputCol=c, outputCol=c + "_idx", handleInvalid="keep") for c in categorical_cols]
assembler = VectorAssembler(inputCols=[c + "_idx" for c in categorical_cols] + numerical_cols, outputCol="features_unscaled")
scaler = StandardScaler(inputCol="features_unscaled", outputCol="features", withStd=True, withMean=False)

pipeline = Pipeline(stages=indexers + [assembler, scaler])
df_features = pipeline.fit(df_clean).transform(df_clean)

train, test = df_features.randomSplit([0.8, 0.2], seed=42)

# LR
lr = LogisticRegression(featuresCol='features', labelCol='IS_DELAYED', maxIter=10)
lr_model = lr.fit(train)
lr_acc = MulticlassClassificationEvaluator(labelCol="IS_DELAYED", metricName="accuracy").evaluate(lr_model.transform(test))
print(f"--- LOGISTIC REGRESSION ACCURACY: {lr_acc:.4f} ---")

# RF
rf = RandomForestClassifier(featuresCol='features', labelCol='IS_DELAYED', numTrees=10, maxDepth=5, seed=42)
rf_model = rf.fit(train)
rf_acc = MulticlassClassificationEvaluator(labelCol="IS_DELAYED", metricName="accuracy").evaluate(rf_model.transform(test))
print(f"--- RANDOM FOREST ACCURACY: {rf_acc:.4f} ---")

spark.stop()
