from pyspark.sql import SparkSession
from pyspark.sql.functions import col, year, month, hour, when
from pyspark.sql.types import IntegerType, DoubleType
from pyspark.sql.functions import greatest, sum as spark_sum
from pyspark.sql.functions import udf
from pyspark.sql.types import IntegerType


# --- 1. Spark Session Initialization ---
# The master URL must match the service name defined in docker-compose.yml (spark-master)
spark = SparkSession.builder \
    .appName("FlightDelayAnalysis") \
    .master("spark://spark-master:7077") \
    .getOrCreate()

# Set logging level to reduce console noise
spark.sparkContext.setLogLevel("ERROR")

print("--- Spark Session Initialized ---")

# --- 2. Distributed Data Loading (Reading the CSV) ---
# The path must be the volume mount path inside the container: /data/
DATA_PATH = "/data/flight_data_2024.csv"

# InferSchema=True is convenient but slower on huge data; we use it here for simplicity.
# header=True ensures the first row is treated as column names.
## Infer schema basically roughly casts the data rather than infering the whole as a integer or float etc
df = spark.read.csv(DATA_PATH, header=True, inferSchema=True)

print(f"Total records loaded: {df.count()}")
print("Schema:")
df.printSchema()

# --- 3. Initial Data Cleaning and Type Casting ---
# Cast key columns used for calculations to the correct numeric types

## Manually casting the columsn with a datatype. 
## withColumn("renaming the column") 
## col("the column we are taking the data from")
## cast("actually casting the datatype")
df_clean = df.withColumn("DEP_DELAY", col("DEP_DELAY").cast(DoubleType())) \
             .withColumn("ARR_DELAY", col("ARR_DELAY").cast(DoubleType())) \
             .withColumn("AIR_TIME", col("AIR_TIME").cast(IntegerType())) \
             .withColumn("DISTANCE", col("DISTANCE").cast(IntegerType()))

# Drop columns that are irrelevant or introduce leakage (like specific flight IDs or cancellation codes)

## using .drop to just drop the column names
## we use * while giving the array of columns_to_drop as an argument because if we dont then we will be giving full list as an output rather then giving singlhen(col("DEP_DELAY") > 15, 1).otherwise(0) #e columns

columns_to_drop = ["FL_NUM", "UNIQUE_CARRIER", "TAIL_NUM", "CANCELLATION_CODE"]
df_clean = df_clean.drop(*columns_to_drop)

# Handle missing/Null values in the key analysis column (DEP_DELAY)
# A simple strategy for this project: Impute missing delays with 0 (assuming they were on time if data is missing).
# For a more rigorous project, you might drop these rows or use a machine learning imputation.
df_clean = df_clean.fillna(0, subset=['DEP_DELAY', 'ARR_DELAY'])

# --- 4. Feature Engineering for Analysis ---
# Create a binary column for 'is_delayed'
## here we just create a new column called is delayed which is a boolean column for showing if the flight is delayed or not
## 0 if the flight is not delayed and 1 if it is , Taking 15 minutes as the industry standard

df_clean = df_clean.withColumn(
    "IS_DELAYED",
    when(col("DEP_DELAY") > 15, 1).otherwise(0) ## Industry standard for 'delay' is usually 15 minutes
)

# Extract time components for time-based analysis (e.g., peak delay hour)
# Use 'fl_date' (the timestamp column) to reliably get the hour and month.
## hour is used to extract the hour from the fl_date
## month is used to extract the month from the fl_date

df_clean = df_clean.withColumn("DEP_HOUR", hour(col("fl_date")))
df_clean = df_clean.withColumn("FL_MONTH", month(col("fl_date")))


# Show a summary of the cleaned data
print("--- Cleaned and Engineered Data Schema ---")
df_clean.printSchema()

# Keep the DataFrame for the next analysis steps
# spark.stop() # Do not stop yet, as we will use df_clean next


# --- ANALYSIS 1: Top 10 Most Delayed Carriers ---
print("\n--- Running Analysis 1: Top 10 Most Delayed Carriers ---")

# We filter for only delayed flights (DEP_DELAY > 0) to get a true measure of delay performance.
## Here we first filter the rows which have dep_delay greater than zero
## then we group by carrier and take the avg of the delay and total delayed flights
## then we rename the columns to be more readable
## then we order by avg delay in descending order
## then we take the top 10
carrier_delay_df = df_clean.filter(col("DEP_DELAY") > 0) \
                           .groupBy("op_unique_carrier") \
                           .agg({"DEP_DELAY": "avg", "IS_DELAYED": "count"}) \
                           .withColumnRenamed("avg(DEP_DELAY)", "AVG_DEP_DELAY_MIN") \
                           .withColumnRenamed("count(IS_DELAYED)", "TOTAL_DELAYED_FLIGHTS") \
                           .orderBy(col("AVG_DEP_DELAY_MIN").desc()) \
                           .limit(10)

# Collect the distributed results to the driver program for visualization (since the result is small)
carrier_delay_df.show()
print("--- Analysis 1 Result (Top 10 Carriers): ---")

# Save the result to a CSV file in your local 'data' folder (mounted to /data)
# Spark writes distributed output into a folder with multiple part-files.
carrier_delay_df.write.mode("overwrite").csv("/data/results/top_carriers", header=True)
print("Saved analysis results to /data/results/top_carriers")


# --- ANALYSIS 2: Delay Count by Hour of Day ---
print("\n--- Running Analysis 2: Delay Count by Hour of Day ---")

# Ensure DEP_HOUR is correctly derived from CRS_DEP_TIME (which is in HHMM format)
# Example: 530 → 5, 1430 → 14
## We create a function for extracting the exact hour 
def extract_hour(value):
    try:
        value = int(value)
        return value // 100
    except:
        return None

## Wrap it in a UDF so that Spark can actually run the function
extract_hour_udf = udf(extract_hour, IntegerType())

# Add or fix the DEP_HOUR column
## Then we use withColumn to actually give the function a column to work with 
df_clean = df_clean.withColumn("DEP_HOUR", extract_hour_udf(col("CRS_DEP_TIME")))

# Now group by hour and count delayed flights
## We first find the hours which have is_delayed as 1
## After this we grouo by the hour 
## We then count the delays for each hour 
## We rename the columns to be more readable
## We then order the results by hour in ascending order
hourly_delay_df = (
    df_clean.filter(col("IS_DELAYED") == 1)
    .groupBy("DEP_HOUR")
    .count()
    .withColumnRenamed("count", "DELAY_COUNT")
    .orderBy("DEP_HOUR")
)

hourly_delay_df.show(24, False)
print("--- Analysis 2 Result (Delays by Hour): ---")

# Save the result
output_path = "/data/results/hourly_delays"
hourly_delay_df.write.mode("overwrite").csv(output_path, header=True)
print(f"Saved analysis results to {output_path}")


# --- ANALYSIS 3: Delay Proportion by Distance Group ---
print("\n--- Running Analysis 3: Delay Proportion by Distance Group ---")

# Map/Transformation: Create distance categories
## We create a Column called distance group by taking the values and labeling them short , medium and long haul
df_distance = df_clean.withColumn(
    "DISTANCE_GROUP",
    when(col("DISTANCE") < 500, "Short Haul (< 500 mi)")
    .when((col("DISTANCE") >= 500) & (col("DISTANCE") < 2000), "Medium Haul (500-2000 mi)")
    .otherwise("Long Haul (> 2000 mi)")
)

# Reduce: Calculate the total flights and total delayed flights per group
## First we group by the distance group being small , large etc after this 
## WE aggregate for the group with 1 + 1 + 1 = 3 which means 3 delayed flight for the distance grp x
## and for distance_group we count the number of flights no filter on this just the total number of flight for x
## We dont just use the count for finding the number of delayed flight because then we will lose the ability to find the total number of flights 
distance_delay_df = df_distance.groupBy("DISTANCE_GROUP") \
                               .agg({"IS_DELAYED": "sum", "DISTANCE_GROUP": "count"}) \
                               .withColumnRenamed("sum(IS_DELAYED)", "TOTAL_DELAYED") \
                               .withColumnRenamed("count(DISTANCE_GROUP)", "TOTAL_FLIGHTS")

# Final calculation: Delay Rate = (TOTAL_DELAYED / TOTAL_FLIGHTS)
distance_delay_df = distance_delay_df.withColumn(
    "DELAY_RATE",
    (col("TOTAL_DELAYED") / col("TOTAL_FLIGHTS")) * 100
).orderBy(col("DELAY_RATE").desc())

distance_delay_df.show()
print("--- Analysis 3 Result (Delay Rate by Distance): ---")

# Save the result
distance_delay_df.write.mode("overwrite").csv("/data/results/distance_rate", header=True)
print("Saved analysis results to /data/results/distance_rate")



# --- ANALYSIS 4: Primary Delay Cause Breakdown ---
print("\n--- Running Analysis 4: Primary Delay Cause Breakdown ---")

# 1. Define the delay columns to analyze
## Takes the columns through which the flight mightve actually delayed 
delay_cols = [
    "carrier_delay", 
    "weather_delay", 
    "nas_delay", 
    "security_delay", 
    "late_aircraft_delay"
]

# 2. Impute Nulls with 0 for the delay cause columns
# This is crucial because a NULL means "not delayed by this factor", which should be 0.
df_delay_imputed = df_clean.fillna(0, subset=delay_cols)

# 3. Distributed Transformation (Map): Find the maximum delay cause for each flight
# 'greatest' compares values across the columns for each row to find the max delay value.
# We only care about flights that were actually delayed (IS_DELAYED == 1).
df_cause = df_delay_imputed.filter(col("IS_DELAYED") == 1).withColumn(
                               "MAX_DELAY_MIN", 
                               greatest(*delay_cols) # This finds the value of the largest delay cause
                           )

# 4. Define the Primary Delay Cause Category based on which column holds the MAX_DELAY_MIN value
# This is a complex transformation (Map) applied across all worker nodes.
## This just goes through each of the delay cols and then checks which has the same value with the max value 
df_cause = df_cause.withColumn(
    "PRIMARY_DELAY_CAUSE",
    when(col("MAX_DELAY_MIN") == col("carrier_delay"), "Carrier")
    .when(col("MAX_DELAY_MIN") == col("weather_delay"), "Weather")
    .when(col("MAX_DELAY_MIN") == col("nas_delay"), "NAS (System)") # NAS = National Airspace System
    .when(col("MAX_DELAY_MIN") == col("security_delay"), "Security")
    .when(col("MAX_DELAY_MIN") == col("late_aircraft_delay"), "Late Aircraft")
    .otherwise("Other/Unknown")
)

# 5. Distributed Aggregation (Reduce): Count the number of flights per cause
## This groups by the primary delay cause and then aggregates the total delay minutes for each cause 
## After this it alises the ugly name of the colum to total delay minutes
cause_breakdown_df = df_cause.groupBy("PRIMARY_DELAY_CAUSE") \
                             .agg(spark_sum("MAX_DELAY_MIN").alias("TOTAL_DELAY_MINUTES")) \
                             .orderBy(col("TOTAL_DELAY_MINUTES").desc())
                             
# Use .show() for immediate console feedback (avoids Pandas dependency)
## Truncate just snips the information and doesnt show the full infor
print("Primary Delay Cause Breakdown:")
cause_breakdown_df.show(truncate=False)

# 6. Save the result for Visualization 4
cause_breakdown_df.write.mode("overwrite").csv("/data/results/delay_cause_breakdown", header=True)
print("Saved analysis results to /data/results/delay_cause_breakdown")

# Stop the Spark session to release resources
spark.stop()
print("\n--- Spark Session Stopped. Analysis Complete! ---")



