# Flight Delay Prediction & Analysis

A distributed data processing and machine learning project for analyzing and predicting flight delays using **Apache Spark**, **PySpark MLlib**, **Docker**, and **Python**.

The project processes 2024 flight data, performs large-scale data cleaning and analysis with Spark, generates visual insights, and trains machine learning models to predict whether a flight will be delayed.

## Overview

Flight delays are influenced by factors such as departure time, carrier, route, distance, and operational conditions. This project uses Apache Spark to process flight data and extract meaningful patterns while also building machine learning models for delay prediction.

The project is divided into three major stages:

1. **Data Processing & Analysis** — Clean and transform the raw flight dataset using PySpark.
2. **Visualization** — Generate analytical results and visualizations from Spark output.
3. **Machine Learning** — Build and evaluate models for predicting flight delays.

## Architecture

```text
                 Flight Dataset
                       |
                       v
              +------------------+
              |   Apache Spark   |
              | Data Processing  |
              +------------------+
                       |
             +---------+---------+
             |                   |
             v                   v
       Data Analysis       Feature Engineering
             |                   |
             v                   v
       Spark Results        ML Pipeline
             |                   |
             v                   v
       Visualizations      ML Prediction
```

The Spark environment is containerized using Docker Compose, allowing the project to run using a Spark master and worker architecture.

## Technologies

* **Python**
* **PySpark**
* **Apache Spark**
* **Spark MLlib**
* **Docker & Docker Compose**
* **Pandas**
* **Matplotlib**
* **Git / GitHub**

## Project Structure

```text
FlightDelayProject/
│
├── code/
│   ├── create_visuals.py
│   ├── flight_analysis.py
│   ├── flight_prediction.py
│   └── sample_run.py
│
├── conf/
│   └── spark-defaults.conf
│
├── data/
│   └── flight_data_2024.csv
│
├── visuals/
│   ├── 1_Top_Delayed_Carriers.png
│   ├── 2_Hourly_Delay_Trend.png
│   ├── 3_Distance_Delay_Rate.png
│   └── 4_Primary_Delay_Causes.png
│
├── docker-compose.yml
└── FlightDelayProject.code-workspace
```

The raw dataset and generated Spark output are intentionally excluded from version control because of their size.

## Data Processing

The analysis pipeline performs several preprocessing operations on the flight dataset.

These include:

* Casting columns to appropriate data types
* Handling missing values
* Removing unnecessary columns
* Filtering invalid origin, destination, and carrier records
* Extracting departure-hour information
* Creating a binary `IS_DELAYED` target variable

A flight is classified as delayed when its departure delay exceeds **15 minutes**.

Conceptually:

```text
DEP_DELAY > 15
      |
      +----> IS_DELAYED = 1
      |
      +----> IS_DELAYED = 0
```

## Data Analysis

The Spark analysis pipeline produces several analytical datasets, including:

* Top delayed carriers
* Hourly delay trends
* Distance-based delay rates
* Primary causes of flight delays

These outputs are subsequently used to generate the visualizations stored in the `visuals/` directory.

## Machine Learning

The prediction pipeline uses Spark MLlib to build a machine learning workflow.

The pipeline includes:

```text
Raw Data
   |
   v
Data Cleaning
   |
   v
Categorical Encoding
(StringIndexer)
   |
   v
Feature Assembly
(VectorAssembler)
   |
   v
Feature Scaling
(StandardScaler)
   |
   v
Machine Learning Model
   |
   v
Prediction & Evaluation
```

Categorical features are transformed using `StringIndexer`, while numerical and categorical features are combined using `VectorAssembler`.

The dataset is split into training and testing sets using an **80/20 split** with a fixed random seed for reproducibility.

The project also contains a Random Forest model workflow and a Logistic Regression-based prediction workflow.

## Running the Project

### Prerequisites

Install the following:

* Docker
* Docker Compose
* Python 3
* Git

You do not need to install Apache Spark directly if you are using the provided Docker environment.

### 1. Clone the repository

```bash
git clone https://github.com/tanaybhomia/FlightDelay.git
cd FlightDelay
```

### 2. Add the dataset

The raw flight dataset is not stored in this repository because it is approximately **1 GB** in size.

Place the dataset at:

```text
data/flight_data_2024.csv
```

The expected filename is:

```text
flight_data_2024.csv
```

### 3. Start Spark

Start the Dockerized Spark environment:

```bash
docker compose up -d
```

Check that the containers are running:

```bash
docker compose ps
```

### 4. Run the analysis

Run the Spark analysis script from the appropriate Spark container/environment:

```bash
python code/flight_analysis.py
```

### 5. Generate visualizations

After the analysis results have been generated:

```bash
python code/create_visuals.py
```

The generated charts will be available in:

```text
visuals/
```

### 6. Run the prediction pipeline

```bash
python code/flight_prediction.py
```

## Visualizations

The project generates visualizations covering several aspects of flight delays.

### Top Delayed Carriers

Shows which airlines have the highest proportion or volume of delayed flights.

### Hourly Delay Trend

Examines how flight delays vary throughout the day and helps identify periods with higher delay rates.

### Distance vs Delay Rate

Analyzes the relationship between flight distance and the probability of delay.

### Primary Delay Causes

Breaks down the major contributors to flight delays and provides insight into the operational factors affecting flight schedules.

## Dataset

The project uses 2024 flight data containing flight-level information such as:

* Flight date
* Airline
* Origin and destination
* Scheduled departure time
* Actual departure delay
* Arrival delay
* Flight distance
* Delay causes
* Other operational attributes

The dataset is **not included in this GitHub repository** because its size exceeds GitHub's standard file-size limit.

Place the downloaded dataset at:

```text
data/flight_data_2024.csv
```

## Why Spark?

The project uses Apache Spark rather than processing the entire dataset with traditional Pandas workflows.

This provides experience with:

* Distributed data processing
* Spark DataFrames
* Lazy evaluation
* Transformations and actions
* Spark SQL-style operations
* Distributed machine learning
* Containerized Spark clusters

The Docker Compose setup also makes it possible to reproduce the Spark environment without manually configuring a local Spark cluster.

## Key Learning Outcomes

This project demonstrates practical experience with:

* Large-scale data processing
* PySpark DataFrame operations
* Data cleaning and feature engineering
* Distributed computing
* Dockerized Spark environments
* Exploratory data analysis
* Data visualization
* Machine learning pipelines
* Classification models
* Model evaluation
* Reproducible development environments

## Future Improvements

Potential improvements include:

* Hyperparameter tuning for the ML models
* Cross-validation and more extensive model comparison
* Feature importance analysis
* Model performance optimization
* Automated data ingestion
* A REST API for flight-delay predictions
* A web dashboard for interactive analysis
* CI/CD automation for the Spark application
* Deployment to a cloud-based Spark environment

## Author

**Tanay Bhomia**

This project was developed as a practical project combining data engineering, distributed processing, visualization, and machine learning.

