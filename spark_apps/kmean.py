from pyspark.sql import SparkSession
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import col
import pandas as pd

spark = SparkSession.builder.appName('Kmean').getOrCreate() 

# Load the customer data into a DataFrame
df = spark.read.csv("/opt/spark/data/kmean_dataset.csv")
df.show()
# Convert columns to float
df = df.select(*(col(c).cast("float").alias(c) for c in df.columns))

assembler = VectorAssembler(
    inputCols=["_c0", "_c1"],
    outputCol="features")

df = assembler.transform(df)
df = df.drop("_c0")
df = df.drop("_c1")

# Train the K-means clustering model
kmeans = KMeans(k=3)
model = kmeans.fit(df)

# Make predictions
predictions = model.transform(df)
predictions.show(700, truncate=False)

predictions.toPandas().to_csv("/opt/spark/data/result_kmean.csv", mode='w+')
