from pyspark.sql import SparkSession

spark = (
    SparkSession.builder.appName("RecSys Shell")
    .config("spark.driver.memory", "32g")
    .config("spark.sql.session.timeZone", "Asia/Seoul")
    .config("spark.driver.extraJavaOptions", "-Duser.timezone=Asia/Seoul")
    .config("spark.executor.extraJavaOptions", "-Duser.timezone=Asia/Seoul")
    .getOrCreate()
)
