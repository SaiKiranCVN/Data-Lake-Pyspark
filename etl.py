import configparser
from datetime import datetime
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.functions import year, month, dayofmonth, hour, weekofyear, date_format

#Importing types to fix schema
from pyspark.sql.types import StructType as R, StructField as Fld, DoubleType as Dbl, StringType as Str, IntegerType as Int, DateType as Date, TimestampType
from pyspark.sql.functions import monotonically_increasing_id

config = configparser.ConfigParser()
config.read('dl.cfg')

os.environ['AWS_ACCESS_KEY_ID']= config.get('AWS', 'AWS_ACCESS_KEY_ID')
os.environ['AWS_SECRET_ACCESS_KEY']=config.get('AWS', 'AWS_SECRET_ACCESS_KEY')


def create_spark_session():
    """
    Create or obtain a spark session and return it.
    """
    spark = SparkSession \
        .builder \
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:2.7.0") \
        .getOrCreate()
    return spark


def process_song_data(spark, input_data, output_data):
    """
    Process the song data dataset and creates the song table and artist table.
    Takes spark context, input_data path and output_data path as parameters.
    """
    
    # get filepath to song data file, data is song_data/A/A/A/song.json
    song_data =  input_data + 'song_data/A/A/A/*.json'
    
    # Assigning right datatype
    songSchema = R([
        Fld("artist_id",Str()),
        Fld("artist_latitude",Dbl()),
        Fld("artist_location",Str()),
        Fld("artist_longitude",Dbl()),
        Fld("artist_name",Str()),
        Fld("duration",Dbl()),
        Fld("num_songs",Int()),
        Fld("title",Str()),
        Fld("year",Int()),
    ])
    
    # read song data file
    df = spark.read.json(song_data,schema=songSchema,columnNameOfCorruptRecord='corrupt_record').dropDuplicates()
    print("Read completed")

    # extract columns to create songs table
    songs_table = ["title", "artist_id","year", "duration"]
    
    # Selecting rows that doesn't have duplicate song_id, in increasing order
    # Link - https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.sql.functions.monotonically_increasing_id.html
    songs_table = df.select(songs_table).dropDuplicates().withColumn("song_id", monotonically_increasing_id())
    
    # write songs table to parquet files partitioned by year and artist
    # partitioning by year and artist_id, and saving in 'songs' folder
    songs_table.write.mode("overwrite").partitionBy("year", "artist_id").parquet(output_data + 'songs')
    print("Songs Write Completed")
    
    ## Artist
    
    # Similarly, get the fields list first. Using 'as' for alias in this list directly, as we can use selectExpr
    artists_fields = ["artist_id", "artist_name as name", "artist_location as location", "artist_latitude as latitude", "artist_longitude as longitude"]

    # extract columns to create artists table
    artists_table = df.selectExpr(artists_fields).dropDuplicates()
    
    # write artists table to parquet files, in 'artists' folder
    artists_table.write.mode("overwrite").parquet(output_data + 'artists')
    print("Artists Write Completed")


def process_log_data(spark, input_data, output_data):
    """
    Processes all log data JSON files in the input folder and stores them in parquet format in the output folder.
    Takes spark context, input_data path and output_data path as parameters.
    """
    # get filepath to log data file, data is log_data/year/month/log.json
    log_data = input_data + 'log_data/2018/11/*.json'

    # read log data file
    df = spark.read.json(log_data)
    print("Read completed")
    
    
    # filter by actions for song plays
    df = df.filter(df.page == 'NextSong')
    
    #Listing the field for user table
    users_fields = ["userId as user_id", "firstName as first_name", "lastName as last_name", "gender", "level"]

    # extract columns for users table    
    users_table = df.selectExpr(users_fields).dropDuplicates()
    
    # write users table to parquet files, in 'users' folder
    users_table.write.mode("overwrite").parquet(output_data + 'users')
    print("Users table write complete")

    # create timestamp column from original timestamp column
    get_timestamp = udf(lambda x: x / 1000, TimestampType())
    df = df.withColumn("timestamp", get_timestamp(df.ts))
    
    # create datetime column from original timestamp column
    get_datetime = udf(lambda x: datetime.fromtimestamp(x), TimestampType())
    df = df.withColumn("start_time", get_datetime(df.timestamp))
    
    #Create columns
    df = df.withColumn("hour", hour("start_time")) \
        .withColumn("day", dayofmonth("start_time")) \
        .withColumn("week", weekofyear("start_time")) \
        .withColumn("month", month("start_time")) \
        .withColumn("year", year("start_time")) \
        .withColumn("weekday", dayofweek("start_time"))
    
    # extract columns to create time table
    time_table = df.select("start_time", "hour", "day", "week", "month", "year", "weekday")
    
    # write time table to parquet files partitioned by year and month, in "time" folder
    time_table.write.mode("overwrite").partitionBy("year", "month").parquet(output_data + "time")
    print("Time_table write complete")

    ## For Songplays, we need both artist and song tables
    # read in song data to use for songplays table, reading from the file we saved
    song_df = spark.read.parquet(os.path.join(output_data, "songs/*/*/*"))
    
    # df is logs, select the same songs
    songs_logs = df.join(songs_df, (df.song == df.title))
    #Read artists table
    artists_df = spark.read.parquet(os.path.join(output_data, "artists"))
    #Select the same artist
    artists_songs_logs = songs_logs.join(artists_df, (songs_logs.artist == artists_df.name))
    #Left join with artists and time_table
    songplays = artists_songs_logs.join(
        time_table,
        artists_songs_logs.ts == time_table.ts, 'left'
    ).drop(artists_songs_logs.year)

    # extract columns from joined song and log datasets to create songplays table, repartitioning with year and month 
    songplays_table =  songplays.select(
        col('start_time'),
        col('userId').alias('user_id'),
        col('level'),
        col('song_id'),
        col('artist_id'),
        col('sessionId').alias('session_id'),
        col('location'),
        col('userAgent').alias('user_agent'),
        col('year'),
        col('month'),
    ).repartition("year", "month")

    # write songplays table to parquet files partitioned by year and month, in "songplays" folder
    songplays_table.write.mode("overwrite").partitionBy("year", "month").parquet(output_data, 'songplays')
    print("Songplays write complete")


def main():
    spark = create_spark_session()
    input_data = "s3a://udacity-dend/"
    output_data = "s3a://datalakeoutputnd/"
    
    process_song_data(spark, input_data, output_data)    
    process_log_data(spark, input_data, output_data)


if __name__ == "__main__":
    main()
