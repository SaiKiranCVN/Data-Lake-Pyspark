import configparser
from datetime import datetime
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col, to_timestamp
from pyspark.sql.functions import year, month, dayofmonth, hour, weekofyear, date_format, dayofweek

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
    song_data =  input_data + 'song_data/A/A/A/TRAAAPK128E0786D96.json'
    
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
        Fld("song_id", Str()),
    ])
    
    # read song data file
    df = spark.read.json(song_data,schema=songSchema,columnNameOfCorruptRecord='corrupt_record').dropDuplicates()
    print("Read completed")

    # created song view to write SQL Queries
    df.createOrReplaceTempView("songs_table")
    
    # extract columns to create songs table
    songs_table = ["song_id", "title", "artist_id","year", "duration"]
    
    # Selecting rows that doesn't have duplicate song_id, in increasing order
    songs_table = df.select(songs_table).dropDuplicates()
    
    # write songs table to parquet files partitioned by year and artist
    # partitioning by year and artist_id, and saving in 'songs' folder
    songs_table.write.partitionBy("year", "artist_id").mode("overwrite").parquet(output_data + 'songs')
    print("Songs Write Completed")
    
    ## Artist
    
    # Similarly, get the fields list first. Using 'as' for alias in this list directly, as we can use selectExpr
    artists_fields = ["artist_id", "artist_name as name", "artist_location as location", "artist_latitude as latitude", "artist_longitude as longitude"]

    # extract columns to create artists table
    artists_table = df.selectExpr(artists_fields).dropDuplicates()
    
    # write artists table to parquet files, in 'artists' folder
    artists_table.write.mode("overwrite").parquet(output_data + 'artists')
    print("Artists Write Completed")
    print("----------process_song_data Complete-----------")

def process_log_data(spark, input_data, output_data):
    """
    Processes all log data JSON files in the input folder and stores them in parquet format in the output folder.
    Takes spark context, input_data path and output_data path as parameters.
    """
    # get filepath to log data file, data is log_data/year/month/log.json
    log_data = input_data + 'log_data/2018/11/2018-11-01-events.json'

    # read log data file
    log_df = spark.read.json(log_data)
    print("Read completed")
    
    
    # filter by actions for song plays
    log_df = log_df.filter(log_df.page == 'NextSong')
    
    # created log view to write SQL Queries
    log_df.createOrReplaceTempView("logs_table")
    
    #Listing the field for user table
    users_fields = ["userId as user_id", "firstName as first_name", "lastName as last_name", "gender", "level"]

    # extract columns for users table    
    users_table = log_df.selectExpr(users_fields).dropDuplicates()
    
    
    # write users table to parquet files, in 'users' folder
    users_table.write.mode("overwrite").parquet(output_data + 'users')
    print("Users table write complete")

    # create timestamp column from original timestamp column
    get_timestamp = udf(lambda x: x / 1000, TimestampType())
    log_df = log_df.withColumn("timestamp", get_timestamp(log_df.ts))
    
    # create datetime column from original timestamp column
    get_datetime = udf(lambda x: datetime.fromtimestamp(x), TimestampType())
    log_df = log_df.withColumn("start_time", get_datetime(log_df.timestamp))
    
    #Create columns
    log_df = log_df.withColumn("hour", hour("start_time")) \
        .withColumn("day", dayofmonth("start_time")) \
        .withColumn("week", weekofyear("start_time")) \
        .withColumn("month", month("start_time")) \
        .withColumn("year", year("start_time")) \
        .withColumn("weekday", dayofweek("start_time"))
    
    # extract columns to create time table
    time_table = log_df.select("start_time", "hour", "day", "week", "month", "year", "weekday")
    
    
    # write time table to parquet files partitioned by year and month, in "time" folder
    time_table.write.partitionBy("year", "month").mode("overwrite").parquet(output_data + "time")
    print("Time_table write complete")

    ## For Songplays, we need both artist and song tables
    # extract columns from joined song and log datasets to create songplays table, repartitioning with year and month
    # read in song data to use for songplays table, reading from the file we saved
    
    
    songplays_table = spark.sql("""select monotonically_increasing_id() as songplay_id,
    to_timestamp(l.ts/1000) as start_time, 
    month(to_timestamp(l.ts/1000)) as month, 
    year(to_timestamp(l.ts/1000)) as year, 
    l.userId as user_id, l.level as level, 
    s.song_id as song_id, s.artist_id as artist_id, 
    l.sessionId as session_id, l.location as location, 
    l.userAgent as user_agent FROM
    logs_table l JOIN songs_table s on l.artist = s.artist_name and l.song = s.title
    """)

    # write songplays table to parquet files partitioned by year and month, in "songplays" folder
    songplays_table.write.partitionBy("year", "month").mode("overwrite").parquet(output_data, 'songplays')
    print("Songplays write complete")


def main():
    spark = create_spark_session()
    input_data = "s3a://udacity-dend/"
    output_data = "s3a://udacitydataengineernd/"
    
    process_song_data(spark, input_data, output_data)    
    process_log_data(spark, input_data, output_data)


if __name__ == "__main__":
    main()
