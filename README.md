# Data Lake with Pyspark

There is a startup called Sparkify who wants to analyze the data that they have been collecting. So, the aim of the project is to perform ETL of the data(JSON to a Readshift Database) that Sparkify has collected for easy analysis.The dataset used here is an subset of [Million songs Dataset](http://millionsongdataset.com/).<br>

Song Dataset consists of meta data of each song. File structure is as follows,<br>

```
song_data/A/B/C/TRABCEI128F424C983.json
song_data/A/A/B/TRAABJL12903CDCF1A.json
```
Below is the structure of song file TRAABJL12903CDCF1A.json,

```
{"num_songs": 1, "artist_id": "ARJIE2Y1187B994AB7", "artist_latitude": null, "artist_longitude": null, "artist_location": "", "artist_name": "Line Renaud", "song_id": "SOUPIRU12A6D4FA1E1", "title": "Der Kleine Dompfaff", "duration": 152.92036, "year": 0}
```
Log Dataset simulate the activity of an music streaming app, generated from [this](https://github.com/Interana/eventsim) event stimulator, and has the following structure.

```
log_data/2018/11/2018-11-12-events.json
log_data/2018/11/2018-11-13-events.json
```
And, a log file(2018-11-12-events.json) looks as follows,

![alt txt](log-data.png)

## Database Design

Fact Table - `'songplays'` - records in log data associated with song plays. <br>
Columns - songplay_id, start_time, user_id, level, song_id, artist_id, session_id, location, user_agent
<br><br>
Dimension Tables -
<br>
`users` - user_id, first_name, last_name, gender, level
<br>
`songs` - song_id, title, artist_id, year, duration
<br>
`artists` - artist_id, name, location, latitude, longitude
<br>
`time`- timestamps of records in songplays - start_time, hour, day, week, month, year, weekday
<br><br>

![alt txt](DB%20Designs.jpg)

## Implementation Steps

<ol>
 <li>Connect with Pyspark using the configuration in `dl.cfg`.</li>
<li>Write SQL 'CREATE', 'DROP' and 'INSERT' statements in `sql_queries.py` and run <br>`create_tables.py` to run it to create the tables.</li>
<li>Run `etl.py` to load the data(json files) from S3 to Redshift staging tables, then select the required columns and insert into fact and dimentional tables.</li>
</ol>

