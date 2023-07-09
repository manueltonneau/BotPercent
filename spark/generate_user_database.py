srun --cpus-per-task=1 --time=4:00:00 --mem=50000 --pty /bin/bash
module load pyspark/2.4.7
pyspark --conf spark.serializer=org.apache.spark.serializer.KryoSerializer \
       --conf spark.speculation=false \
       --conf spark.executorEnv.LANG=en_US.UTF-8 \
       --conf spark.yarn.appMasterEnv.LANG=en_US.UTF-8 \
       --conf spark.sql.files.ignoreCorruptFiles=true \
       --driver-cores 20 \
       --driver-memory 150g \
       --num-executors 50 \
       --executor-cores 5 \
       --executor-memory 50g \
       --deploy-mode client \
       --conf spark.memory.offHeap.enabled=true \
       --conf spark.memory.offHeap.size=50g \
       --conf spark.sql.autoBroadcastJoinThreshold=-1

from pyspark.sql.functions import col, udf, lit, from_unixtime
from pyspark.sql.functions import col, current_date, datediff, from_unixtime
from pyspark.sql.types import FloatType
import pyspark.sql.functions as F
from pyspark.sql.functions import avg, stddev, sqrt, count

df = spark.read.parquet('/scratch/spf248/twitter_social_cohesion/data/data_collection/lookup_users/output/25052023')
location_df = spark.read.parquet('/scratch/spf248/twitter_social_cohesion/data/user_location/snowball/29032023')
location_df = location_df.withColumnRenamed("user_location", "location")
merged_df = df.join(location_df, on="location", how="inner")

merged_df = merged_df.select(['statuses_count',
                              'followers_count',
                              'friends_count',
                              'listed_count',
                              'profile_image_url',
                              'verified',
                              'id',
                              'screen_name',
                              'name',
                              'description',
                              'protected',
                              'location',
                              'entities',
                              'url',
                              'created_at',
                              ])
merged_df.write.format("parquet").mode("overwrite").save('/scratch/mt4493/bot_detection/data/user_profiles')

## 95K
df = spark.read.parquet('/scratch/spf248/twitter_social_cohesion/data/ethnic_hate/ethnic_hate_users_selection/ethnic_hate_users/iter=24/hate_cutoff=0.99/target_cutoff=0.5/production_cutoff=2/consumption_cutoff=2/date_cutoff=01012022/19062023')
profiles_df = spark.read.parquet('/scratch/spf248/twitter_social_cohesion/data/data_collection/lookup_users/output/25052023')
# Rename the "id_str" column to "user_id" in profiles_df
profiles_df = profiles_df.withColumnRenamed("id_str", "user_id")

# Merge df and profiles_df on "user_id" (dropping unmatched rows)
merged_df = df.join(profiles_df, on="user_id", how="inner")

merged_df = merged_df.select(['statuses_count',
                              'followers_count',
                              'friends_count',
                              'listed_count',
                              'profile_image_url',
                              'verified',
                              'id',
                              'screen_name',
                              'name',
                              'description',
                              'protected',
                              'location',
                              'entities',
                              'url',
                              'created_at',
                              ])

merged_df.write.format("parquet").mode("overwrite").save('/scratch/mt4493/bot_detection/data/user_profiles/ethnic_hate_users')

# just another day users
from pyspark.sql.functions import col, udf, lit, from_unixtime

df = spark.read.parquet('/scratch/mt4493/just_another_day/data/processed/users/dedup')

# Define a UDF (User Defined Function) to extract values from the list
extract_values = lambda metrics, idx: metrics[idx] if metrics is not None and len(metrics) > idx else None
extract_values_udf = udf(extract_values)

# Create four new columns by applying the UDF to the "public_metrics" column
df = df.withColumn("followers_count", extract_values_udf(col("public_metrics"), lit(0))) \
                   .withColumn("friends_count", extract_values_udf(col("public_metrics"), lit(1))) \
                   .withColumn("listed_count", extract_values_udf(col("public_metrics"), lit(2))) \
                   .withColumn("statuses_count", extract_values_udf(col("public_metrics"), lit(3)))

# df = df.withColumn("profile_image_url", lit(None))
# df = df.withColumn("entities", lit(None))
df = df.withColumnRenamed("username", "screen_name")
df = df.withColumn("protected", lit(False))

df = df.select(['statuses_count',
                              'followers_count',
                              'friends_count',
                              'listed_count',
                              # 'profile_image_url',
                              'verified',
                              'id',
                              'screen_name',
                              'name',
                              'description',
                              'protected',
                              'location',
                              # 'entities',
                              'url',
                              'created_at',
                              ])
# df = df.na.fill("")
df.write.format("parquet").mode("overwrite").save('/scratch/mt4493/bot_detection/data/user_profiles/just_another_day')
