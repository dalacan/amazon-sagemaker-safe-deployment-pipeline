"""Feature engineers the abalone dataset."""
import argparse
import logging
import os
import pathlib
import requests
import tempfile

import boto3
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


parse_dates= ['lpep_dropoff_datetime', 'lpep_pickup_datetime']

if __name__ == "__main__":
    logger.debug("Starting preprocessing.")
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-data", type=str, required=True)
    args = parser.parse_args()

    base_dir = "/opt/ml/processing"
    pathlib.Path(f"{base_dir}/data").mkdir(parents=True, exist_ok=True)
    input_data = args.input_data
    bucket = input_data.split("/")[2]
    key = "/".join(input_data.split("/")[3:])

    logger.info("Downloading data from bucket: %s, key: %s", bucket, key)
    fn = f"{base_dir}/data/green_tripdata_2018-02.csv"
    s3 = boto3.resource("s3")
    s3.Bucket(bucket).download_file(key, fn)

    logger.debug("Reading downloaded data.")
    df = pd.read_csv(
        fn,
        parse_dates=parse_dates
    )
    
    os.unlink(fn)

    #
    # Data manipulation
    #
    # Instead of the raw date and time features for pick-up and drop-off, 
    # let's use these features to calculate the total time of the trip in minutes, which will be easier to work with for our model.

    df['duration_minutes'] = (df['lpep_dropoff_datetime'] - df['lpep_pickup_datetime']).dt.seconds/60
    
    # Remove unnecessary columns
    cols = ['total_amount', 'duration_minutes', 'passenger_count', 'trip_distance']
    data_df = df[cols]
    
    # Remove outliers
    data_df = data_df[(data_df.total_amount > 0) & (data_df.total_amount < 200) & 
                  (data_df.duration_minutes > 0) & (data_df.duration_minutes < 120) & 
                  (data_df.trip_distance > 0) & (data_df.trip_distance < 121) & 
                  (data_df.passenger_count > 0)].dropna()

    logger.info("Splitting %d rows of data into train, validation, test datasets.", len(data_df))
    np.random.shuffle(data_df)
    train, validation, test = np.split(data_df, [int(0.7 * len(data_df)), int(0.85 * len(data_df))])

    logger.info("Writing out datasets to %s.", base_dir)
    pd.DataFrame(train).to_csv(f"{base_dir}/train/train.csv", header=False, index=False)
    pd.DataFrame(validation).to_csv(
        f"{base_dir}/validation/validation.csv", header=False, index=False
    )
    pd.DataFrame(test).to_csv(f"{base_dir}/test/test.csv", header=False, index=False)
