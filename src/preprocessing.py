from connect import Connect
import pandas as pd
import numpy as np

client = Connect.get_connection()
db = client.news

def read_mongo(query={}, no_id=True):
    """ Read from Mongo and Store into DataFrame """

    cursor = db.fake_real_hope.find(query)
    df =  pd.DataFrame(list(cursor))
    if no_id:
        del df['_id']
    print("Data loaded from mongodb. The length is {}".format(len(df)))
    return df

def clean_df(df):
    df.loc[df.subject == 'politicsNews', "subject"] = "politics"
    df.subject = df.subject.str.lower()
    return df

def create_datetime(df):
    df['datetime'] = pd.to_datetime(df['date'])
    return df

def drop_nan(df):
    df.loc[df['text'].str.isspace() == True, 'text'] = np.NaN
    empty_values = len(df[df.isna().any(axis=1)])
    if empty_values:
        df = df.dropna()
    empty_values_after = len(df[df.isna().any(axis=1)])
    if not empty_values_after:
        print("Successfully deleted {} empty values".format(empty_values))
    return df

def get_df():
    # tested with csv
    # df = pd.read_csv('../data/fake_real_news.csv')

    df = read_mongo()
    df = clean_df(df)
    df = create_datetime(df)
    df = drop_nan(df)
    return df

df = get_df()