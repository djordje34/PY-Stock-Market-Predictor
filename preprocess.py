import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def load_data(start,end):
    df = pd.read_csv("preprocessed_CAC40.csv")
    df = df.interpolate(method='linear')
    df['Date'] = pd.to_datetime(df['Date'])
    df.drop(df.filter(regex="Unname"),axis=1, inplace=True)
    df = df[df["Name"] == "Peugeot"]
    
    df = df.loc[(df['Date'] > start) & (df['Date'] < end), :]
    
    df['Volume'] = pd.to_numeric(df['Volume'].str.replace(',', ''), errors='raise')
    df['Volume'] = df['Volume'].ffill()
    df = df.sort_values('Date')
    
    return df

    
def create_sequences(arr, seq_len):
    X_train = []
    y_train = []

    for x in range(seq_len, len(arr)):
        X_train.append(arr[x - seq_len : x, -1])
        y_train.append(arr[x, -1]) 

    X_train, y_train = np.array(X_train), np.array(y_train)

    return X_train, y_train
