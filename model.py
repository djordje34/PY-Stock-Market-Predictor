from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from preprocess import load_data, create_sequences
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from sklearn.preprocessing import MinMaxScaler
import datetime as dt

def preprocess_data(data, scaler, cols):
    temp = data.copy(deep=True)
    temp[cols] = scaler.fit_transform(temp[cols])
    return temp[cols].values

def build_lstm_model(input_shape=(60, 1)):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_model(model, X_train, y_train, epochs=100, batch_size=32, validation_split=0.15):
    checkpointer = ModelCheckpoint(
        filepath='best_weights.h5',
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        mode='min'
    )

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=1e-5
    )

    tensorboard = TensorBoard(
        log_dir='logs',
        histogram_freq=1,
        write_graph=True,
        write_images=True
    )

    model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=[checkpointer, early_stopping, reduce_lr, tensorboard]
    )

def prepare_test_data(train_data, test_data, scaler, seq_length=60):
    actual_prices = test_data['Closing_Price'].values
    total_dataset = pd.concat((train_data['Closing_Price'], test_data['Closing_Price']), axis=0)
    model_inputs = total_dataset[len(total_dataset) - len(test_data) - seq_length:].values
    model_inputs = model_inputs.reshape(-1, 1)
    model_inputs = scaler.transform(model_inputs)
    X_test, _ = create_sequences(model_inputs, seq_length)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    return actual_prices, X_test

def plot_results(actual_prices, predicted_prices, title):
    print(np.shape(actual_prices))
    plt.plot(actual_prices, color='black', label="Права вредност")
    plt.plot(predicted_prices, color='green', label="Предложена вредност")
    plt.title(title)
    plt.xlabel("Време")
    plt.ylabel("Peugeot цена акција")
    plt.legend()
    plt.savefig("gen/rezultat.png")
    plt.close()

START_DATE = dt.datetime(2015, 1, 1)
END_DATE = dt.datetime(2019, 12, 30)
START_DATE_TEST = END_DATE

scaler = MinMaxScaler(feature_range=(0, 1))

cols = ['Closing_Price']

train_data = load_data(START_DATE, END_DATE)
temp_train = preprocess_data(train_data, scaler, cols)

X_train, y_train = create_sequences(temp_train, 60)
print(np.shape(X_train), np.shape(y_train))

model = build_lstm_model(input_shape=(60, 1))
train_model(model, X_train, y_train)

test_data = load_data(START_DATE_TEST, dt.datetime(2020, 12, 30))

actual_prices, X_test = prepare_test_data(train_data, test_data, scaler)
print(np.shape(X_test[0]),np.shape(X_test),len(test_data))
predicted_prices = model.predict(X_test)
predicted_prices = scaler.inverse_transform(predicted_prices)

plot_results(actual_prices, predicted_prices, "Peugeot цена акција")