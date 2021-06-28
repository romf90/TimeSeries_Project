import tensorflow as tf
import numpy as np
from sklearn import preprocessing


def split_sequence(sequences, to_predict, window_size):
    X, Y = [], []
    for i in range(sequences.shape[0]):
        if (i + window_size) >= sequences.shape[0]:
            break  # Divide sequence between data (input) and labels (output)
        seq_X, seq_Y = sequences[i: i + window_size], to_predict[i + window_size]
        X.append(seq_X)
        Y.append(seq_Y)
    return np.array(X), np.array(Y)


def scale_and_split(time, series_temps, series_GHG):
    series_temps = series_temps.reshape(series_temps.shape[0], 1)
    series_GHG = series_GHG.reshape(series_GHG.shape[0], 1)
    series = np.hstack((series_temps, series_GHG))
    split_time = 2630
    time_train = time[:split_time]
    x_train = series[:split_time]
    time_valid = time[split_time:]
    x_valid = series[split_time:]

    size_window = 12
    size_batch = 120

    scaler_temps = preprocessing.MinMaxScaler(feature_range=(0, 1))
    scaler_GHG = preprocessing.MinMaxScaler(feature_range=(0, 1))

    temps_scaled = scaler_temps.fit_transform(series[:, 0].reshape(-1, 1))
    GHG_scaled = scaler_GHG.fit_transform(series[:, 1].reshape(-1, 1))
    scaled_data_set = np.hstack((temps_scaled, GHG_scaled))
    return scaled_data_set, split_time, size_window, size_batch, scaler_temps, time_valid, x_valid, x_train, time_train


def train(time, series_temps, series_GHG):
    scaled_data_set, split_time, size_window, size_batch, scaler_temps, time_valid, x_valid, x_train, time_train = scale_and_split(time, series_temps, series_GHG)
    tf.keras.backend.clear_session()
    tf.random.set_seed(51)
    np.random.seed(51)

    X_train, Y_train = split_sequence(scaled_data_set[:split_time], scaled_data_set[:split_time],
                                      window_size=size_window)
    X_valid, Y_valid = split_sequence(scaled_data_set[split_time:], scaled_data_set[split_time:],
                                      window_size=size_window)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv1D(filters=256, kernel_size=16, strides=1, padding="causal", activation="relu",
                               input_shape=[None, 2]),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=False)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32),
        tf.keras.layers.Dense(16),
        tf.keras.layers.Dense(2),
    ])

    optimizer = tf.keras.optimizers.SGD(lr=1e-2, momentum=0.9)
    model.compile(loss=tf.keras.losses.Huber(),
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=["mae"])
    history = model.fit(X_train, Y_train, batch_size=size_batch, epochs=150)

    return model, X_valid, scaler_temps, size_window, time_valid, x_valid


def train_lr(time, series_temps, series_GHG):
    scaled_data_set, split_time, size_window, size_batch, scaler_temps, time_valid, x_valid, x_train, time_train = scale_and_split(time, series_temps, series_GHG)
    tf.keras.backend.clear_session()
    tf.random.set_seed(51)
    np.random.seed(51)

    X_train, Y_train = split_sequence(scaled_data_set[:split_time], scaled_data_set[:split_time],
                                      window_size=size_window)
    X_valid, Y_valid = split_sequence(scaled_data_set[split_time:], scaled_data_set[split_time:],
                                      window_size=size_window)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv1D(filters=256, kernel_size=16, strides=1, padding="causal", activation="relu",
                               input_shape=[None, 2]),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=False)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32),
        tf.keras.layers.Dense(16),
        tf.keras.layers.Dense(2),
    ])

    lr_schedule = tf.keras.callbacks.LearningRateScheduler(
        lambda epoch: 1e-8 * 10 ** (epoch / 20))
    optimizer = tf.keras.optimizers.SGD(lr=1e-8, momentum=0.9)
    model.compile(loss=tf.keras.losses.Huber(),
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=["mae"])
    history = model.fit(X_train, Y_train, batch_size=size_batch, epochs=200, callbacks=[lr_schedule])
    return history







