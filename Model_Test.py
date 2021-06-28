import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
import Model_Train
import info
import visual


def Test_After_Train(model, X_valid, scaler_temps, size_window, time_valid, x_valid):
    forecast = []
    forecast = model.predict(X_valid)
    results_temps = np.roll(scaler_temps.inverse_transform(forecast[:, 0].reshape(-1, 1)), size_window - 1)

    plt.figure(figsize=(10, 6))
    visual.plot_series_Temp(time_valid[:-size_window], x_valid[:, 0][:-size_window])
    visual.plot_series_Temp(time_valid[:-size_window], results_temps)
    plt.legend('RP')
    info.info_error(results_temps, x_valid, size_window)
    plt.show()


def test_model_NoTrain(time, series_temps, series_GHG):
    scaled_data_set, split_time, size_window, size_batch, scaler_temps, time_valid, x_valid, x_train, time_train = Model_Train.scale_and_split(time, series_temps, series_GHG)
    X_valid, Y_valid = Model_Train.split_sequence(scaled_data_set[split_time:], scaled_data_set[split_time:],
                                      window_size=size_window)
    model = keras.models.load_model('Project_Model')
    Test_After_Train(model, X_valid, scaler_temps, size_window, time_valid, x_valid)

