import matplotlib.pyplot as plt
import info


def plot_series_Temp(time, series, format="-", start=0, end=None):
    plt.plot(time[start:end], series[start:end], format)
    plt.xlabel("Years")
    plt.ylabel("Average temperature anomaly ")
    plt.grid(True)


def plot_series_GHG(time, series, format="-", start=0, end=None):
    plt.plot(time[start:end], series[start:end], format)
    plt.xlabel("Year")
    plt.ylabel("GHG emissions each year")
    plt.grid(True)


def ploting_graphes(time, series_temps, series_GHG):
    plt.figure(figsize=(10, 6))
    plot_series_Temp(time, series_temps)

    plt.figure(figsize=(10, 6))
    plot_series_GHG(time, series_GHG)
    plt.show()


def plot_lr(history):
    plt.semilogx(history.history["lr"], history.history["loss"])
    plt.axis([1e-8, 1, 0, 0.1])
    plt.show()
    info.info_conclusion()
    