import csv
import tkinter
from functools import partial
from tkinter import *
import numpy as np
from PIL import Image
from PIL import ImageTk
import Model_Test
import Model_Train
import info
import visual


def Load_Data():
    time_step = []   # list containing the time points
    temps_series = []  # list containing the temp points
    GHG_series = []  # list containing the green house gases points

    with open('Finale_DataSet.csv') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader)
        step = 1880
        for row in reader:
            temps_series.append(float(row[1]))
            GHG_series.append(float(row[2]))
            time_step.append(step)
            step = step + 1 / 24

    series_temps = np.array(temps_series)
    series_GHG = np.array(GHG_series)
    time = np.array(time_step)

    return time, series_temps, series_GHG


def train_model():
    train_window = Tk()
    train_window.title('Training Model')
    train_window.geometry('300x300')
    lbl = Label(train_window, text="The Model Has Been Trained!", foreground="white", background="black").pack()
    model, X_valid, scaler_temps, size_window, time_valid, x_valid = Model_Train.train(time, series_temps, series_GHG)
    btn_test_after_train = tkinter.Button(train_window, text='test_model',
                                          command=partial(Model_Test.Test_After_Train, model, X_valid, scaler_temps, size_window,
                                                          time_valid, x_valid), height=10, width=20).pack(pady=10)
    train_window.mainloop()


def Find_lr():
    info.info_lr_before()
    history = Model_Train.train_lr(time, series_temps, series_GHG)
    info.info_lr_after()
    LR_window = Tk()
    LR_window.title('Best Learning Rate')
    LR_window.geometry('300x300')
    lr_lbl = Label(LR_window, text="Hello, User. Please choose an option:", foreground="white",
                     background="black").pack(pady=10)

    btn_lr = Button(LR_window, text="check the correlation between the lr and the loss", command=partial(visual.plot_lr, history)).pack(pady=10)

    LR_window.mainloop()


def menu():
    main_window = Tk()
    main_window.title('Rom Project Menu')
    main_window.geometry('300x600')
    width = 200  # representing the width of the photo
    height = 100  # representing the height of the photo

    main_lbl = Label(main_window, text="Hello, User. Please choose an option:", foreground="white",
                     background="black").pack(pady=10)

    train_photo = Image.open('train_model.png')
    train_photo = train_photo.resize((width, height), Image.ANTIALIAS)
    train_photo = ImageTk.PhotoImage(train_photo)
    btn_train = Button(main_window, image=train_photo, command=train_model).pack(pady=10)

    test_photo = Image.open('testing_model.png')
    test_photo = test_photo.resize((width, height), Image.ANTIALIAS)
    test_photo = ImageTk.PhotoImage(test_photo)
    btn_test = Button(main_window, image=test_photo, command=partial(Model_Test.test_model_NoTrain, time, series_temps, series_GHG)).pack(pady=10)

    Learning_Rate_photo = Image.open('learning_rate.png')
    Learning_Rate_photo = Learning_Rate_photo.resize((width, height), Image.ANTIALIAS)
    Learning_Rate_photo = ImageTk.PhotoImage(Learning_Rate_photo)
    btn_Learning_Rate = Button(main_window, image=Learning_Rate_photo, command=Find_lr).pack(pady=10)

    Graph_photo = Image.open('show_graph.png')
    Graph_photo = Graph_photo.resize((width, height), Image.ANTIALIAS)
    Graph_photo = ImageTk.PhotoImage(Graph_photo)
    btn_Graph = Button(main_window, image=Graph_photo, command=partial(visual.ploting_graphes, time, series_temps, series_GHG)).pack(pady=10)

    main_window.mainloop()


time, series_temps, series_GHG = Load_Data()
menu()




