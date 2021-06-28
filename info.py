from tkinter import messagebox
import tensorflow as tf


def info_lr_before():
    messagebox.showinfo('info', "The model now is going to train with different lr between the values 1e-8 to 1 and output\
 the loss.than, we will see which lr has the smallest loss")


def info_lr_after():
    messagebox.showinfo('info', "The model Has been Trained with different Learning rates!")


def info_conclusion():
    messagebox.showinfo('info', "we can see that the best values for lr are between 1e-4 to 1e-2 as they produce the best\
 loss rates")


def info_error(results_temps, x_valid, size_window):
    MAE_error = tf.keras.losses.MAE(results_temps, x_valid[:, 0][:-size_window].reshape(-1, 1)).numpy().mean()
    MAPE_error = tf.keras.losses.MAPE(results_temps, x_valid[:, 0][:-size_window].reshape(-1, 1)).numpy().mean()
    lines = ['Mean absolute error:' + str(MAE_error), 'Relative error:' + str(MAPE_error) + '%']
    messagebox.showinfo("The Model's Error", "\n".join(lines))

