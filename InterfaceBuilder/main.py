# import pandas as pd
import tkinter as tk
from tkinter import *
from tkinter import ttk

# Data
selected_features = set()
selected_labels = set()


window = tk.Tk()
window.title("Single layer preceptron")

cmb_style = {
    'row': 0,
    'column': 0,
    'padx': 10,
    'pady': 10
}


def lay_cmb_selection(text, selectable, max, selected, winow, style=cmb_style):
    frame = tk.Frame(master=winow)
    frame.grid(
        row=style['row'], column=style['column'], padx=style['padx'], pady=style['pady'])

    tk.Label(frame, text=text).grid(column=0, padx=10)

    for i in range(max):
        cmb = ttk.Combobox(frame, values=selectable)
        cmb.set("Item " + str(i + 1))
        cmb.bind("<<ComboboxSelected>>", selected_features.add(cmb.get()))
        cmb.grid(row=0, column=i + 1, padx=5)


lay_cmb_selection("Feature Selection", [1, 2, 3, 4, 5], 2, selected_features, window, {
                  'row': 0, 'column': 0, 'padx': 10, 'pady': 10})
lay_cmb_selection("Label Selection", ['cat', 'dog', 'goat'], 2, selected_labels, window,
                  {'row': 1, 'column': 0, 'padx': 10, 'pady': 10})

window.mainloop()


class InterfaceBuilder:
    def __init__(self, data):
        self.data = data

    def build(self):
        list = [1, 2, 3, 4, 5]
        pass
