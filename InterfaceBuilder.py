import tkinter as tk
from tkinter import *
from tkinter import ttk

class InterfaceBuilder:
    common_style = {
        'padx': 10,
        'pady': 10,
        'sticky': 'ew'
    }

    def __init__(self, title, data):
        self.data = data

        self.window = tk.Tk()
        self.window.title(title)

        self.curr_row = 0

    def lay_cmb_selection(self, text, selectable, max, handler):
        frame = tk.Frame(master=self.window)
        frame.grid(
            row=self.curr_row, padx=self.common_style['padx'], pady=self.common_style['pady'], sticky=self.common_style['sticky'])

        tk.Label(frame, text=text).grid(column=0, padx=10)

        for i in range(max):
            cmb = ttk.Combobox(frame, values=selectable)
            cmb.set("Item " + str(i + 1))
            cmb.bind("<<ComboboxSelected>>",
                     lambda event: handler(event.widget.get()))
            cmb.grid(row=0, column=i + 1, padx=5)

        self.curr_row += 1

    def add_btn(self, text, handler):
        btn = tk.Button(self.window, text=text,
                        command=handler, anchor=CENTER)
        btn.grid(
            row=self.curr_row, padx=self.common_style['padx'], pady=self.common_style['pady'], sticky=self.common_style['sticky'])

        self.curr_row += 1

    def add_checkbox(self, text, handler):
        var = IntVar()
        cb = Checkbutton(self.window, text=text, variable=var,
                         command=lambda: handler(var.get()))
        cb.grid(
            row=self.curr_row, padx=self.common_style['padx'], pady=self.common_style['pady'], sticky='w')

        self.curr_row += 1

    def add_entry(self, text, handler):
        frame = tk.Frame(master=self.window)
        frame.grid(
            row=self.curr_row, padx=self.common_style['padx'], pady=self.common_style['pady'], sticky=self.common_style['sticky'])

        tk.Label(frame, text=text).grid(column=0, padx=10)

        sv = StringVar()
        sv.trace("w", lambda name, index, mode, sv=sv: handler(sv))
        entry = Entry(frame, textvariable=sv)
        entry.grid(row=0, column=1, padx=5)

        self.curr_row += 1

    def show(self):
        self.window.mainloop()
