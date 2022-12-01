from InterfaceBuilder import InterfaceBuilder
from tasks.task3_backprop import Task3
import os
os.system('cls' if os.name == 'nt' else 'clear')

activations = ['tanh', 'sigmoid']
config = {
    "size": [],
    "activation": "tanh",
    "eta": 0.0001,
    "epochs": 5000,
    "include_bias": True,
}

task = Task3()


def selected_activation_changed(event):
    config["activation"] = event.widget.get()


def size_changed(value):
    config['size'] = value.replace(" ", "").split(",")


def eta_changed(value):
    try:
        x = float(value)
        config["eta"] = x
    except:
        pass


def epochs_changed(value):
    try:
        x = int(value)
        config["epochs"] = x
    except:
        pass


def include_bias_changed(value):
    config["include_bias"] = value == 1


def sumbit_handler():

    task.run(config)


builder = InterfaceBuilder(title="Single layer preceptron", data=None)
builder.lay_cmb_selection(
    "Activation function", activations, 1, selected_activation_changed)
builder.add_entry("Network size", lambda x: size_changed(x.get()))
builder.add_entry("Learning Rate", lambda x: eta_changed(x.get()))
builder.add_entry("Epochs", lambda x: epochs_changed(x.get()))
builder.add_checkbox("Include Bias", include_bias_changed)
builder.add_btn("Train", sumbit_handler)
builder.show()
