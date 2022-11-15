from InterfaceBuilder import InterfaceBuilder
from tasks.task1 import Task1
from tasks.task2 import Task2

config = {
    "selected_features": {},
    "selected_labels": {},
    "eta": 0.01,
    "epochs": 100,
    "include_bias": False,
}

#task = Task1()
task = Task2()

def uniquely_add(cmb_event, dict):
    val = cmb_event.widget.get()
    id = cmb_event.widget.winfo_name()

    # Don't add if exits
    if val in list(dict.values()):
        return

    dict[id] = val


def selected_feature_changed(event):
    uniquely_add(event, config["selected_features"])


def selected_label_changed(event):
    uniquely_add(event, config["selected_labels"])


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
    "Feature Selection", task.features, 2, selected_feature_changed
)
builder.lay_cmb_selection(
    "Label Selection", task.labels, 2, selected_label_changed)
builder.add_entry("Learning Rate", lambda x: eta_changed(x.get()))
builder.add_entry("Epochs", lambda x: epochs_changed(x.get()))
builder.add_checkbox("Include Bias", include_bias_changed)
builder.add_btn("Train", sumbit_handler)
builder.show()
