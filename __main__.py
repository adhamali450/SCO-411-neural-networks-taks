from InterfaceBuilder import InterfaceBuilder
from tasks.task1 import task1

config = {
    'selected_features': set(),
    'selected_labels': set(),
    'eta': 0.01,
    'epochs': 100,
    'include_bias': False,
}

def selected_feature_changed(feature):
    config['selected_features'].add(feature)


def selected_label_changed(label):
    config['selected_labels'].add(label)


def eta_changed(value):
    try:
        x = float(value)
        config['eta'] = x
    except:
        pass


def epochs_changed(value):
    try:
        x = int(value)
        config['epochs'] = x
    except:
        pass


def include_bias_changed(value):
    config['include_bias'] = value == 1


def sumbit_handler():
    task1()

builder = InterfaceBuilder(title="Single layer preceptron", data=None)
builder.lay_cmb_selection("Feature Selection", [
                          1, 2, 3, 4, 5], 2, selected_feature_changed)
builder.lay_cmb_selection(
    "Label Selection", ['cat', 'dog', 'goat'], 2, selected_label_changed)
builder.add_entry("Learning Rate", lambda x: eta_changed(x.get()))
builder.add_entry("Epochs", lambda x: epochs_changed(x.get()))
builder.add_checkbox("Include Bias", include_bias_changed)
builder.add_btn("Train", sumbit_handler)
builder.show()
