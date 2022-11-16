import matplotlib.pyplot as plt


def visualize(model) -> None:
    # for dataset visualization after training

    if model.bias == False:
        plotbias = 0
    else:
        plotbias = model.b

    plt.figure('f1')
    plotbias = model.b

    fit11 = model.X.iloc[0:30, 0]
    fit12 = model.X.iloc[0:30, 1]

    fit21 = model.X.iloc[30:60, 0]
    fit22 = model.X.iloc[30:60, 1]

    plt.scatter(fit11, fit12)
    plt.scatter(fit21, fit22)

    x1Valus = []
    x2Values = []

    x1Valus.append(model.X.iloc[0, 0])
    x1Valus.append(model.X.iloc[1, 0])
    x1Valus.append(model.X.iloc[2, 0])
    x1Valus.append(50)

    x1Valus.append(model.X.iloc[40, 0])
    x1Valus.append(model.X.iloc[41, 0])
    x1Valus.append(model.X.iloc[42, 0])

    for i in x1Valus:
        x2Values.append(
            ((-plotbias) - (model.weights[0] * i))/model.weights[1])

    plt.plot(x1Valus, x2Values)
    colNames = model.X.columns.tolist()
    plt.xlabel(str(colNames[0]))
    plt.ylabel(str(colNames[1]))
    plt.show()
