import marimo

__generated_with = "0.23.13"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Exploring Bias and Variance using Linear Fitting

    This script will walk you through the process of fitting a linear model using polynomial basis functions, and exploring the trade-off between bias and variance in models based on their complexity.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Data Generation

    The following section of the code generates the training and testing sets using the simple model
    $$ y = f(x) + \epsilon$$
    $$ f(x) = 5x(x-0.5)(x-1) $$
    where $\epsilon \sim N(0,0.01)$ is a Gaussian noise term with zero mean and standard deviation equal to 0.1.
    """)
    return


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt

    # Defining data generation model
    def f(x):
        return 5 * (x - 0) * (x - 0.5) * (x - 1)
    stdNoise = 0.1

    # Defining std of noise
    rng = np.random.default_rng(seed=42)
    N = 80

    # Setting a seed for random number generator
    x_train = np.sort(rng.random(size=N))
    f_train = f(x_train)

    # Generating training, validation and test data
    y_train = f_train + stdNoise * rng.normal(size=N)
    N = 20
    x_test = rng.random(size=N)
    y_test = f(x_test) + stdNoise * rng.normal(size=N)
    return f, np, plt, rng, stdNoise, x_test, x_train, y_test, y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Visualizing the data that was just generated
    """)
    return


@app.cell
def _(f, np, plt, x_test, x_train, y_test, y_train):
    ax = plt.gca()
    x = np.linspace(0, 1, 100)
    y = f(x)
    ax.plot(x, y, 'k-', label='True Model')
    ax.plot(x_train, y_train, 'k.', label='Training')
    ax.plot(x_test, y_test, 'r.', label='Testing')
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Fitting a Polynomial to the Data

    The functions below provide an implementation of a linear fitting of monomials to the data.
    """)
    return


@app.cell
def _(np):
    # Function that creates the X matrix as defined for fitting our model
    def create_X(x, deg):
        X = np.ones((len(x), deg + 1))
        for i in range(1, deg + 1):
            X[:, i] = x ** i
        return X

    # Function for predicting the response
    def predict(x, theta):
        return np.dot(create_X(x, len(theta) - 1), theta)

    # Function for fitting the model
    def fit(x, y, deg):
        return np.linalg.lstsq(create_X(x, deg), y, rcond=None)[0]

    # Function for computing the MSE
    def mse(y, yPred):
        se = (y - yPred) ** 2
        return np.mean(se)

    return fit, mse, predict


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    First, we fit the data to the entire training set and compute the corresponding training and test errors. Note that we are using MSE as a metric for performance. We refer to it as the error and apply it to the training and testing sets for comparison.
    """)
    return


@app.cell
def _(f, fit, mse, np, plt, predict, x_test, x_train, y_test, y_train):
    # Choose polynomial degree for this fit
    deg = 1

    # Fit model parameters (theta) on training data
    theta = fit(x_train, y_train, deg)

    # Evaluate training performance
    y_train_pred = predict(x_train, theta)
    err = mse(y_train, y_train_pred)
    print('Training Error / Cost = {:2.3}'.format(err))

    # Evaluate test performance
    y_test_pred = predict(x_test, theta)
    err = mse(y_test, y_test_pred)
    print('Test Error / Metric = {:2.3}'.format(err))

    # Build a dense grid for smooth plotting of the learned curve
    x_1 = np.linspace(0, 1, 100)

    # Predict on grid and compute ground-truth model values
    _y = predict(x_1, theta)
    y_model = f(x_1)

    # Plot true model, fitted model, and observed data
    plt.plot(x_1, y_model, 'k-', x_1, _y, 'b-', x_train, y_train, 'k.', x_test, y_test, 'r.')
    plt.legend(['True Model', 'Prediction', 'Training', 'Test'])
    plt.show()
    return x_1, y_model


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now we will compute the bias and variance of the models for various degrees as we change the training set. Note that we leave the test set out of this discussion since we are going to be comparing the results to the function that generated the data. This is something that is often not possible. In our case, we are able to do this since this is a synthetic dataset.
    """)
    return


@app.cell
def _(f, fit, mse, np, plt, predict, rng, stdNoise, x_1, y_model):
    # Number of model degrees to compare
    degList = [1, 3, 5]

    # Number of repeated simulations and training samples per simulation
    N_iter = 500
    N_samples = 100

    # Fit many models on newly generated datasets and collect predictions
    def getFittedModels(N_iter, N_samples, x, deg):
        _y_arr = np.empty((N_iter, len(x)))
        for k in range(N_iter):
            # Generate a fresh training set
            x_tmp = np.sort(rng.random(N_samples))
            f_tmp = f(x_tmp)
            y_tmp = f_tmp + stdNoise * rng.normal(size=N_samples)

            # Fit polynomial and predict on the evaluation grid
            theta = fit(x_tmp, y_tmp, deg)
            _y_arr[k, :] = predict(x, theta)

        # Average prediction and prediction variance across runs
        _y_mean = np.mean(_y_arr, axis=0)
        _y_var = np.var(_y_arr, axis=0)
        return (_y_arr, _y_mean, _y_var)

    fig1 = plt.figure()
    fig1.set_figwidth(12)
    fig1.set_figheight(2)

    for _degCt, _deg in enumerate(degList):
        _y_arr, _y_mean, _y_var = getFittedModels(N_iter, N_samples, x_1, _deg)

        plt.subplot(1, len(degList), _degCt + 1)
        plt.plot(x_1, _y_arr.T, '-', color=[0.7, 0.7, 0.7])

        # Set y-axis limits using a tuple, not a list
        plt.ylim((-1, 1))

        # Plot mean prediction and +/- one standard deviation
        plt.plot(x_1, _y_mean, 'b-', x_1, _y_mean - np.sqrt(_y_var), 'b--',
                 x_1, _y_mean + np.sqrt(_y_var), 'b--')

        # Plot the true underlying function
        plt.plot(x_1, y_model, 'k-')
        plt.title('Degree = {}'.format(_deg))

        print(
            'Deg {}: (Bias , Variance) = ({:2.5f} , {:2.5f})'.format(
                _deg, mse(y_model, _y_mean), np.mean(_y_var)
            )
        )

    plt.show()
    return N_iter, degList, getFittedModels


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We repeat the same analysis but this time showing the results for multiple numbers of samples.
    """)
    return


@app.cell
def _(N_iter, degList, getFittedModels, mse, np, plt, x_1, y_model):
    N_samples_List = (10, 20, 50, 100)

    for _N in N_samples_List:
        fig = plt.figure()
        fig.set_figwidth(12)
        fig.set_figheight(2)

        txt = 'N = {:3d}'.format(_N)

        for _degCt, _deg in enumerate(degList):
            _y_arr, _y_mean, _y_var = getFittedModels(N_iter, _N, x_1, _deg)

            plt.subplot(1, len(degList), _degCt + 1)
            plt.plot(x_1, _y_arr.T, '-', color=[0.7, 0.7, 0.7])

            # Set y-axis limits using a tuple, not a list
            plt.ylim((-1, 1))

            # Plot mean prediction and +/- one standard deviation
            plt.plot(x_1, _y_mean, 'b-', x_1, _y_mean - np.sqrt(_y_var), 'b--',
                x_1, _y_mean + np.sqrt(_y_var), 'b--')

            # Plot the true underlying function
            plt.plot(x_1, y_model, 'k-')
            plt.title('N = {}, Degree = {}'.format(_N, _deg))

            txt = txt + ' | Deg {}: (Bias , Variance) = ({:2.5f} , {:2.5f})'.format(
                _deg, mse(y_model, _y_mean), np.mean(_y_var))

        print(txt)
        plt.show()
    return


if __name__ == "__main__":
    app.run()
