import marimo

__generated_with = "0.23.13"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Exploring Logistic Regression

    This notebook explores 1D and 2D logistic regression using manual fitting on random gaussian data.
    """)
    return


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    import torch

    def sigmoid(x):
      return 1/(1+np.exp(-x))

    return np, plt, sigmoid


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Logistic Regression in 1D

    Let's start by generating a subset of data. The error is computed over the entire dataset just for illustration since we are not fitting a model yet.
    """)
    return


@app.cell
def _(mo):
    n_samples_slider = mo.ui.slider(10, 500, value=100, label="Number of samples")
    n_samples_slider  # display the slider
    return (n_samples_slider,)


@app.cell
def _(n_samples_slider, np, plt):
    n_samples = n_samples_slider.value

    # Generating some data coming from two Gaussian Distributions
    x0 = np.random.normal(0, 1, n_samples)
    y0 = np.zeros(x0.shape)
    x1 = np.random.normal(2, 1, n_samples)
    y1 = np.ones(x1.shape)
    x = np.concatenate((x0, x1))

    # Combining data points
    idx = np.argsort(x)
    x = x[idx]
    y = np.concatenate((y0, y1))
    y = y[idx]

    # Plotting the datapoints
    plt.plot(x0, y0, 'o', x1, y1, 'o')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(['Samples with y = 0', 'Samples with y = 1'])
    plt.grid(True)
    plt.gca()
    return x, y


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Next, we are selecting some of the parameters manually to see how the fit looks.
    """)
    return


@app.cell
def _(mo):
    w_slider = mo.ui.slider(-2, 2, value=0, step=0.1, label="w")
    b_slider = mo.ui.slider(-4, 4, value=0, step=0.1, label="b")
    th_slider = mo.ui.slider(0, 1, value=0.5, step=0.02, label="threshold")

    mo.hstack([w_slider, b_slider, th_slider])
    return b_slider, th_slider, w_slider


@app.cell
def _(b_slider, np, plt, sigmoid, th_slider, w_slider, x, y):
    w = w_slider.value
    b = b_slider.value
    th = th_slider.value

    # Creating a dense set of x values and their prediction for visualization
    x_full = np.linspace(np.min(x), np.max(x), 100)
    y_full_hat = sigmoid(w * x_full + b)

    # Estimate for the random samples
    yhat = sigmoid(w * x + b)

    plt.plot(x[yhat<=th], y[yhat<=th], '.', x[yhat>th], y[yhat>th], '.')
    plt.plot(x_full, y_full_hat, 'k-', linewidth=1)
    plt.axhline(y=th, color='red', linestyle='--')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(['y_pred = 0', 'y_pred = 1', 'Logistic Model', 'Threshold'])
    plt.grid(True)
    plt.gca()
    return b, th, w


@app.cell
def _(b, mo, np, sigmoid, th, w, x, y):
    def compute_model_error(w, b, th):
        """Compute the misclassification error, precision, and recall for a logistic regression model.

        Args:
            w (float): Weight parameter.
            b (float): Bias parameter.
            th (float): Threshold for converting probabilities to class labels.
            x_test (np.ndarray): Test feature values.
            y_test (np.ndarray): True test labels (0 or 1).

        Returns:
            dict: Dictionary containing accuracy, misclassification error, precision, and recall.
        """
        _yhat = sigmoid(w * x + b)
        _y_pred = (_yhat >= th).astype(int)

        _tp = np.sum((_y_pred == 1) & (y == 1))
        _tn = np.sum((_y_pred == 0) & (y == 0))
        _fp = np.sum((_y_pred == 1) & (y == 0))
        _fn = np.sum((_y_pred == 0) & (y == 1))

        _accuracy = (_tp + _tn) / y.shape[0]
        _error = 1 - _accuracy
        _precision = _tp / (_tp + _fp) if (_tp + _fp) > 0 else 0.0
        _recall = _tp / (_tp + _fn) if (_tp + _fn) > 0 else 0.0

        return {
            "accuracy": _accuracy,
            "misclassification_error": _error,
            "precision": _precision,
            "recall": _recall,
        }

    _val = compute_model_error(w, b, th)

    mo.vstack([mo.md(
        f"**Accuracy:** {_val["accuracy"]:.2%}<br>"
        f"**Precission:** {_val["precision"]:.2%}<br>"
        f"**Recall:** {_val["recall"]:.2%}"
    )])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Logistic Regression in 2D

    Let's start by generating a subset of data.
    """)
    return


@app.cell
def _(mo):
    # UI: how many 2‑D points to generate
    n_samples_2d = mo.ui.slider(10, 500, value=200, label="Number of 2‑D samples")
    n_samples_2d      # ← last expression → the slider is rendered
    return (n_samples_2d,)


@app.cell
def _(n_samples_2d, np, plt):
    # Read the slider value (cannot be done in the same cell that defines it)
    N2 = n_samples_2d.value

    # Two Gaussian blobs, one for each class
    rng = np.random.default_rng()
    x0_2d = rng.normal(loc=[-2, -2], scale=1.0, size=(N2 // 2, 2))
    y0_2d = np.zeros(N2 // 2, dtype=int)

    x1_2d = rng.normal(loc=[ 2,  2], scale=1.0, size=(N2 - N2 // 2, 2))
    y1_2d = np.ones(N2 - N2 // 2, dtype=int)

    # Concatenate and shuffle the dataset
    X_2d = np.vstack((x0_2d, x1_2d))
    y_2d = np.concatenate((y0_2d, y1_2d))
    perm = rng.permutation(N2)
    X_2d = X_2d[perm]
    y_2d = y_2d[perm]

    # Scatter plot – colour by true class
    plt.scatter(
        X_2d[:, 0], X_2d[:, 1],
        c=y_2d, cmap="coolwarm", edgecolor="k", alpha=0.7
    )
    plt.title("2‑D samples")
    plt.xlabel("x₁")
    plt.ylabel("x₂")
    plt.axis("equal")
    plt.grid(True)
    plt.gca()
    return (X_2d,)


@app.cell
def _(mo):
    # Model parameters (weights, bias, probability threshold)
    w1_slider = mo.ui.slider(-4, 4, value=0.0, step=0.1, label="w1")
    w2_slider = mo.ui.slider(-4, 4, value=0.0, step=0.1, label="w2")
    b_2d_slider  = mo.ui.slider(-4, 4, value=0.0, step=0.1, label="b")
    th_2d_slider = mo.ui.slider(0, 1, value=0.5, step=0.01, label="Threshold")

    # Display the sliders (they are shown side‑by‑side)
    mo.hstack([w1_slider, w2_slider, b_2d_slider, th_2d_slider])
    return b_2d_slider, th_2d_slider, w1_slider, w2_slider


@app.cell
def _(X_2d, b_2d_slider, np, plt, sigmoid, th_2d_slider, w1_slider, w2_slider):
    # Read parameter values (must be a different cell)
    w1 = w1_slider.value
    w2 = w2_slider.value
    b_2d  = b_2d_slider.value
    th_2d = th_2d_slider.value

    # Grid on which we evaluate the model
    grid_res = 200
    x_grid = np.linspace(X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1, grid_res)
    y_grid = np.linspace(X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1, grid_res)
    xx, yy = np.meshgrid(x_grid, y_grid)
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    # Probability for each grid point
    probs = sigmoid(grid_points @ np.array([w1, w2]) + b_2d).reshape(grid_res, grid_res)

    # Probability for sample points
    y_2d_pred = sigmoid(X_2d @ np.array([w1, w2]) + b_2d)

    # Plot the heat‑map and overlay the scatter points
    #plt.subplot(1, 2, 2)
    heat = plt.imshow(
        probs,
        extent=(x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()),
        origin="lower",
        cmap="viridis",
        alpha=0.8,
        aspect="auto"
    )
    plt.colorbar(heat, label="P(y=1)")
    plt.title("Predicted probability heat‑map")
    plt.xlabel("x₁")
    plt.ylabel("x₂")
    plt.axis("equal")

    # Optional: overlay the same scatter points (transparent)
    plt.scatter(
        X_2d[:, 0], X_2d[:, 1],
        c=(y_2d_pred>th_2d), cmap="coolwarm", edgecolor="k", alpha=0.4
    )

    # Return the axes so Marimo displays the figure
    plt.grid(True)
    plt.gca()
    return


if __name__ == "__main__":
    app.run()
