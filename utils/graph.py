import matplotlib.pyplot as plt

def plot_actual_vs_predicted(y_actual, y_pred, title="Predicted vs Actual Total Cost"):
    """
    Plots the predicted vs actual values for the latest updated model.

    Parameters:
    - y_actual (pd.Series or np.array): Actual values.
    - y_pred (pd.Series or np.array): Predicted values.
    - title (str): Title of the plot.

    Returns:
    - None
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(y_actual, y_pred, alpha=0.6, edgecolor='k', label="Predictions")
    plt.plot([y_actual.min(), y_actual.max()], [y_actual.min(), y_actual.max()], 
             'r--', lw=2, label="Ideal Prediction")
    plt.title(title)
    plt.xlabel("Actual Total Cost (TZS)")
    plt.ylabel("Predicted Total Cost (TZS)")
    plt.legend()
    plt.grid()
    plt.show()
