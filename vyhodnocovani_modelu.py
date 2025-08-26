"""Vyhodnocování modelu pomocí různých variací MSE a R^2"""


def ss_res(ytrue, ypred):
    """sum of squared residuals"""
    errors = [(yt - yp) ** 2 for yt, yp in zip(ytrue, ypred)]
    return sum(errors)


def mse(ytrue, ypred):
    """mean squared error"""
    return ss_res(ytrue, ypred) / len(ypred)


def rmse(ytrue, ypred):
    """root mean squared error"""
    return mse(ytrue, ypred) ** 0.5


def mae(ytrue, ypred):
    """mean absolute error"""
    errors = [abs(yt - yp) for yt, yp in zip(ytrue, ypred)]
    return sum(errors) / len(errors)


def r2(ytrue, ypred):
    """r squared"""
    mean_y = sum(ytrue) / len(ytrue)
    ssr = ss_res(ytrue, ypred)  # sum of squares of the residuals
    sst = sum(
        (yt - mean_y) ** 2 for yt in ytrue
    )  # total sum of squares (from the mean)
    return 1 - ssr / sst


# y values
y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]

# printing our calculated values
print("MSE:", mse(y_true, y_pred))
print("RMSE:", rmse(y_true, y_pred))
print("MAE:", mae(y_true, y_pred))
print("R2:", r2(y_true, y_pred))
