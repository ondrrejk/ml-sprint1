"""trenuju si, jak zpameti udelat kod na vypocet mse, rmse, mae, r2 apod"""


def mse(y_true, y_predicted):
    """mean squared error funkce

    Args:
        y_true (list[float]): true y values
        y_predicted (list[float]): predicted y values
    """
    return sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_predicted)) / len(y_predicted)


def rmse(y_true, y_predicted):
    """root mean squared error

    Args:
        y_true (list[float]): true y values
        y_predicted (list[float]): predicted y values
    """
    return mse(y_true, y_predicted) ** 0.5


def mae(y_true, y_predicted):
    """mean absolute error

    Args:
        y_true (list[float]): true y values
        y_predicted (list[float]): predicted y values
    """
    return sum(abs(yt - yp) for yt, yp in zip(y_true, y_predicted)) / len(y_predicted)


def r2(y_true, y_predicted):
    """r squared

    Args:
        y_true (list[float]): true y values
        y_predicted (list[float]): predicted y values
    """
    ss_r = sum(
        (yt - yp) ** 2 for yt, yp in zip(y_true, y_predicted)
    )  # sum of squared residuals
    y_mean = sum(y_true) / len(y_true)  # mean of the true y values
    ss_t = sum((yt - y_mean) ** 2 for yt in y_true)  # sum of squares total
    r_squared = 1 - ss_r / ss_t  # r squared
    return r_squared


true_y = [3, -0.5, 2, 7]  # true y values, can be random
predicted_y = [2.5, 0.0, 2, 8]  # temporary values, normally we would've calculated them

# printing our calculated values
print("MSE:", mse(true_y, predicted_y))
print("RMSE:", rmse(true_y, predicted_y))
print("MAE:", mae(true_y, predicted_y))
print("R2:", r2(true_y, predicted_y))
