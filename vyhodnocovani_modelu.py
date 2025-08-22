def ss_res(y_true, y_pred):
    errors = [(yt - yp) ** 2 for yt, yp in zip(y_true, y_pred)]
    return sum(errors)


def mse(y_true, y_pred):
    return ss_res(y_true, y_pred) / len(y_pred)


def rmse(y_true, y_pred):
    return mse(y_true, y_pred) ** 0.5


def mae(y_true, y_pred):
    errors = [abs(yt - yp) for yt, yp in zip(y_true, y_pred)]
    return sum(errors) / len(errors)


def r2(y_true, y_pred):
    mean_y = sum(y_true) / len(y_true)
    ssr = ss_res(y_true, y_pred)  # sum of squares of the residuals
    sst = sum(
        (yt - mean_y) ** 2 for yt in y_true
    )  # total sum of squares (from the mean)
    return 1 - ssr / sst


y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]

print("MSE:", mse(y_true, y_pred))
print("RMSE:", rmse(y_true, y_pred))
print("MAE:", mae(y_true, y_pred))
print("R2:", r2(y_true, y_pred))
