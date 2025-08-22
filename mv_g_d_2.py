import numpy as np

# Simulace dat
np.random.seed(42)
X = np.random.rand(100, 3)  # 100 vzorků, 3 featury
true_w = np.array([3, -2, 5])
true_b = 4
y = X @ true_w + true_b + np.random.randn(100) * 0.5  # přidáme šum

# Inicializace parametrů
w = np.zeros(3)
b = 0.0
lr = 0.1
epochs = 1000

# Gradient descent
for epoch in range(epochs):
    y_pred = X @ w + b
    error = y - y_pred

    # gradienty
    dw = -(2 / len(X)) * (X.T @ error)  # vektor vah
    db = -(2 / len(X)) * np.sum(error)  # bias

    # update
    w -= lr * dw
    b -= lr * db

    if epoch % 100 == 0:
        mse = np.mean(error**2)
        print(f"Epoch {epoch}: MSE={mse:.4f}")

print("Naučené váhy:", w)
print("Naučený bias:", b)