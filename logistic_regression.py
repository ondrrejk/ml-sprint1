import numpy as np
from sklearn.datasets import make_classification

# === 1. Data ===
# Vygenerujeme si jednoduchý dataset (100 vzorků, 2 featury, binární cíl 0/1)
X, y = make_classification(n_samples=100, n_features=2, n_classes=2, random_state=42)


# === 2. Pomocné funkce ===
def sigmoid(z):
    return 1 / (
        1 + np.exp(-z)
    )  # Funkce, která mapuje libovolné číslo do intervalu (0,1)


def predict(X, w, b):
    return sigmoid(X @ w + b)  # Pravděpodobnost, že výstup je 1


def binary_accuracy(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    return np.mean(y_true == y_pred)


# === 3. Trénink (gradient descent) ===
def fit_logistic_gd(X, y, lr=0.1, epochs=1000, verbose_every=100):
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0

    for epoch in range(1, epochs + 1):
        y_prob = predict(X, w, b)  # Dopředný průchod (pravděpodobnosti)
        error = y - y_prob  # Rozdíl mezi skutečností a predikcí
        dw = -(1.0 / n) * (X.T @ error)  # Gradient vůči vahám
        db = -(1.0 / n) * np.sum(error)  # Gradient vůči biasu
        w -= lr * dw
        b -= lr * db

        if verbose_every and epoch % verbose_every == 0:
            acc = binary_accuracy(y, y_prob)
            print(f"epoch {epoch:4d}  accuracy={acc:.4f}")

    return w, b


# === 4. Demo ===
if __name__ == "__main__":
    w, b = fit_logistic_gd(X, y, lr=0.1, epochs=1000, verbose_every=200)
    probs = predict(X, w, b)
    acc = binary_accuracy(y, probs)
    print(f"Final accuracy on training: {acc:.4f}")
