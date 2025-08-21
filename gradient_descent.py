# GRADIENT DESCENT
# 1. Máme nějakou funkci chyby (loss function). Pro lineární regresi to je MSE (Mean Squared Error).
#   MSE(x,y) = sum((y_i-(wx_i+b))^2)/len(x nebo y)
# 2. Chceme najít takové w a b, které tu chybu minimalizují.
#   Nelze to vždy spočítat vzorcem, tak použijeme iterativní postup.
# 3. Gradient = směr, kterým chyba roste nejrychleji.
#   Když chceme chybu snižovat, musíme jít v opačném směru gradientu.
# 4. Update pravidlo: w←w−η⋅∂w/∂L, b←b−η⋅∂b/∂L
#   η = learning rate (jak velký krok uděláme).
#   derivace nám řeknou, jak moc se chyba mění vzhledem k w a b.

import random

# trenovaci data x, y
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]  # dokonala linearni data: y=2x

# pocatecni parametry
w = random.uniform(-1, 1)
b = random.uniform(-1, 1)

# ucici rychlost (learning rate)
lr = 0.01

# pocet iteraci
epochs = 1000

# promenna n = pocet data pointu
n = len(x)  # lze zamenit za len(y)

for epoch in range(epochs):
    # spocitame predikce
    y_pred = (w * xi + b for xi in x)

    # spocitame gradienty podle vzorcu
    #   hledame, kde je v gradient descent "propasti" derivative o hodnote 0
    dw = (-2 / n) * sum(xi * (yi - y_hat) for xi, yi, y_hat in zip(x, y, y_pred))
    db = (-2 / n) * sum((yi - y_hat) for yi, y_hat in zip(y, y_pred))

    # update parametru
    w -= lr * dw
    b -= lr * db

    # loss (jen pro info)
    loss = sum((yi - y_hat) ** 2 for yi, y_hat in zip(y, y_pred)) / n

    # obcas vypiseme progres
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: w={w:.4f}, b={b:.4f}, loss={loss:.4f}")

print(f"\nFinal model: y = {w:.2f}x + {b:.2f}")
