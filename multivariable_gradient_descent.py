import numpy as np

# 5 vzorků, 3 vstupní proměnné
X = np.array([[1, 2, 3], [2, 0, 1], [3, 1, 2], [4, 3, 0], [5, 2, 1]], dtype=float)

# Skutecne hodnoty y
y = np.array([14, 8, 13, 17, 19], dtype=float)

# inicializace parametru
n_features = X.shape[1]  # vrati 3, jako 3 sloupce, neboli n_features
w = np.zeros(
    n_features
)  # vytvori array, a kdyz zadame do .zeros(3), da nam to array [0, 0, 0]
b = 0.0  # bias
alpha = 0.01  # learning rate
epochs = 1000  # pocet opakovani

# gradient descent
for epoch in range(epochs):
    # predikce
    y_pred = X.dot(w) + b  # dot product, matrix * vektor

    # chyba
    error = (
        y - y_pred
    )  # rozdil mezi skutecnou hodnotou y a predikci, delka vektoru je stejna jako pocet vzorku

    # gradient pro w a b
    dw = -(2 / len(y)) * X.T.dot(error)  # transpozice matice
    db = -(2 / len(y)) * np.sum(error)

    # aktualizace parametru
    w = w - alpha * dw
    b = b - alpha * db

# vysledek
print("Váhy w:", w)
print("Bias b:", b)

# test predikce
new_sample = np.array([3, 2, 1])
prediction = new_sample.dot(w) + b
print("Predikce pro nový vzorek [3,2,1]:", prediction)
