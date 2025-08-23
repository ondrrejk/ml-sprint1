import numpy as np

# utility: split, scaling, metrics


def train_test_split(X, y, test_size=0.2, seed=42):
    rng = np.random.default_rng(
        seed
    )  # Vytvoří nový generátor náhodných čísel (novější API než globální np.random). Díky seedu míchá vždy stejně.
    n = len(
        X
    )  # Počet vzorků (řádků) v X. Očekáváme, že X.shape = (n, d(radky, sloupce)).
    idx = np.arange(n)  # Vektor indexů [0, 1, 2, ..., n-1].
    rng.shuffle(idx)  # Náhodně promíchá indexy in-place (bez kopie).
    n_test = int(n * test_size)  # Kolik vzorků půjde do testu.
    test_idx = idx[:n_test]  # Rozdělí promíchané indexy na test.
    train_idx = idx[n_test:]  # Rozdělí promíchané indexy na trénink.
    return (
        X[train_idx],
        X[test_idx],
        y[train_idx],
        y[test_idx],
    )  # Vráti X_train, X_test, y_train, y_test přes pokročilé indexování NumPy.


def standardize(X, eps=1e-8):  # Standardizace featur: Z-score po sloupcích.
    mean = X.mean(axis=0)  # Průměr každého sloupce X. Výsledek tvaru (d,).
    std = X.std(axis=0)  # Směrodatná odchylka každého sloupce. Tvar (d,).
    std = np.where(
        std < eps, 1.0, std
    )  # Ochrana proti dělení nulou: příliš malé std nahradíme 1.0, zbytek zustane std.
    Xs = (
        (X - mean) / std
    )  # Broadcasting: odečte sloupcové průměry a vydělí sloupcovým std. Xs má stejný tvar jako X.
    return (
        Xs,
        mean,
        std,
    )  # Vrátí škálovaná data a parametry, abychom jimi škálovali i test.


def apply_standardize(X, mean, std):  # Aplikace stejného škálování na nová data.
    std = np.where(std == 0, 1.0, std)  # Ještě jednou ochrana proti nule.
    return (X - mean) / std  # Stejné Z-score s danými mean a std.


def mse(y_true, y_pred):
    return np.mean(
        (y_true - y_pred) ** 2
    )  # Mean squared error. y_true a y_pred mají tvar (n,)


def rmse(y_true, y_pred):
    return np.sqrt(mse(y_true, y_pred))  # Odmocnina MSE, zpátky v jednotkách cíle.


def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))  # Průměr absolutních hodnot chyb.


def r2(y_true, y_pred):
    y_bar = np.mean(y_true)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_bar) ** 2)
    return (
        1 - ss_res / ss_tot
    )  # Koeficient determinace. ss_res je zbytková suma čtverců, ss_tot celková. 1 je ideál, 0 je „stejné jako průměr“.


# Model
def predict(X, w, b):
    return X @ w + b


# Predikce lineárního modelu:

#   X: (n, d), w: (d,), výsledek X @ w: (n,).
#   +b se broadcastne na každý prvek.
#   Pro dot product je důležité použít operátor @ namísto *.


# Trénink: Gradient descent
def fit_linear_gd(
    X,
    y,
    lr=0.1,
    epochs=1000,
    l2=0.0,  # Ridge koeficient (lambda)
    l1=0.0,  # Lasso koeficient (lambda)
    verbose_every=100,
):  # Hlavní trenér. lr je krok učení, epochs počet průchodů, l2/l1 penalizace vah, verbose_every frekvence logu.
    # Gradient descent pro multivariátní lineární regresi s volitelnou L2 a L1.
    # Regularizace se aplikuje jen na w (ne na bias b).
    n, d = X.shape  # Počet vzorků a počet featur.
    w = np.zeros(d, dtype=float)
    b = 0.0
    # Inicializace parametrů na nulu. w tvar (d,), b skalar.
    for epoch in range(1, epochs + 1):
        # Hlavní tréninková smyčka. 1 až epochs včetně.
        y_pred = predict(X, w, b)  # Dopředný průchod: y_pred tvar (n,).
        error = y - y_pred  # Residua. Klidně by šlo residuals.
        dw = -(2.0 / n) * (X.T @ error)
        # X.T: (d, n), error: (n,), takže X.T @ error: (d,).
        # Faktor -(2/n) vyplývá z derivace weights podle y_pred = Xw+b
        db = -(2.0 / n) * np.sum(
            error
        )  # Gradient MSE vůči biasu b. Suma přes všechna residua, krát -(2/n).
        if l2 > 0:
            dw += 2.0 * l2 * w
            # Ridge penalizace
        if l1 > 0:
            dw += l1 * np.sign(w)
            # Lasso penalizace
        w -= lr * dw
        b -= lr * db
        # Update parametrů po směru záporného gradientu. Když ti to diverguje, lr je moc veliké. Když to leze jak šnek, je moc malé.
        if verbose_every and epoch % verbose_every == 0:
            loss = mse(y, y_pred)
            print(f"epoch {epoch:4d}  MSE={loss:.6f}")
            # Logování průběhu. Pomáhá chytat „jojo efekt“ gradientu a další katastrofy.
        return w, b  # Vrátí naučené koeficienty.


# Demo (spouští se jen při python linear_multi.py)

if (
    __name__ == "__main__"
):  # Zajistí, že tenhle blok běží jen když soubor spouštíš přímo, ne při importu.
    np.random.seed(2)
    
