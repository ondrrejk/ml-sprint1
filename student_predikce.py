# model na predikci toho, jestli student projde testem podle vstupnich dat
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# --------------------------
# 1. Vytvoříme si fake dataset
# --------------------------
np.random.seed(42)  # ať je to opakovatelné

n = 200  # počet studentů

# Features
hours = np.random.randint(0, 20, n)  # hodiny učení
homework = np.random.randint(0, 10, n)  # počet odevzdaných úkolů
attendance = np.random.randint(50, 100, n)  # procenta docházky
grades = np.random.randint(1, 5, n)  # průměrná známka (1=nejlepší, 5=nejhorší)

# Label (zjednodušená "pravda")
# Studenti s více hodinami, lepší docházkou a známkou mají větší šanci projít
y = (
    hours * 0.3
    + homework * 0.2
    + attendance * 0.3
    - grades * 5
    + np.random.randn(n) * 5
) > 25
y = y.astype(int)  # 1 = projde, 0 = neprojde

# DataFrame (jen pro hezčí pohled)
data = pd.DataFrame(
    {
        "hours": hours,
        "homework": homework,
        "attendance": attendance,
        "grades": grades,
        "pass_exam": y,
    }
)

print(data.head())

# --------------------------
# 2. Rozdělení na trénovací/testovací data
# --------------------------
X = data[["hours", "homework", "attendance", "grades"]]
y = data["pass_exam"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --------------------------
# 3. Trénink logistické regrese
# --------------------------
model = LogisticRegression()
model.fit(X_train, y_train)

# --------------------------
# 4. Vyhodnocení
# --------------------------
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"Přesnost modelu: {acc:.2f}")
print("Predikce prvních 10 studentů:", y_pred[:10])
