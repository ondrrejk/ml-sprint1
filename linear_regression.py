import random
from random import shuffle
import matplotlib.pyplot as plt

# vstupni data
x = list(range(0, 101))
y = [3 * xi + 2 + random.uniform(-5, 5) for xi in x]  # y = 3x + 2 + šum

# rychly nahled
print(x[:5], y[:5])


# pomocne funkce
def mean(numbers):
    return sum(numbers) / len(numbers)


def variance(numbers):
    m = mean(numbers)
    return sum((xi - m) ** 2 for xi in numbers) / len(numbers)


def covariance(xcov, ycov):
    mx, my = mean(xcov), mean(ycov)
    return sum((xi - mx) * (yi - my) for xi, yi in zip(xcov, ycov)) / len(xcov)


# analyticke reseni, w = slope, b = bias
w = covariance(x, y) / variance(x)
b = mean(y) - w * mean(x)

# predikce y
y_pred = [w * xi + b for xi in x]

# matplotlib graf
# zobrazi data pointy, jejich x a y, nazev dat
plt.scatter(x, y, label="Data")

# x a y primky, barva primky, nazev primky
plt.plot(x, y_pred, color="red", label="Predikovaná přímka")

# ukaze legendu toho co jsme si pojmenovali pomoci label
plt.legend()

# nazev grafu
plt.title("Můj první graf")

# pojmenovani x a y axis
plt.xlabel("x")
plt.ylabel("y")

# mrizka v grafu
plt.grid(True)

# otevrit graf
plt.show()

# normalizace dat
def normalize(lst):
    min_val = min(lst)
    max_val = max(lst)
    return [(x - min_val) / (max_val - min_val) for x in lst]


x_norm = normalize(x)

# vypocitani mean squared error
def mse(y_true, y_p):
    n = len(y_true)
    return sum((yt - yp)**2 for yt, yp in zip(y_true, y_p)) / n

print("MSE:", mse(y, y_pred))

# ROZDELENI DAT NA TRAIN A TEST
# zamicha data
data = list(zip(x,y))
shuffle(data)

# rozdeli data na 80%-20% split
split = int(0.8*len(data)) # index na poloze 80% z celych dat

train = data[:split] # vsechno az po index
test = data[split:] # od indexu dale

x_train, y_train = zip(*train)
x_test, y_test = zip(*test)
