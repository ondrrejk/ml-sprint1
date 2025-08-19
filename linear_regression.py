import random
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
