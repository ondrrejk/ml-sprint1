import random
import matplotlib.pyplot as plt

# vstupni data
x = list(range(0, 101))
y = [3 * xi + 2 + random.uniform(-5, 5) for xi in x] # y = 3x + 2 + šum

# rychly nahled
print(x[:5], y[:5])

# pomocne funkce
def mean(numbers):
    return sum(numbers) / len(numbers)
    
def variance(numbers):
    m = mean(numbers)
    return sum((xi - m) ** 2 for xi in numbers) / len(numbers)

def covariance(x, y):
    mx, my = mean(x), mean(y)
    return sum((xi - mx) * (yi - my) for xi, yi in zip(x, y)) / len(x)

# analyticke reseni, w = slope, b = bias
w = covariance(x, y) / variance(x)
b = mean(y) - w*mean(x)

# predikce y
y_pred = [w*xi + b for xi in x]

# matplotlib graf
plt.scatter(x, y, label="Data")
plt.plot(x, y_pred, color="red", label="Predikovaná přímka")
plt.legend()
plt.show()

# matplotlib zbytecnosti
plt.title("Moje první grafy")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
