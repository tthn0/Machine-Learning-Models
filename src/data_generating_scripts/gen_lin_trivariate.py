import math
import random


def noise():
    return 0.2 * (random.random() - 0.5)


def f(x):
    return 10 * (0.5 * math.sin(1.1 * x) + noise() + 3)


def g(x):
    return 10 * (0.6 * math.cos(0.9 * x) + noise() - 5)


def h(x):
    return 10 * (-1 / 250 * abs(x - 6.5) ** 2.5 + noise() + 10)


x1 = ""
x2 = ""
x3 = ""
y = ""

for i in {x * 0.02 for x in range(-350, 350)}:
    x1 += "%.2f , " % f(i)
    x2 += "%.2f , " % g(i)
    x3 += "%.2f , " % h(i)
    y += "%.2f , " % (f(i) - g(i) + 25 * noise())

with open("generated.csv", "w") as f:
    f.write(x1[:-2] + "\n")
    f.write(x2[:-2] + "\n")
    f.write(x3[:-2] + "\n")
    f.write(y[:-2])
