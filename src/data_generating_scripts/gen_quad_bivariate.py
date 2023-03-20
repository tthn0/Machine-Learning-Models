import math
import random


def noise():
    return 2 * (random.random() - 0.5)


def f(x, y):
    return 0.15 + x**2 / 5 - 0.12 * x + y**2 / 7 - 0.08 * y + noise()


x = ""
y = ""
z = ""

for i in {x for x in range(-8, 8, 2)}:
    for j in {x for x in range(-8, 8, 2)}:
        x += "%.2f , " % (i + noise())
        y += "%.2f , " % (j + noise())
        z += "%.2f , " % f(i, j)

with open("generated.csv", "w") as f:
    f.write(x[:-2] + "\n")
    f.write(y[:-2] + "\n")
    f.write(z[:-2])
