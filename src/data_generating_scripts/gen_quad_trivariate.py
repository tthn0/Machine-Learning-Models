from math import cos, sin
import random


def noise():
    return random.random() - 0.5


def f(x):
    return (10 + 2 * cos(5 * x)) * cos(x) + noise()


def g(x):
    return sin(5 * x) + noise()


def h(x):
    return (10 + 2 * cos(5 * x)) * sin(x) + noise()


x1 = ""
x2 = ""
x3 = ""
y = ""

for i in {x * 0.05 for x in range(-350, 350)}:
    x1 += "%.2f , " % f(i)
    x2 += "%.2f , " % g(i)
    x3 += "%.2f , " % h(i)
    y += "%.2f , " % (2 * h(i) + 2 * noise())

with open("generated.csv", "w") as f:
    f.write(x1[:-2] + "\n")
    f.write(x2[:-2] + "\n")
    f.write(x3[:-2] + "\n")
    f.write(y[:-2])
