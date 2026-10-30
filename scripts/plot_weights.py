#!/usr/bin/env -S uv run python
import csv
import sys

import matplotlib.pyplot as plt


def usage():
    print("Usage: plot_weights.py {csv_file}")


if len(sys.argv) <= 1:
    usage()
    exit()

data: list[dict[str, str]] = []
with open(sys.argv[1], newline="") as csvfile:
    spamreader = csv.DictReader(csvfile, delimiter=",", quotechar="|")
    data = list(spamreader)

for col in (0, 1):
    selected = [row for row in data if int(row["row"]) == 2 and int(row["col"]) == col]
    epochs = [int(row["epoch"]) for row in selected]
    weights = [float(row["weight"]) for row in selected]
    plt.plot(epochs, weights, label=f"col {col}")

plt.xlabel("Epoch")
plt.ylabel("Weight")
plt.axhline(0, color="gray", linewidth=0.5)
plt.legend()
plt.show()
