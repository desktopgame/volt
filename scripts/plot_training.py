#!/usr/bin/env -S uv run python
import csv
import sys

import matplotlib.pyplot as plt


def usage():
    print("Usage: plot_training.py {csv_file}")


if len(sys.argv) <= 1:
    usage()
    exit()

data: list[dict[str, str]] = []
with open(sys.argv[1], newline="") as csvfile:
    spamreader = csv.DictReader(csvfile, delimiter=",", quotechar="|")
    data = list(spamreader)

x = [int(row["epoch"]) for row in data if int(row["input_id"]) == 0]
y = [float(row["target_prob"]) for row in data if int(row["input_id"]) == 0]

plt.xlabel("Epoch")
plt.ylabel("Target probability")
plt.ylim(0, 1)
plt.plot(x, y)
plt.show()
