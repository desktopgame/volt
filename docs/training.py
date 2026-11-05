import copy
import csv
import json
import math
import subprocess

# ~/.emacs.d/playground
from helper import dispmat, dispvec, matvec, pprint

vocab: list[str] = []
embeddings: list[list[float]] = []


def softmax(x: list[int | float]) -> list[int | float]:
    if len(x) == 0:
        raise Exception("x is empty")
    max = x[0]
    sum = 0
    for e in x:
        if e > max:
            max = e
    for e in x:
        sum += math.exp(e - max)
    result = []
    for e in x:
        result.append(math.exp(e - max) / sum)
    return result


def prepare(m, token_id):
    global vocab
    global embeddings
    score = matvec(m, embeddings[token_id])
    prob = softmax(score)
    return (score, prob)


def dispcalc(m, score, prob, token_id, other, succ):
    global vocab
    global embeddings
    if True:
        i = other
        # for i in range(0, 4):
        target = 0
        if succ:
            target = 1
        row = m[i]
        learning_rate = 0.1
        error = prob[i] - target
        for j in range(0, 2):
            row[j] -= learning_rate * error * embeddings[token_id][j]
    return m


def generate(m):
    global vocab
    global embeddings
    print("train report")
    print("<details open>")
    print("<summary>Configuration</summary>")
    print("<table>")
    print(f"<tr><th>名前</th><th>X</th><th>Y</th>")
    for i in range(0, len(vocab)):
        print(
            f"<tr><td>{vocab[i]}</td><td>{embeddings[i][0]}</td><td>{embeddings[i][1]}</td></tr>"
        )
    print("</table>")
    print("</details>")
    print()

    for i in range(0, 10):
        if i == 0 or i == 9:
            print("<details open>")
        else:
            print("<details>")
        print(f"<summary>Epoch {i + 1}</summary>")
        score, prob = prepare(m, 0)
        pprint(f"output\\_weight={dispmat(m)}")

        print("<details open>")
        print("<summary>token[0]</summary>")
        pprint(f"score={dispvec(score)}, prob={dispvec(prob)}")
        nm = dispcalc(copy.deepcopy(m), score, prob, 0, 0, False)
        pprint(dispmat(nm, m))
        m = nm

        nm = dispcalc(copy.deepcopy(m), score, prob, 0, 1, False)
        pprint(dispmat(nm, m))
        m = nm

        nm = dispcalc(copy.deepcopy(m), score, prob, 0, 2, True)
        pprint(dispmat(nm, m))
        m = nm

        nm = dispcalc(copy.deepcopy(m), score, prob, 0, 3, False)
        pprint(dispmat(nm, m))
        m = nm

        pprint(f"output\\_weight={dispmat(m)}")
        print("</details>")

        print("<details open>")
        print("<summary>token[1]</summary>")
        score, prob = prepare(m, 1)
        pprint(f"score={dispvec(score)}, prob={dispvec(prob)}")
        nm = dispcalc(copy.deepcopy(m), score, prob, 1, 0, False)
        pprint(dispmat(nm, m))
        m = nm

        nm = dispcalc(copy.deepcopy(m), score, prob, 1, 1, False)
        pprint(dispmat(nm, m))
        m = nm

        nm = dispcalc(copy.deepcopy(m), score, prob, 1, 2, False)
        pprint(dispmat(nm, m))
        m = nm

        nm = dispcalc(copy.deepcopy(m), score, prob, 1, 3, True)
        pprint(dispmat(nm, m))
        m = nm

        pprint(f"output\\_weight={dispmat(m)}")
        print("</details>")
        print("</details>")

    print("<hr>")
    pprint(f"trained\\_weight={dispmat(m)}")


rows = 4
cols = 2


def main(profile_path: str, weights_path: str):
    global vocab
    global embeddings

    profile = json.load(open(profile_path, encoding="utf8"))
    vocab = profile["vocab"]
    embeddings = profile["embeddings"]

    data: list[dict[str, str]] = []
    with open(weights_path, newline="") as csvfile:
        spamreader = csv.DictReader(csvfile, delimiter=",", quotechar="|")
        data = list(spamreader)

    epoch = 0
    matrixes = []
    for i in range(0, len(data), rows * cols):
        m = []
        for j in range(0, rows):
            mm = []
            for k in range(0, cols):
                line = data[i + (j * cols) + k]
                mm.append(float(line["weight"]))
            m.append(mm)
        matrixes.append(m)

    generate([[0, 0], [0, 0], [0, 0], [0, 0]])
