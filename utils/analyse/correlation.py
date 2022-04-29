from scipy.stats import kendalltau, spearmanr


def correlations(vectors: tuple):
    tau = kendalltau(x=vectors[0], y=vectors[1], variant="c", alternative="greater")
    rho = spearmanr(a=vectors[0], b=vectors[1], axis=0, alternative="greater")
    return {"tau": tau, "rho": rho}


if __name__ == "__main__":

    x = [
        1,
        2,
        3,
        4,
        5,
    ]

    y = [100, 300, 200, 400, 500]

    v = (x, y)

    print(correlations(v))
