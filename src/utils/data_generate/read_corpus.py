from pathlib import Path


def read_corpus(path: Path):
    corpus = open(path, "r").read()
    corpus: list[str] = [w for w in corpus.split("\n") if w != ""]
    corpus: list[str] = [w for w in corpus if len(w)]
    return corpus


if __name__ == "__main__":
    corpus = read_corpus(Path("assets", "corpus.txt"))
    print(len(corpus))
