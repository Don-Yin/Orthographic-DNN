from pathlib import Path

import pandas

from read_corpus import read_corpus


class ConvertData2014:
    def __init__(self):
        self.path_output_targets = Path("assets", "2014-targets.txt")
        self.path_output_foils = Path("assets", "2014-foils.txt")
        self.read_data()
        self.save()

    def read_data(self):
        data = pandas.read_excel(Path("assets", "adelman.xlsx"), sheet_name=2)
        targets, foils = data.loc[:, "Target"][:420], data.loc[:, "Target"][420:]
        self.targets, self.foils = [i.lower() for i in targets], [i.lower() for i in foils]

    def save(self):
        with open(self.path_output_targets, "w") as writer:
            writer.write("\n".join(self.targets))

        with open(self.path_output_foils, "w") as writer:
            writer.write("\n".join(self.foils))


class CompleteCorpus:
    def __init__(self):
        self.path_corpus_2014 = Path("assets", "2014-targets.txt")
        self.path_corpus_google = Path("assets", "google-20000-english-no-swears.txt")
        self.path_output_combined = Path("assets", "1000-corpus.txt")
        self.read_data()
        self.filter_google_corpus()
        self.combine()
        self.save()

    def read_data(self):
        self.corpus_2014 = read_corpus(self.path_corpus_2014)
        self.corpus_google = read_corpus(self.path_corpus_google)

    def filter_google_corpus(self):
        self.corpus_google: list[str] = [i for i in self.corpus_google if len(i) in (3, 4, 5, 7, 8)]
        self.corpus_google: list[str] = [i for i in self.corpus_google if i not in self.corpus_2014]
        self.corpus_google: list[str] = [i for i in self.corpus_google if len(set(i)) == len(i)]
        self.corpus_google_l3 = [i for i in self.corpus_google if len(i) == 3][:116]
        self.corpus_google_l4 = [i for i in self.corpus_google if len(i) == 4][:116]
        self.corpus_google_l5 = [i for i in self.corpus_google if len(i) == 5][:116]
        self.corpus_google_l7 = [i for i in self.corpus_google if len(i) == 7][:116]
        self.corpus_google_l8 = [i for i in self.corpus_google if len(i) == 8][:116]
        self.corpus_google = (
            self.corpus_google_l3
            + self.corpus_google_l4
            + self.corpus_google_l5
            + self.corpus_google_l7
            + self.corpus_google_l8
        )

    def combine(self):
        self.corpus_combined = self.corpus_2014 + self.corpus_google

    def save(self):
        with open(self.path_output_combined, "w") as writer:
            writer.write("\n".join(self.corpus_combined))


if __name__ == "__main__":
    CompleteCorpus()
