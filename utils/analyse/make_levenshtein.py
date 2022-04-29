import json
from pathlib import Path

from Levenshtein import distance as lev
from tqdm import tqdm

from utils.old20.old20 import old20, old_n


class MakeDataLevenshtein:
    def __init__(self):
        self.read_2014_prime_data()
        self.main()

    def read_2014_prime_data(self):
        self.data = json.load(open(Path("assets", "2014-prime-data-words-only.json"), "r"))
        for i in self.data:
            i.pop("target", None)

    def main(self):
        self.data = [self._transform_to_levenshtein(i) for i in self.data]
        json.dump(self.data, open(Path("assets", "levenshtein", "2014-prime-data-words-only-levenshtein.json"), "w"))

    def _transform_to_levenshtein(self, dict):
        for key in dict.keys():
            if key != "ID":
                dict[key] = lev(dict["ID"], dict[key])

        dict["ID"] = 0
        return dict


class MakeDataOLD20:
    def __init__(self):
        self.read_2014_prime_data()
        self.read_word_list()
        self.main()

    def read_2014_prime_data(self):
        self.data = json.load(open(Path("assets", "2014-prime-data-words-only.json"), "r"))
        for i in self.data:
            i.pop("target", None)

    def read_word_list(self):
        self.wordlist = open(Path("assets", "2014-targets.txt")).readlines()
        self.wordlist = [i.replace("\n", "") for i in self.wordlist]

    def main(self):
        self.data = [self._transform_to_old20(i) for i in tqdm(self.data)]
        json.dump(self.data, open(Path("assets", "2014-prime-data-words-only-old20.json"), "w"))

    def _transform_to_old20(self, dict):
        for key in dict.keys():
            dict[key] = float(old_n([dict[key]], self.wordlist, n=20))

        return dict


if __name__ == "__main__":
    MakeDataLevenshtein()
    MakeDataOLD20()
