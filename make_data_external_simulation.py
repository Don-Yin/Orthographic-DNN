import json
from collections import Counter
from pathlib import Path


class MakeMatchCalculatorData:
    """Output a txt file to feed to the match calculator"""

    def __init__(self):
        """Note: remove the strings that share 0 letters"""
        self.read_prime_data()
        self.main()

    def read_prime_data(self):
        self.prime_data = json.load(open(Path("assets", "2014-prime-data-words-only.json"), "r"))

    def test_shared_letters(self, strings: tuple[str]):
        counters = [Counter(s) for s in strings]
        return sum((counters[0] & counters[1]).values())

    def main(self):
        content = []
        for D in self.prime_data:
            content.append(self.make_content_dict(D))
        open(Path("assets", "prime_data_for_match_calculator.txt"), "w").write("".join(content))

    def make_content_dict(self, D: dict):
        dummy_dict = D
        dummy_dict.pop("target", None)
        content = []
        for key in dummy_dict.keys():
            if self.test_shared_letters((dummy_dict["ID"], dummy_dict[key])) > 0:
                content.append(self.make_content_line((dummy_dict["ID"], dummy_dict[key])))
        return "".join(content)

    def make_content_line(self, words: tuple[str]):
        return "	".join([words[0], words[1]]) + "\n"


class MakeLtrsData:
    def __init__(self):
        self.read_prime_data()
        self.main()

    def read_prime_data(self):
        self.prime_data = json.load(open(Path("assets", "2014-prime-data-words-only.json"), "r"))

    def main(self):
        content = "\n".join([self.make_content_dict(i) for i in self.prime_data])
        open(Path("assets", "prime_data_for_ltrs.txt"), "w").write(content)

    def make_content_dict(self, D: dict):
        dummy_dict = D.copy()
        dummy_dict.pop("target", None)
        lines = "\n".join([self.make_line(D=dummy_dict, condition=i) for i in dummy_dict.keys()])
        return lines

    def make_line(self, D, condition):
        return f"{D['ID']} {D[condition]} {D['ALD-ARB']} 50"


if __name__ == "__main__":
    MakeLtrsData()
