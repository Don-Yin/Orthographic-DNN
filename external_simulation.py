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


if __name__ == "__main__":
    MakeMatchCalculatorData()
