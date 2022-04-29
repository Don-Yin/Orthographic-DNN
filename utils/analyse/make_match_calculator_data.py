import json
from collections import Counter
from pathlib import Path

import pandas


class ProcessMatchCalculatorData:
    def __init__(self):
        self.read_prime_data()
        self.save_calculator_result_in_csv()
        self.read_calculator_result_from_csv()
        self.append_match_value_unrelated_strings()
        self.append_prime_type()
        # self.reverse()
        self.save_final()

    def read_prime_data(self):
        self.prime_data = json.load(open(Path("assets", "2014-prime-data-words-only.json"), "r"))

    def save_calculator_result_in_csv(self):
        self.calculator_result = open(Path("assets", "match_calculator_result.txt"), "r").read()
        self.calculator_result = "\n".join(self.calculator_result.split("\n")[13:])
        self.calculator_result = self.calculator_result.replace("	", ",").replace(" ", "").replace(",\n", "\n")

        match_dict = {
            "Stim1": "ID",
            "Stim2": "prime",
            "Match1": "Absolute",
            "Match2": "Vowel-centric(L-R)",
            "Match3": "Vowel-centric(R-L)",
            "Match4": "Ends-first",
            "Match5": "SOLAR(Spatial_Coding)",
            "Match6": "Binary_Open_Bigram",
            "Match7": "Overlap_Open_Bigram(OOB)",
            "Match8": "SERIOL_2001_(Open_Bigram)",
        }

        for key in match_dict.keys():
            self.calculator_result = self.calculator_result.replace(key, match_dict[key])

        open(Path("assets", "match_calculator_result.csv"), "w").write(self.calculator_result)

    def read_calculator_result_from_csv(self):
        self.calculator_result_csv = pandas.read_csv(Path("assets", "match_calculator_result.csv"), index_col=False)

    def append_match_value_unrelated_strings(self):
        for D in self.prime_data:
            dummy_dict = D
            dummy_dict.pop("target", None)
            for key in dummy_dict.keys():
                if self.test_shared_letters((D["ID"], D[key])) == 0:
                    self.calculator_result_csv = self.calculator_result_csv.append(
                        {
                            "ID": D["ID"],
                            "prime": D[key],
                            "Absolute": 0,
                            "Vowel-centric(L-R)": 0,
                            "Vowel-centric(R-L)": 0,
                            "Ends-first": 0,
                            "SOLAR(Spatial_Coding)": 0,
                            "Binary_Open_Bigram": 0,
                            "Overlap_Open_Bigram(OOB)": 0,
                            "SERIOL_2001_(Open_Bigram)": 0,
                        },
                        ignore_index=True,
                    )

    def append_prime_type(self):
        self.calculator_result_csv["prime_type"] = self.calculator_result_csv.apply(
            lambda row: self.retrive_prime_type((row.ID, row.prime)), axis=1
        )

    def save_final(self):
        self.calculator_result_csv.to_csv(Path("assets", "match_calculator_result.csv"), index=False)

    def test_shared_letters(self, strings: tuple[str]):
        counters = [Counter(s) for s in strings]
        return sum((counters[0] & counters[1]).values())

    def retrive_prime_type(self, pair: tuple[str]):
        target_dict = [i for i in self.prime_data if i["ID"] == pair[0]][0]
        type = list(target_dict.keys())[list(target_dict.values()).index(pair[1])]
        error_dict = json.load(open(Path("assets", "label_error.json"), "r"))
        if type in error_dict.keys():
            type = error_dict[type]
        return type


class MakeMatchCalculatorData:
    """Output a txt file to feed to the match calculator
    """

    def __init__(self):
        """Note: remove the strings that share 0 letters
        """
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
