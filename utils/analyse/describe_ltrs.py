import json
from pathlib import Path

import numpy
import pandas
from utils.plot.ridge_plot import RidgePlot


class DescribeLtrsData:
    def __init__(self):
        self.analysis_settings = json.load(open(Path("analysis_settings.json"), "r"))
        self.error_dict = json.load(open(Path("assets", "label_error.json"), "r"))
        self.sorter = json.load(open(Path("assets", "sorter_human_data.json"), "r"))
        self.prime_data = json.load(open(Path("assets", "2014-prime-data-words-only.json"), "r"))
        self.data = open(Path("assets", "ltrs", "output.txt"), "r").read()
        self.get_means()
        self.plot_ridge()

    def get_means(self):
        lines = self.data.split("\n")
        lines = [i.split("\t") for i in lines]
        for i in range(len(lines)):
            lines[i] += [self.retrieve_prime_type(lines[i][0], lines[i][1])]
        lines = [{"Priming": int(float(i[4])), "Primes": i[5]} for i in lines]

        for i in range(len(lines)):
            if lines[i]["Primes"] in self.error_dict.keys():
                lines[i]["Primes"] = self.error_dict[lines[i]["Primes"]]

        self.lines_df = pandas.DataFrame(lines)

        self.descriptive_stats = self.lines_df.groupby("Primes").describe()
        self.descriptive_stats = self.descriptive_stats.reindex(self.sorter)
        self.descriptive_stats.to_csv(Path("results", self.analysis_settings["result_folder"], "ltrs", "ltrs.csv"))

    def retrieve_prime_type(self, target, prime):
        D = [i for i in self.prime_data if i["ID"] == target][0]
        return list(D.keys())[list(D.values()).index(prime)]

    def plot_ridge(self):
        dummy_dataframe = self.lines_df.copy()
        dummy_dataframe["Primes"] = dummy_dataframe["Primes"].astype("category")
        dummy_dataframe["Primes"] = dummy_dataframe["Primes"].cat.set_categories(self.sorter)

        # RTs = [-float(i) for i in dummy_dataframe["Priming"].values]
        # dummy_dataframe["Priming"] = RTs

        try:
            RidgePlot(
                dataframe=dummy_dataframe,
                colname_group="Primes",
                colname_variable="Priming",
                name_save="ridge_ltrs.png",
                hue_level=len(self.sorter),
                means=None,
                path_save=Path("results", self.analysis_settings["result_folder"], "ltrs"),
                draw_density=True,
                whether_double_extreme_lines=False,
                x_lim=None,
            )
        except numpy.linalg.LinAlgError:
            pass


# example format
#        Cosine Similarity      primes
# 23520           1.000000          ID
# 23521           0.674831        TL12
# 23522           0.849880        TL-M
# 23523           0.781005        TL56
# 23524           0.634362  NATL-24/35
# ...                  ...         ...
# 35275           0.480254       IL-1I
# 35276           0.624140       IL-1F
# 35277           0.309900      IL-2MR
# 35278           0.113068     ALD-ARB
# 35279           0.071164      ALD-PW


if __name__ == "__main__":
    DescribeLtrsData()
