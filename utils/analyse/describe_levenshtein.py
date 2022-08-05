import json
from pathlib import Path

import numpy
import pandas
from utils.plot.ridge_plot import RidgePlot


class DescribeLevenshtein:
    def __init__(self, target_file: str = "levenshtein"):
        self.analysis_settings = json.load(open(Path("analysis_settings.json"), "r"))
        self.sorter = json.load(open(Path("assets", "sorter_human_data.json"), "r"))
        self.target_file = target_file

        self.read_error_dict()
        self.read_data()
        self.to_dataframe()
        self.plot_ridge()

    def read_error_dict(self):
        self.error_dict = json.load(open(Path("assets", "label_error.json"), "r"))

    def read_data(self):
        self.data_main = json.load(
            open(Path("assets", "levenshtein", f"2014-prime-data-words-only-{self.target_file}.json"), "r")
        )
        for key in self.error_dict.keys():
            if key in self.data_main.keys():
                self.data_main[self.error_dict[key]] = self.data_main.pop(key)

    def to_dataframe(self):
        self.data_main = pandas.DataFrame(self.data_main, index=[0]).transpose()
        self.data_main = self.data_main.reindex(self.sorter)
        self.data_main.columns = ["LD"]
        self.data_main.to_csv(Path("results", self.analysis_settings["result_folder"], "levenshtein", f"{self.target_file}.csv"))

    def plot_ridge(self):
        dummy_dataframe = self.data_main.copy()
        dummy_dataframe["prime_type"] = dummy_dataframe.index

        means = list(self.data_main["LD"])
        means = [-i for i in means]

        print(means)

        try:
            RidgePlot(
                dataframe=dummy_dataframe,
                colname_group="prime_type",
                colname_variable="LD",
                name_save=f"{self.target_file}.png",
                hue_level=len(self.sorter),
                means=means,
                path_save=Path("results", self.analysis_settings["result_folder"], "levenshtein"),
                draw_density=False,
                whether_double_extreme_lines=False,
            )
        except numpy.linalg.LinAlgError:
            pass


if __name__ == "__main__":
    DescribeLevenshtein()
