import json
from pathlib import Path

import numpy
import pandas
from utils.plot.ridge_plot import RidgePlot


class DescribeImageRawSimilarity:
    def __init__(self):
        self.analysis_settings = json.load(open(Path("analysis_settings.json"), "r"))
        self.sorter = json.load(open(Path("assets", "sorter_human_data.json"), "r"))
        self.read_error_dict()
        self.read_data()
        self.dict_error_correction()
        self.to_dataframe()
        self.set_descriptive_data()
        self.plot_ridge()

    def read_error_dict(self):
        self.error_dict = json.load(open(Path("assets", "label_error.json"), "r"))

    def read_data(self):
        self.data_main = json.load(
            open(Path("assets", "image_raw_cosine_similarity", self.analysis_settings["image_raw_similarity_name"]), "r")
        )

    def dict_error_correction(self):
        # correct labeling error + translate primes from list to str
        for i in range(len(self.data_main)):
            if self.data_main[i]["primes"][1] in self.error_dict.keys():
                self.data_main[i]["primes"] = self.error_dict[self.data_main[i]["primes"][1]]
            else:
                self.data_main[i]["primes"] = self.data_main[i]["primes"][1]

    def to_dataframe(self):
        self.data_main = pandas.DataFrame(self.data_main)
        self.data_main = self.data_main[["cosine_similarity", "primes"]]
        self.data_main.rename(
            {"cosine_similarity": self.analysis_settings["label_model_ridge_plot_x_axis"]}, axis=1, inplace=True
        )
        
    def set_descriptive_data(self):
        self.descriptive_stats = self.data_main.groupby("primes").describe()
        self.descriptive_stats = self.descriptive_stats.reindex(self.sorter)
        self.descriptive_stats.to_csv(Path("results", self.analysis_settings["result_folder"], "image_raw_cosine_similarity", "image_raw_cosine_similarity.csv"))

    def plot_ridge(self):
        dummy_dataframe = self.data_main.copy()
        dummy_dataframe["primes"] = dummy_dataframe["primes"].astype("category")
        dummy_dataframe["primes"] = dummy_dataframe["primes"].cat.set_categories(self.sorter)
        dummy_dataframe.sort_values(by="primes", inplace=True)

        try:
            RidgePlot(
                dataframe=dummy_dataframe,
                colname_group="primes",
                colname_variable=self.analysis_settings["label_model_ridge_plot_x_axis"],
                name_save="image_raw_cosine_similarity.png",
                hue_level=len(self.sorter),
                means=None,
                path_save=Path("results", self.analysis_settings["result_folder"], "image_raw_cosine_similarity"),
                draw_density=True,
                whether_double_extreme_lines=False,
            )
        except numpy.linalg.LinAlgError:
            pass


if __name__ == "__main__":
    DescribeImageRawSimilarity()
