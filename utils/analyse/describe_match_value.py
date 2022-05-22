import json
from pathlib import Path

import numpy
import pandas
from utils.plot.ridge_plot import RidgePlot


class DescribeMatchValue:
    def __init__(self):
        self.analysis_settings = json.load(open(Path("analysis_settings.json"), "r"))
        coding_schemes = json.load(open(Path("assets", "coding_schemes.json"), "r"))
        self.sorter = json.load(open(Path("assets", "sorter_human_data.json"), "r"))
        for model in coding_schemes:
            self.which_conceptual_model = model
            self.read_data()
            self.select_col()
            self.save_descriptive_stats()
            self.plot_ridge()

    def read_data(self):
        self.data = pandas.read_csv(Path("assets", "match_calculator_result.csv"))

    def select_col(self):
        self.data = self.data[[self.which_conceptual_model, "prime_type"]]

    def save_descriptive_stats(self):
        """_summary_
            create and save descriptive stats of reaction time in human data and save in results folder
        """
        self.descriptive_stats = self.data.groupby("prime_type").describe()
        self.descriptive_stats = self.descriptive_stats.reindex(self.sorter)
        self.descriptive_stats.to_csv(Path("results", self.analysis_settings["result_folder"], "match_value", f"{self.which_conceptual_model}.csv"))

    def plot_ridge(self):
        dummy_data = self.data.copy()

        try:
            RidgePlot(
                dataframe=dummy_data,
                colname_group="prime_type",
                colname_variable=self.which_conceptual_model,
                name_save=f"{self.which_conceptual_model}.png",
                hue_level=len(self.sorter),
                means=list(self.descriptive_stats[(self.which_conceptual_model, "mean")]),
                path_save=Path("results", self.analysis_settings["result_folder"], "match_value"),
                draw_density=False,
                whether_double_extreme_lines=False,
            )
        except numpy.linalg.LinAlgError:
            pass

    def get_columns_summary(self, colname: str):
        return self.data_trial_by_trial[colname].describe()

    def get_frequency_table(self, colname: str):
        return getattr(self.data_trial_by_trial, colname).value_counts()

    def get_unique_elements(self, colname: str):
        return getattr(self.data_trial_by_trial, colname).unique()

    def report_num_trials(self):
        return self.data_trial_by_trial.shape[0]


if __name__ == "__main__":
    DescribeMatchValue()

