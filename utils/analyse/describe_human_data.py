import json
from pathlib import Path

import pandas
from utils.plot.ridge_plot import RidgePlot


class DescribeHumanData:
    def __init__(self):
        self.analysis_settings = json.load(open(Path("analysis_settings.json"), "r"))
        self.read_data(source="csv")
        self.filter_row()
        self.select_col()
        self.save_descriptive_stats()
        self.plot_ridge()

    def read_data(self, source: str):
        if source == "xlsx":
            self.data_trial_by_trial = pandas.read_excel(Path("assets", "adelman.xlsx"), sheet_name=0)
        elif source == "csv":
            self.data_trial_by_trial = pandas.read_csv(Path("assets", "adelman.csv"), dtype={"subID": str, "Use": bool})

    def filter_row(self):
        """_summary_
            removes trails by conditions
        - Problems with timing responses with the equipment at the University of Nebraska, Omaha, led to those participants being excluded from further analysis.
        - Participants whose accuracy at primed lexical decision was below 75% were replaced. For this purpose, failure to respond before a 2,000-ms timeout, described in the Procedure section (0.47% of all trials), was counted as an error.
        - remove non-word trials
        - remove erroneous trails
        """
        self.data_trial_by_trial.drop(
            self.data_trial_by_trial[
                (self.data_trial_by_trial.lab == "Nebraska")
                | (self.data_trial_by_trial.RT > 2000)
                | (self.data_trial_by_trial.RT < 0)
                | (self.data_trial_by_trial.lexStatus == 0)
                | (self.data_trial_by_trial.Use != True)
            ].index,
            inplace=True,
        )

    def select_col(self):
        self.data_trial_by_trial = self.data_trial_by_trial[["RT", "cond.label"]]

    def save_descriptive_stats(self):
        """_summary_
        create and save descriptive stats of reaction time in human data and save in results folder
        """
        self.descriptive_stats = (
            self.data_trial_by_trial.groupby("cond.label").describe().sort_values(by=[("RT", "mean")], ascending=True)
        )

        self.descriptive_stats[("RT", "priming_arb")] = (
            max(self.descriptive_stats[("RT", "mean")]) - self.descriptive_stats[("RT", "mean")]
        )
        self.descriptive_stats.to_csv(
            Path("results", self.analysis_settings["result_folder"], "human", "descriptive_stats_human_data.csv")
        )

    def plot_ridge(self):
        """_summary_
            making a ridge plot in results folder
        put this in figure caption: e.g, removed for visual purposes;
        limited to data within *n standard deviations ()
        """
        sorter = list(self.descriptive_stats.index)
        json.dump(sorter, open(Path("assets", "sorter_human_data.json"), "w"))

        dummy_data = self.data_trial_by_trial.copy()

        # set range with sd
        overall_RT_summary = self.get_columns_summary("RT")

        plot_range = (
            overall_RT_summary["mean"] - (2 * overall_RT_summary["std"]),
            overall_RT_summary["mean"] + (2 * overall_RT_summary["std"]),
        )

        dummy_data.drop(
            dummy_data[(dummy_data.RT < plot_range[0]) | (dummy_data.RT > plot_range[1])].index,
            inplace=True,
        )

        dummy_data["cond.label"] = dummy_data["cond.label"].astype("category")
        dummy_data["cond.label"] = dummy_data["cond.label"].cat.set_categories(sorter)
        dummy_data.sort_values(by="cond.label", inplace=True)

        # ---

        RidgePlot(
            dataframe=dummy_data,
            colname_group="cond.label",
            colname_variable="RT",
            name_save="human_data_ridge_plot.png",
            hue_level=len(sorter),
            means=list(self.descriptive_stats[("RT", "mean")]),
            path_save=Path("results", self.analysis_settings["result_folder"], "human"),
        )

        dummy_data.rename({"RT": self.analysis_settings["label_human_data_ridge_plot_x_axis"]}, axis=1, inplace=True)

        RidgePlot(
            dataframe=dummy_data,
            colname_group="cond.label",
            colname_variable=self.analysis_settings["label_human_data_ridge_plot_x_axis"],
            name_save="human_data_ridge_plot_priming_arb.png",
            hue_level=len(sorter),
            means=list(self.descriptive_stats[("RT", "priming_arb")]),
            path_save=Path("results", self.analysis_settings["result_folder"], "human"),
            draw_density=False,
        )

    def get_columns_summary(self, colname: str):
        return self.data_trial_by_trial[colname].describe()

    def get_frequency_table(self, colname: str):
        return getattr(self.data_trial_by_trial, colname).value_counts()

    def get_unique_elements(self, colname: str):
        return getattr(self.data_trial_by_trial, colname).unique()

    def report_num_trials(self):
        return self.data_trial_by_trial.shape[0]


if __name__ == "__main__":
    DescribeHumanData()
