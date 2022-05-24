import json
from pathlib import Path

import numpy
import pandas
from utils.plot.ridge_plot import RidgePlot


class DescribeIaDataFull:
    def __init__(self, duration: str):
        self.analysis_settings = json.load(open(Path("analysis_settings.json"), "r"))
        self.error_dict = json.load(open(Path("assets", "label_error.json"), "r"))
        self.sorter = json.load(open(Path("assets", "sorter_human_data.json"), "r"))
        self.duration = duration
        self.read_data()
        self.plot_ridge()

    def read_data(self):
        data = open(Path("assets", "ia", f"ia_target_duration_{self.duration}.txt"), "r").read()
        self.initial_lines = data.split("\n")[:2]
        self.initial_lines = [i.split("\t") for i in self.initial_lines]
        self.initial_lines = [[i for i in line if i] for line in self.initial_lines]

        for i in range(len(self.initial_lines[1])):
            if self.initial_lines[1][i] in self.error_dict.keys():
                self.initial_lines[1][i] = self.error_dict[self.initial_lines[1][i]]

        lines = [i.split("\t") for i in data.split("\n")[2:]]
        lines = [[i for i in line if i] for line in lines]
        lines = [i for i in lines if i]
        lines = [[int(i.strip("No")) if not i.isalpha() else i for i in line] for line in lines]
        for i in range(len(lines)):
            lines[i] = dict(zip(self.initial_lines[1], lines[i]))
            lines[i].pop("Prime", None)
            lines[i].pop("Target", None)

        content = []
        for line in lines:
            for key, value in line.items():
                content.append({"Primes": key, "Predicted RT": value})

        self.data = pandas.DataFrame(content)

    def plot_ridge(self):
        dummy_dataframe = self.data.copy()
        dummy_dataframe["Primes"] = dummy_dataframe["Primes"].astype("category")
        dummy_dataframe["Primes"] = dummy_dataframe["Primes"].cat.set_categories(self.sorter)

        RTs = [-float(i) for i in dummy_dataframe["Predicted RT"].values]
        dummy_dataframe["Predicted RT"] = RTs

        try:
            RidgePlot(
                dataframe=dummy_dataframe,
                colname_group="Primes",
                colname_variable="Predicted RT",
                name_save="ridge_ia_model.png",
                hue_level=len(self.sorter),
                means=None,
                path_save=Path("results", self.analysis_settings["result_folder"], "ia"),
                draw_density=True,
                whether_double_extreme_lines=False,
                x_lim=(-160, -80),
            )
        except numpy.linalg.LinAlgError:
            pass


class DescribeIaData:
    def __init__(self, duration: str):
        self.analysis_settings = json.load(open(Path("analysis_settings.json"), "r"))
        self.error_dict = json.load(open(Path("assets", "label_error.json"), "r"))
        self.sorter = json.load(open(Path("assets", "sorter_human_data.json"), "r"))
        self.duration = duration
        self.read_data()

    def read_data(self):
        data = open(Path("assets", "ia", f"ia_target_duration_{self.duration}.txt"), "r").read()
        initial_lines = data.split("\n")[:2]
        line_1 = [i for i in initial_lines[0].split("	") if i]
        line_2 = [i for i in initial_lines[1].split("	") if i not in ["", "Prime", "Target"]]
        for i in range(len(line_2)):
            if line_2[i] in self.error_dict.keys():
                line_2[i] = self.error_dict[line_2[i]]

        self.data = pandas.DataFrame({"prime_type": line_2, "Predicted RT": line_1})
        self.data.to_csv(
            Path("results", self.analysis_settings["result_folder"], "ia", f"ia_target_duration_{self.duration}.csv"),
            index=False,
        )


if __name__ == "__main__":
    DescribeIaData()
