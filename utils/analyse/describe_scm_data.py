import json
from pathlib import Path

import numpy
import pandas
from utils.plot.ridge_plot import RidgePlot


class DescribeScmDataFull:
    def __init__(self, duration: str):
        self.analysis_settings = json.load(open(Path("analysis_settings.json"), "r"))
        self.error_dict = json.load(open(Path("assets", "label_error.json"), "r"))
        self.sorter = json.load(open(Path("assets", "sorter_human_data.json"), "r"))
        self.duration = duration
        self.read_data()
        self.plot_ridge()

    def read_data(self):
        data = open(Path("assets", "scm", f"SOLAR_result_target_duration_{self.duration}.txt"), "r").read()
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

        # dummy_dataframe.set_index("Primes", inplace=True)
        RTs = [-float(i) for i in dummy_dataframe["Predicted RT"].values]
        dummy_dataframe["Predicted RT"] = RTs

        # dummy_dataframe = dummy_dataframe.reindex(self.sorter)
        # dummy_dataframe["Primes"] = dummy_dataframe.index

        try:
            RidgePlot(
                dataframe=dummy_dataframe,
                colname_group="Primes",
                colname_variable="Predicted RT",
                name_save=f"ridge_solar_model.png",
                hue_level=len(self.sorter),
                means=None,
                path_save=Path("results", self.analysis_settings["result_folder"], "scm"),
                draw_density=True,
                whether_double_extreme_lines=False,
                x_lim=(-125, -50),
            )
        except numpy.linalg.LinAlgError:
            pass


class DescribeScmData:
    def __init__(self, duration: str):
        self.analysis_settings = json.load(open(Path("analysis_settings.json"), "r"))
        self.error_dict = json.load(open(Path("assets", "label_error.json"), "r"))
        self.sorter = json.load(open(Path("assets", "sorter_human_data.json"), "r"))
        self.duration = duration
        self.read_data()

    def read_data(self):
        data = open(Path("assets", "scm", f"SOLAR_result_target_duration_{self.duration}.txt"), "r").read()
        initial_lines = data.split("\n")[:2]
        line_1 = [i for i in initial_lines[0].split("	") if i]
        line_2 = [i for i in initial_lines[1].split("	") if i not in ["", "Prime", "Target"]]
        for i in range(len(line_2)):
            if line_2[i] in self.error_dict.keys():
                line_2[i] = self.error_dict[line_2[i]]

        self.data = pandas.DataFrame({"prime_type": line_2, "Predicted RT": line_1})
        self.data.to_csv(
            Path(
                "results", self.analysis_settings["result_folder"], "scm", f"solar_model_target_duration_{self.duration}.csv"
            ),
            index=False,
        )


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
    DescribeScmData()
