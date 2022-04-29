import json
from pathlib import Path

import numpy
import pandas
from utils.plot.ridge_plot import RidgePlot


class DescribeSolarData:
    def __init__(self, duration: str):
        self.analysis_settings = json.load(open(Path("analysis_settings.json"), "r"))
        self.error_dict = json.load(open(Path("assets", "label_error.json"), "r"))
        self.sorter = json.load(open(Path("assets", "sorter_human_data.json"), "r"))
        self.duration = duration
        self.read_data()
        self.main()

    def read_data(self):
        data = open(Path("assets", "solar_model", f"SOLAR_result_target_duration_{self.duration}.txt"), "r").read()
        initial_lines = data.split("\n")[:2]
        line_1 = [i for i in initial_lines[0].split("	") if i]
        line_2 = [i for i in initial_lines[1].split("	") if i not in ["", "Prime", "Target"]]
        for i in range(len(line_2)):
            if line_2[i] in self.error_dict.keys():
                line_2[i] = self.error_dict[line_2[i]]

        self.data = pandas.DataFrame({"prime_type": line_2, "Predicted RT": line_1})
        self.data.to_csv(
            Path("results", self.analysis_settings["result"], "solar_model", f"solar_model_target_duration_{self.duration}.csv"),
            index=False,
        )

    def main(self):
        dummy_dataframe = self.data.copy()
        dummy_dataframe.set_index("prime_type", inplace=True)
        RTs = [-float(i) for i in dummy_dataframe['Predicted RT'].values]
        dummy_dataframe["Predicted RT"] = RTs

        dummy_dataframe = dummy_dataframe.reindex(self.sorter)
        dummy_dataframe["prime_type"] = dummy_dataframe.index

        try:
            RidgePlot(
                dataframe=dummy_dataframe,
                colname_group="prime_type",
                colname_variable="Predicted RT",
                name_save="ridge_solar_model.png",
                hue_level=len(self.sorter),
                means=None,
                path_save=Path("results", self.analysis_settings["result"], "solar_model"),
                draw_density=False,
                whether_double_extreme_lines=False,
            )
        except numpy.linalg.LinAlgError:
            pass


if __name__ == "__main__":
    DescribeSolarData()
