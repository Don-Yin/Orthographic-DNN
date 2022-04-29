import json
import os
from pathlib import Path

import pandas

from utils.analyse.correlation import correlations
from utils.analyse.describe_correlation_layer_wise import DescribeCosineSimilarityLayerWise
from utils.analyse.describe_human_data import DescribeHumanData
from utils.analyse.describe_image_raw_similarity import DescribeImageRawSimilarity
from utils.analyse.describe_levenshtein import DescribeLevenshtein
from utils.analyse.describe_match_value import DescribeMatchValue
from utils.analyse.describe_model_data import DescribeModelData
from utils.analyse.describe_solar_data import DescribeSolarData
from utils.analyse.make_image_raw_cosine_similarity_data import MakeImageRawCosineSimilarityData
from utils.analyse.make_match_calculator_data import ProcessMatchCalculatorData
from utils.plot.correlation_matrix import CorrelationMatrix


class Analyse:
    def __init__(self):
        self.analysis_settings = json.load(open(Path("analysis_settings.json"), "r"))
        self.selected_model_labels = json.load(open(Path("assets", "selected_model_labels.json"), "r"))
        self.correlation_method = self.analysis_settings["correlation_method"]  # kendall/spearman/pearson
        self.solar_duration = self.analysis_settings["solar_duration"]
        self.force_process = self.analysis_settings["force_process"]

        self.ensure_folders()

        self.path_human_data = Path("results", self.analysis_settings["result"], "human", "descriptive_stats_human_data.csv")
        self.path_model_data = Path("results", self.analysis_settings["result"], "model", "alexnet.csv")
        self.path_solar_model_data = Path(
            "results",
            self.analysis_settings["result"],
            "solar_model",
            f"solar_model_target_duration_{self.solar_duration}.csv",
        )
        self.path_levenshtein_data = Path("results", self.analysis_settings["result"], "levenshtein", "levenshtein.csv")
        self.path_old20_data = Path("results", self.analysis_settings["result"], "levenshtein", "old20.csv")
        self.path_output_correlation_matrix_main = Path(
            "results", self.analysis_settings["result"], "correlation_matrix_main.png"
        )
        self.path_output_correlation_matrix_model_score = Path(
            "results", self.analysis_settings["result"], "correlation_matrix_model_score.png"
        )
        self.path_image_raw_cosine_similarity_data = Path(
            "assets", "image_raw_cosine_similarity", self.analysis_settings["image_raw_similarity_name"]
        )
        self.path_processed_image_raw_cosine_similarity_data = Path(
            "results", self.analysis_settings["result"], "image_raw_cosine_similarity", "image_raw_cosine_similarity.csv"
        )
        self.path_cosine_similarity_layer_wise = Path(
            "results", self.analysis_settings["result"], "cosine_similarity_layer_wise", "cosine_similarity_layer_wise.csv"
        )
        self.path_descriptive_stats_main = Path("results", self.analysis_settings["result"], "descriptive_stats_main.csv")

        self.conceptual_model_names = json.load(open(Path("assets", "conceptual_models.json"), "r"))
        self.neural_networks_names = json.load(open(Path("assets", "selected_models.json"), "r"))

        self.create_human_data()

        self.create_model_data()
        self.create_solar_model_data()
        self.create_levenshtein_data()
        self.create_image_raw_cosine_similarity_data()
        self.create_cosine_similarity_layer_wise_data()

        self.read_descriptive_stats_human()
        self.join_model_data()
        self.join_image_raw_cosine_similarity()
        self.join_levenshtein()
        self.join_solar_data()

        # self.create_old20_data()
        # self.join_old20()

        # self.path_match_value_data = Path("results", self.analysis_settings["result"], "match_value", "Absolute.csv")
        # self.create_match_value_data()
        # self.join_descriptive_stats_match_value()

        self.create_cosine_similarity_layer_wise_data()

        self.save_descriptive_stats()
        self.draw_correlation_matrix()
        self.draw_correlation_matrix_model_scores()

    def ensure_folders(self):
        if (
            not os.path.exists(
                Path(
                    "results",
                    self.analysis_settings["result"],
                )
            )
            or self.force_process
        ):
            self.ensure_one_folder(
                Path(
                    "results",
                    self.analysis_settings["result"],
                )
            )
            self.ensure_one_folder(Path("results", self.analysis_settings["result"], "human"))
            # self.ensure_one_folder(Path("results", self.analysis_settings["result"], "match_value"))
            self.ensure_one_folder(Path("results", self.analysis_settings["result"], "model"))
            self.ensure_one_folder(Path("results", self.analysis_settings["result"], "model_architecture"))
            self.ensure_one_folder(Path("results", self.analysis_settings["result"], "solar_model"))
            self.ensure_one_folder(Path("results", self.analysis_settings["result"], "levenshtein"))
            self.ensure_one_folder(Path("results", self.analysis_settings["result"], "image_raw_cosine_similarity"))
            self.ensure_one_folder(Path("results", self.analysis_settings["result"], "correlation_layer_wise"))

    def ensure_one_folder(self, path):
        if not os.path.exists(path):
            os.mkdir(path)

    def create_human_data(self):
        if (not os.path.exists(self.path_human_data)) or self.force_process:
            DescribeHumanData()

    def create_match_value_data(self):
        if (not os.path.exists(self.path_match_value_data)) or self.force_process:
            ProcessMatchCalculatorData()
            DescribeMatchValue()

    def create_model_data(self):
        if (not os.path.exists(self.path_model_data)) or self.force_process:
            DescribeModelData()

    def create_solar_model_data(self):
        if (not os.path.exists(self.path_solar_model_data)) or self.force_process:
            DescribeSolarData(duration=self.solar_duration)

    def create_levenshtein_data(self):
        if (not os.path.exists(self.path_levenshtein_data)) or self.force_process:
            DescribeLevenshtein(target_file="levenshtein")

    def create_old20_data(self):
        if (not os.path.exists(self.path_old20_data)) or self.force_process:
            DescribeLevenshtein(target_file="old20")

    def create_image_raw_cosine_similarity_data(self):
        if (not os.path.exists(self.path_image_raw_cosine_similarity_data)) or self.force_process:
            MakeImageRawCosineSimilarityData()
            DescribeImageRawSimilarity()

    def create_cosine_similarity_layer_wise_data(self):
        if (not os.path.exists(self.path_cosine_similarity_layer_wise)) or self.force_process:
            DescribeCosineSimilarityLayerWise(to_which="human")
            DescribeCosineSimilarityLayerWise(to_which="raw_image")

    def read_descriptive_stats_human(self):
        self.data_descriptive_main = self._read_descriptive_csv(
            drop_row="cond.label", sub_folder="human", file_name="descriptive_stats_human_data.csv"
        )
        self.data_descriptive_main = self.data_descriptive_main[["priming_arb"]]

    def join_descriptive_stats_match_value(self):
        for name in self.conceptual_model_names:
            dataframe = self._read_descriptive_csv(drop_row="prime_type", sub_folder="match_value", file_name=f"{name}.csv")
            dataframe.rename({"mean": name}, axis=1, inplace=True)
            dataframe = dataframe[[name]]
            self.data_descriptive_main = self.data_descriptive_main.join(dataframe)

    def join_model_data(self):
        for name in self.neural_networks_names:
            descriptive_stats = self._read_descriptive_csv(drop_row="primes", sub_folder="model", file_name=f"{name}.csv")
            descriptive_stats = descriptive_stats[["mean"]]
            descriptive_stats.rename({"mean": name}, axis=1, inplace=True)
            self.data_descriptive_main = self.data_descriptive_main.join(descriptive_stats)

    def join_solar_data(self):
        dataframe = pandas.read_csv(self.path_solar_model_data)
        dataframe.rename({"Predicted RT": "SCM"}, axis=1, inplace=True)
        dataframe.set_index("prime_type", inplace=True)
        self.data_descriptive_main = self.data_descriptive_main.join(-dataframe)

    def join_levenshtein(self):
        dataframe = pandas.read_csv(self.path_levenshtein_data, index_col=[0])
        dataframe = dataframe[["mean"]]
        dataframe.rename({"mean": "LD"}, axis=1, inplace=True)
        self.data_descriptive_main = self.data_descriptive_main.join(-dataframe)

    def join_old20(self):
        dataframe = pandas.read_csv(self.path_old20_data, index_col=[0])
        dataframe = dataframe[["mean"]]
        dataframe.rename({"mean": "OLD20"}, axis=1, inplace=True)
        self.data_descriptive_main = self.data_descriptive_main.join(-dataframe)

    def join_image_raw_cosine_similarity(self):
        dataframe = self._read_descriptive_csv(
            drop_row="primes", sub_folder="image_raw_cosine_similarity", file_name="image_raw_cosine_similarity.csv"
        )
        dataframe = dataframe[["mean"]]
        dataframe.rename({"mean": "pCS"}, axis=1, inplace=True)
        self.data_descriptive_main = self.data_descriptive_main.join(dataframe)

    def save_descriptive_stats(self):
        self.data_descriptive_main.to_csv(self.path_descriptive_stats_main)

    def draw_correlation_matrix(self):
        dummy_dataframe = self.data_descriptive_main.copy()
        dummy_dataframe.rename(
            {"priming_arb": self.analysis_settings["label_human_priming_data_in_main_matrix"]}, axis=1, inplace=True
        )
        dummy_dataframe.rename(self.selected_model_labels, axis=1, inplace=True)

        CorrelationMatrix(
            dataframe=dummy_dataframe,
            method=self.correlation_method,
            path_save=self.path_output_correlation_matrix_main,
            figure_size=(26, 16),
            adjust_button=0.2,
            adjust_left=0.4,
            which_plot="main",
            human_data_label=self.analysis_settings["label_human_priming_data_in_main_matrix"],
        )

        for name in ["Convolutional DNNs"]:
            CorrelationMatrix(
                dataframe=dummy_dataframe,
                method=self.correlation_method,
                path_save=self.path_output_correlation_matrix_main,
                figure_size=(5, 16),
                adjust_button=0.2,
                adjust_left=0.4,
                which_plot=["side", name],
                human_data_label=self.analysis_settings["label_human_priming_data_in_main_matrix"],
            )

    def draw_correlation_matrix_model_scores(self):
        self.model_scores = pandas.read_csv(Path("assets", "model_scores.csv"), index_col=0)
        self.model_correlations = pandas.read_csv(
            Path("results", self.analysis_settings["result"], "corr_models.csv"), index_col=0
        )
        self.model_scores = self.model_scores.join(self.model_correlations, how="left")
        dummy_dataframe = self.model_scores.copy()

        dummy_dataframe.rename(
            {
                self.analysis_settings["label_human_priming_data_in_main_matrix"]: self.analysis_settings[
                    "label_human_priming_data_in_model_score_matrix"
                ]
            },
            axis=1,
            inplace=True,
        )

        CorrelationMatrix(
            dataframe=dummy_dataframe,
            method=self.correlation_method,
            path_save=self.path_output_correlation_matrix_model_score,
            figure_size=(26, 16),
            adjust_button=0.2,
            adjust_left=0.4,
            which_plot="model_score",
            human_data_label=self.analysis_settings["label_human_priming_data_in_model_score_matrix"],
        )

    def _read_descriptive_csv(self, drop_row: str, sub_folder: str = "", file_name: str = ""):
        path = (
            Path(
                "results",
                self.analysis_settings["result"],
            )
            / sub_folder
            / file_name
        )
        data = pandas.read_csv(path, index_col=[0], skiprows=1)
        data.drop(drop_row, inplace=True)
        return data

    def _report_match_value_correlation(self):
        for name in self.conceptual_model_names:
            data = self._read_descriptive_csv(drop_row="prime_type", sub_folder="match_value", file_name=f"{name}.csv")
            print("".join([name, ":", correlations((data["mean"], self.human_data_descriptive["priming_arb"]))]), "\n")


if __name__ == "__main__":
    Analyse()
