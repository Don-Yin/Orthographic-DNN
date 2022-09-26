import json
import os
from pathlib import Path

import pandas

from utils.analyse.correlation import correlations
from utils.analyse.describe_correlation_layer_wise import \
    DescribeCosineSimilarityLayerWise
from utils.analyse.describe_human_data import DescribeHumanData
from utils.analyse.describe_ia_data import DescribeIaData, DescribeIaDataFull
from utils.analyse.describe_image_raw_similarity import \
    DescribeImageRawSimilarity
from utils.analyse.describe_levenshtein import DescribeLevenshtein
from utils.analyse.describe_ltrs import DescribeLtrsData
from utils.analyse.describe_match_value import DescribeMatchValue
from utils.analyse.describe_model_data import DescribeModelData
from utils.analyse.describe_scm_data import (DescribeScmData,
                                             DescribeScmDataFull)
from utils.analyse.make_image_raw_cosine_similarity_data import \
    MakeImageRawCosineSimilarityData
from utils.analyse.make_match_calculator_data import ProcessMatchCalculatorData
from utils.plot.correlation_matrix import CorrelationMatrix


class Analyse:
    def __init__(self):
        self.analysis_settings = json.load(open(Path("analysis_settings.json"), "r"))
        self.dnn_model_labels = json.load(open(Path("assets", "dnn_model_labels.json"), "r"))
        self.conceptual_model_names = json.load(open(Path("assets", "coding_schemes.json"), "r"))

        self.correlation_method = self.analysis_settings["correlation_method"]  # kendall/spearman/pearson
        self.scm_duration = self.analysis_settings["scm_duration"]
        self.force_process = self.analysis_settings["force_process"]

        self.make_folders()

        # ----paths base----
        self.path_result_base = Path("results", self.analysis_settings["result_folder"])

        # ----paths folder----
        self.path_folder_human_data = self.path_result_base / "human"
        self.path_folder_model_data = self.path_result_base / "model"
        self.path_folder_match_value_data = self.path_result_base / "match_value"
        self.path_folder_correlation_layer_wise = self.path_result_base / "correlation_layer_wise"

        # ----paths objects----
        self.path_output_correlation_matrix_main = self.path_result_base / "correlation_matrix_main.png"
        self.path_output_correlation_matrix_model_score = self.path_result_base / "correlation_matrix_model_score.png"
        self.path_levenshtein_data = self.path_result_base / "levenshtein" / "levenshtein.csv"
        self.path_descriptive_stats_main = self.path_result_base / "descriptive_stats_main.csv"
        self.path_scm_data = self.path_result_base / "scm" / f"solar_model_target_duration_{self.scm_duration}.csv"
        self.path_ia_data = self.path_result_base / "ia" / f"ia_target_duration_{self.scm_duration}.csv"

        # ---paths assets----
        self.path_image_raw_cosine_similarity_data = Path(
            "assets", "image_raw_cosine_similarity", self.analysis_settings["image_raw_similarity_name"]
        )

        # ----make data----
        self.create_human_data()
        self.create_dnn_data()
        self.create_match_value_data()
        self.create_ltrs_data()
        self.create_scm_data()
        self.create_ia_data()
        self.create_levenshtein_data()
        self.create_image_raw_cosine_similarity_data()
        self.create_cosine_similarity_layer_wise_data()

        # ----initialise----
        self.read_descriptive_stats_human()

        # ----DNN models----
        self.join_dnn_data()

        # ----match values----
        self.join_descriptive_stats_match_value()

        # ----full priming models----
        self.join_scm_data()
        self.join_ia_data()
        self.join_ltrs_data()

        # ----baselines----
        self.join_image_raw_cosine_similarity()
        self.join_levenshtein()

        # ----layer-wise----
        self.create_cosine_similarity_layer_wise_data()

        self.save_descriptive_stats()
        self.draw_correlation_matrix()
        # self.draw_correlation_matrix_model_scores()

    def make_folders(self):
        if (
            not os.path.exists(
                Path(
                    "results",
                    self.analysis_settings["result_folder"],
                )
            )
            or self.force_process
        ):
            os.makedirs(
                Path(
                    "results",
                    self.analysis_settings["result_folder"],
                ),
                exist_ok=True,
            )

            for i in [
                "human",
                "match_value",
                "model",
                "scm",
                "ltrs",
                "ia",
                "levenshtein",
                "image_raw_cosine_similarity",
                "cosine_similarity_layer_wise",
                "correlation_layer_wise",
            ]:
                os.makedirs(
                    Path(
                        "results",
                        self.analysis_settings["result_folder"],
                        i,
                    ),
                    exist_ok=True,
                )

    def create_human_data(self):
        if not os.listdir(self.path_folder_human_data) or self.force_process:
            DescribeHumanData()

    def create_match_value_data(self):
        if not os.listdir(self.path_folder_match_value_data) or self.force_process:
            ProcessMatchCalculatorData()
            DescribeMatchValue()

    def create_dnn_data(self):
        if not os.listdir(self.path_folder_model_data) or self.force_process:
            DescribeModelData()

    def create_ltrs_data(self):
        DescribeLtrsData()

    def create_scm_data(self):
        DescribeScmData(duration=self.scm_duration)
        DescribeScmDataFull(duration=self.scm_duration)

    def create_ia_data(self):
        DescribeIaDataFull(duration=self.scm_duration)
        DescribeIaData(duration=self.scm_duration)

    def create_levenshtein_data(self):
        if (not os.path.exists(self.path_levenshtein_data)) or self.force_process:
            DescribeLevenshtein(target_file="levenshtein")

    def create_image_raw_cosine_similarity_data(self):
        if (not os.path.exists(self.path_image_raw_cosine_similarity_data)) or self.force_process:
            MakeImageRawCosineSimilarityData()
            DescribeImageRawSimilarity()

    def create_cosine_similarity_layer_wise_data(self):
        if not os.listdir(self.path_folder_correlation_layer_wise) or self.force_process:
            DescribeCosineSimilarityLayerWise(to_which="human")
            DescribeCosineSimilarityLayerWise(to_which="raw_image")

    # ----join data----

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

    def join_dnn_data(self):
        for name in self.dnn_model_labels.keys():
            descriptive_stats = self._read_descriptive_csv(drop_row="primes", sub_folder="model", file_name=f"{name}.csv")
            descriptive_stats = descriptive_stats[["mean"]]
            descriptive_stats.rename({"mean": name}, axis=1, inplace=True)
            self.data_descriptive_main = self.data_descriptive_main.join(descriptive_stats)

    def join_ltrs_data(self):
        descriptive_stats = self._read_descriptive_csv(drop_row="Primes", sub_folder="ltrs", file_name=f"ltrs.csv")
        descriptive_stats = descriptive_stats[["mean"]]
        descriptive_stats.rename({"mean": "LTRS"}, axis=1, inplace=True)
        self.data_descriptive_main = self.data_descriptive_main.join(descriptive_stats)

    def join_scm_data(self):
        dataframe = pandas.read_csv(self.path_scm_data)
        dataframe.rename({"Predicted RT": "SCM"}, axis=1, inplace=True)
        dataframe.set_index("prime_type", inplace=True)
        self.data_descriptive_main = self.data_descriptive_main.join(-dataframe)

    def join_ia_data(self):
        dataframe = pandas.read_csv(self.path_ia_data)
        dataframe.rename({"Predicted RT": "IA"}, axis=1, inplace=True)
        dataframe.set_index("prime_type", inplace=True)
        self.data_descriptive_main = self.data_descriptive_main.join(-dataframe)

    def join_levenshtein(self):
        dataframe = pandas.read_csv(self.path_levenshtein_data, index_col=[0])
        # dataframe = dataframe[["mean"]]
        # dataframe.rename({"mean": "LD"}, axis=1, inplace=True)
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
        dummy_dataframe.rename(self.dnn_model_labels, axis=1, inplace=True)

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
            Path("results", self.analysis_settings["result_folder"], "corr_models.csv"), index_col=0
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
                self.analysis_settings["result_folder"],
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
