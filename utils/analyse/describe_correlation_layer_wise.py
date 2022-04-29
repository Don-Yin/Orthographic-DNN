import json
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas
import seaborn as sns
from scipy.stats import kendalltau, pearsonr, spearmanr
from utils.data_generate.read_corpus import read_corpus


class DescribeCosineSimilarityLayerWise:
    def __init__(self, to_which="human"):
        self.to_which = to_which
        self.analysis_settings = json.load(open(Path("analysis_settings.json"), "r"))
        self.sorter = json.load(open(Path("assets", "sorter_human_data.json"), "r"))
        self.method = self.analysis_settings["correlation_method"]

        self.read_data()
        self.read_error_dict()
        self.read_selected_model_names()
        self.read_descriptive_stats_human()
        self.read_descriptive_stats_image_raw_cosine_similarity()
        self.dict_error_correction()
        self.to_dataframe()
        self.process_all_models()
        self.plot()

    def read_error_dict(self):
        self.error_dict = json.load(open(Path("assets", "label_error.json"), "r"))
        self.prime_types_2014 = read_corpus(Path("assets", "2014-prime-types.txt"))
        for i in range(len(self.prime_types_2014)):
            if self.prime_types_2014[i] in self.error_dict.keys():
                self.prime_types_2014[i] = self.error_dict[self.prime_types_2014[i]]

    def read_data(self):
        self.dataframe = json.load(
            open(Path("assets", "model_all_layers", self.analysis_settings["model_all_layers_data_name"]), "r")
        )

    def read_selected_model_names(self):
        self.selected_model_names = json.load(open(Path("assets", "selected_models.json"), "r"))

    def read_descriptive_stats_human(self):
        self.data_descriptive_human = self._read_descriptive_csv(
            drop_row="cond.label", sub_folder="human", file_name="descriptive_stats_human_data.csv"
        )
        self.data_descriptive_human = self.data_descriptive_human[["priming_arb"]]

    def read_descriptive_stats_image_raw_cosine_similarity(self):
        self.data_descriptive_image_raw_cosine_similarity = self._read_descriptive_csv(
            drop_row="primes", sub_folder="image_raw_cosine_similarity", file_name="image_raw_cosine_similarity.csv"
        )

        self.data_descriptive_image_raw_cosine_similarity = self.data_descriptive_image_raw_cosine_similarity[["mean"]]

    def dict_error_correction(self):
        # correct labeling error + translate primes from list to str
        for i in range(len(self.dataframe)):
            if self.dataframe[i]["primes"][1] in self.error_dict.keys():
                self.dataframe[i]["primes"] = self.error_dict[self.dataframe[i]["primes"][1]]
            else:
                self.dataframe[i]["primes"] = self.dataframe[i]["primes"][1]

    def to_dataframe(self):
        self.dataframe = pandas.DataFrame(self.dataframe)

    def process_all_models(self):
        self.correlation_layer_wise = [self.process_single_model(model) for model in self.selected_model_names]
        self.correlation_layer_wise_frame = pandas.DataFrame(self.correlation_layer_wise)
        max_len = self.correlation_layer_wise_frame["correlation_each_layer"].apply(len).max()
        combinations = list(product(*[self.selected_model_names, range(max_len)]))
        self.correlation_layer_wise_frame = pandas.DataFrame(
            [
                {
                    "Model": c[0],
                    "Layer": c[1],
                    "Correlation Coefficient τ": self.get_correlation_value_by_name_and_layer_index(
                        model=c[0], layer_index=c[1]
                    ),
                }
                for c in combinations
            ]
        )

        self.correlation_layer_wise_frame["Model"] = self.correlation_layer_wise_frame["Model"].replace(
            json.load(open(Path("assets", "selected_model_labels.json"), "r"))
        )

        if self.to_which == "human":
            self.correlation_layer_wise_frame.to_csv(
                Path("results", self.analysis_settings["result"], "correlation_layer_wise", "correlation_layer_wise.csv"),
                index=False,
            )
        elif self.to_which == "raw_image":
            self.correlation_layer_wise_frame.to_csv(
                Path(
                    "results",
                    self.analysis_settings["result"],
                    "correlation_layer_wise",
                    "correlation_layer_wise_with_raw_image.csv",
                ),
                index=False,
            )

    def plot(self):
        sns.set_theme(style="white", font_scale=1.3)
        if self.to_which == "human":
            data = pandas.read_csv(
                Path("results", self.analysis_settings["result"], "correlation_layer_wise", "correlation_layer_wise.csv")
            )
            data['Correlation Coefficient τ'] = data['Correlation Coefficient τ'].rolling(10, center=True).mean()
            plt.figure(figsize=(30, 14))
            sns.lineplot(x="Layer", y="Correlation Coefficient τ", hue="Model", data=data).figure.savefig(
                Path("results", self.analysis_settings["result"], "correlation_layer_wise", "correlation_layer_wise.png")
            )
        elif self.to_which == "raw_image":
            data = pandas.read_csv(
                Path(
                    "results",
                    self.analysis_settings["result"],
                    "correlation_layer_wise",
                    "correlation_layer_wise_with_raw_image.csv",
                )
            )
            plt.figure(figsize=(32, 8))
            sns.lineplot(x="Layer", y="Correlation Coefficient τ", hue="Model", data=data).figure.savefig(
                Path(
                    "results",
                    self.analysis_settings["result"],
                    "correlation_layer_wise",
                    "correlation_layer_wise_with_raw_image.png",
                )
            )

    def get_correlation_value_by_name_and_layer_index(self, model, layer_index):
        try:
            return [i for i in self.correlation_layer_wise if i["Model"] == model][0]["correlation_each_layer"][layer_index]
        except IndexError:
            return None

    def process_single_model(self, model: str) -> dict:
        """output -> one list of cosines similarity of that particular model"""
        dummy_frame = self.dataframe.copy()
        dummy_frame = dummy_frame[dummy_frame.model == model]

        list_dict_prime_type_layer_average = [
            {"prime_type": t, "average_each_layer": self.get_average_list_from_prime_type(dummy_frame, t)}
            for t in self.prime_types_2014
        ]

        correlation_each_layer = []
        for i in range(len(list_dict_prime_type_layer_average[0]["average_each_layer"])):
            frame = pandas.DataFrame(
                [
                    {"prime_type": d["prime_type"], "average": d["average_each_layer"][i]}
                    for d in list_dict_prime_type_layer_average
                ]
            )
            frame = frame.set_index("prime_type")
            frame = frame.reindex(self.sorter)

            if self.to_which == "human":
                correlation_each_layer.append(
                    self._get_correlation_callable(
                        frame["average"].values, self.data_descriptive_human["priming_arb"].values
                    )
                )
            elif self.to_which == "raw_image":
                correlation_each_layer.append(
                    self._get_correlation_callable(
                        frame["average"].values, self.data_descriptive_image_raw_cosine_similarity["mean"].values
                    )
                )

        return {"Model": model, "correlation_each_layer": correlation_each_layer}

    def get_average_list_from_prime_type(self, dataframe, prime_type: str):
        dataframe = dataframe[dataframe.primes == prime_type]
        arrays = [np.array(x) for x in dataframe.cosine_similarities]
        return [np.mean(k) for k in zip(*arrays)]

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

    def _get_correlation_callable(self, vector_1, vector_2):
        if self.method == "kendall":
            return kendalltau(vector_1, vector_2, variant="c", alternative="greater")[0]
        elif self.method == "spearman":
            return spearmanr(vector_1, vector_2, axis=0, alternative="greater")[0]
        elif self.method == "pearson":
            return pearsonr(vector_1, vector_2)[0]


if __name__ == "__main__":
    DescribeCosineSimilarityLayerWise()
