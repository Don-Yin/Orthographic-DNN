# the purpose of this file is to bootstrap multiple correlation coefficients for each neural network

import itertools
import json
from pathlib import Path
from random import sample

import matplotlib.pyplot as plt
import numpy as np
import pandas
from scipy.stats import kendalltau, ttest_ind
from tqdm import tqdm

# models: {'resnet101', 'vit_l_16', 'vgg16', 'efficientnet_b1', 'alexnet', 'vgg19', 'resnet50', 'densenet169'}

# sorter: ["ID", "DL-1F", "IL-1F", "TL56", "TL-M", "DL-1M", "SN-F", "SN-I", "TL12", "IL-1M", "IL-1I", "SUB3", "IL-2MR", "DL-2M", "SN-M", "N1R", "NATL-24/35", "IL-2M", "T-All", "DSN-M", "RH", "NATL25", "IH", "TH", "ALD-PW", "RF", "EL", "ALD-ARB"]

# example: {
#     "model": "alexnet",
#     "word": "abduct",
#     "primes": [
#         "ID",
#         "ID"
#     ],
#     "similarity_penultimate": 1.0,
#     "similarity_classification": 1.0
# }


class Bootstrap:
    def __init__(self):
        """bootstrap samples from model's output and generate 1000 correlations with human data for each model"""
        self.read_data()
        self.read_human_data()
        self.sample_size = 1
        self.same_word = False
        output = []
        for model in self.models:
            sequence = [self.get_correlation_with_human(model) for _ in tqdm(range(1000))]
            output.append({"model": model, "correlations": sequence})
        json.dump(output, open(Path(f"bootstrap_output-{self.sample_size}.json"), "w"))

        # output = []
        # for _ in range(128):
        #     sequence = [self.get_correlation_with_human("alexnet") for _ in range(128)]
        #     output.append(sequence)
        # json.dump(output, open(Path("alexnet-same-sample.json"), "w"))

    def read_data(self):
        with open(Path("assets", "label_error.json")) as loader:
            self.label_error = json.load(loader)
        with open(Path("assets", "model_output", "normal.json")) as loader:
            self.data = json.load(loader)
        with open(Path("assets", "sorter_human_data.json")) as loader:
            self.sorter = json.load(loader)
        self.data = [self.correct_error(i) for i in self.data]
        self.models = set([i["model"] for i in self.data])
        self.words = list(set([i["word"] for i in self.data]))  # from a set deprecated

    def read_human_data(self, drop_row: str = "cond.label"):
        path = Path("assets", "descriptive_stats_human_data.csv")
        data = pandas.read_csv(path, index_col=[0], skiprows=1)
        data.drop(drop_row, inplace=True)
        self.human_data = data[["priming_arb"]]

    def correct_error(self, d: dict):
        if d["primes"][1] in self.label_error.keys():
            d["primes"][1] = self.label_error[d["primes"][1]]
        return d

    def _get_correlation(self, vector_1, vector_2):
        return kendalltau(vector_1, vector_2, variant="c", alternative="greater")[0]

    def get_mean_score(self, model: str, condition: str):
        if self.same_word:
            sampled_words = sample(self.words, self.sample_size)
            data = [i for i in self.data if i["model"] == model and i["primes"][1] == condition]
            data = [i for i in data if i["word"] in sampled_words]
            data = [i["cosine_similarities"][-2] for i in data]
        else:
            data = [i for i in self.data if i["model"] == model and i["primes"][1] == condition]
            data = sample(data, 1)
            data = [i["cosine_similarities"][-2] for i in data]
        return sum(data) / len(data)  # calculate average

    def get_correlation_with_human(self, model: str):
        mean_similarities = [self.get_mean_score(model=model, condition=i) for i in self.sorter]
        return self._get_correlation(mean_similarities, self.human_data)


class Bootstrap_Priming_Models:
    def __init__(self):
        self.read_data()
        self.read_human_data()
        output = []
        for model in ["IA", "SOLAR"]:
            self.read_model_data("IA")
            sequence = [self.get_correlation_with_human() for _ in tqdm(range(1000))]
            output.append({"model": model, "correlations": sequence})
        json.dump(output, open(Path("bootstrap_output-priming-models.json"), "w"))

    def read_data(self):
        with open(Path("assets", "label_error.json")) as loader:
            self.label_error = json.load(loader)
        with open(Path("assets", "sorter_human_data.json")) as loader:
            self.sorter = json.load(loader)

    def read_model_data(self, model):
        with open(Path(f"{model}_result.json")) as loader:
            self.model_data = json.load(loader)

    #     {
    #     "Primes": "TH",
    #     "Predicted RT": 155
    # }

    def read_human_data(self, drop_row: str = "cond.label"):
        path = Path("assets", "descriptive_stats_human_data.csv")
        data = pandas.read_csv(path, index_col=[0], skiprows=1)
        data.drop(drop_row, inplace=True)
        self.human_data = data[["priming_arb"]]

    def _get_correlation(self, vector_1, vector_2):
        return kendalltau(vector_1, vector_2, variant="c", alternative="greater")[0]

    def get_condition_score(self, condition: str):
        data = [i for i in self.model_data if i["Primes"] == condition]
        data = sample(data, 1)[0]["Predicted RT"]
        return data

    def get_correlation_with_human(self):
        mean_similarities = [self.get_condition_score(condition=i) for i in self.sorter]
        return - self._get_correlation(mean_similarities, self.human_data)


class BootstrapJeff:
    """output: 420 coefficients for each model"""

    def __init__(self):
        self.read_data()
        self.read_human_data()
        output = []
        for model in self.models:
            sequence = self.get_correlation_with_human(model)
            output.append({"model": model, "correlations": sequence})
        json.dump(output, open(Path("bootstrap_output-jeff.json"), "w"))

    def read_data(self):
        with open(Path("assets", "label_error.json")) as loader:
            self.label_error = json.load(loader)
        with open(Path("assets", "model_output", "normal.json")) as loader:
            self.data = json.load(loader)
        with open(Path("assets", "sorter_human_data.json")) as loader:
            self.sorter = json.load(loader)
        self.data = [self.correct_error(i) for i in self.data]
        self.models = set([i["model"] for i in self.data])
        self.words = list(set([i["word"] for i in self.data]))  # from a set deprecated

    def read_human_data(self, drop_row: str = "cond.label"):
        path = Path("assets", "descriptive_stats_human_data.csv")
        data = pandas.read_csv(path, index_col=[0], skiprows=1)
        data.drop(drop_row, inplace=True)
        self.human_data = data[["priming_arb"]]

    def correct_error(self, d: dict):
        if d["primes"][1] in self.label_error.keys():
            d["primes"][1] = self.label_error[d["primes"][1]]
        return d

    def _get_correlation(self, vector_1, vector_2):
        return kendalltau(vector_1, vector_2, variant="c", alternative="greater")[0]

    # fixed ----------------------------------------------

    def get_word_correlation_coefficient(self, model: str, word: str):
        output = []
        for condition in self.sorter:
            data = [i for i in self.data if i["model"] == model and i["primes"][1] == condition and i["word"] == word]
            data = [i["cosine_similarities"][-2] for i in data][0]
            output.append(data)
        return output

    def get_correlation_with_human(self, model: str):
        correlations = [self._get_correlation(self.get_word_correlation_coefficient(model=model, word=i), self.human_data) for i in self.words]
        return correlations


# {'vit_b_16', 'vit_l_32', 'efficientnet_b1', 'vit_l_16', 'resnet101', 'resnet50', 'vgg16', 'densenet169', 'vgg19', 'vit_b_32', 'alexnet'}


class T_T:
    def __init__(self):
        self.read_data()
        self.set_classes()
        # self.get_single_model_p_distribution()
        self.models = set([i["model"] for i in self.data])
        output = self.get_t_test("resnet50", "resnet101")
        # for i in self.models:
        #     output = self.get_same_model_ttest(i)
        #     print(f"{i}: {output}")
        # output = self.get_group_t_test()
        print(output)

    def set_classes(self):
        self.classes = {
            "ViTs": ["vit_b_16", "vit_l_32", "vit_b_32", "vit_l_16"],
            "conv": ["efficientnet_b1", "resnet101", "resnet50", "vgg16", "densenet169", "vgg19", "alexnet"],
        }

    def read_data(self):
        self.data = json.load(open(Path("bootstrap_output-1.json")))
        # self.data_2 = json.load(open(Path("bootstrap_output-42.json")))

    def get_t_test(self, model_1: str, model_2: str):
        data_1 = [i["correlations"] for i in self.data if i["model"] == model_1][0]
        data_2 = [i["correlations"] for i in self.data if i["model"] == model_2][0]
        return ttest_ind(data_1, data_2)

    def get_same_model_ttest(self, model):
        data_1 = [i["correlations"] for i in self.data if i["model"] == model][0]
        data_2 = [i["correlations"] for i in self.data_2 if i["model"] == model][0]
        return ttest_ind(data_1, data_2)

    def get_group_t_test(self):
        data_1 = [i["correlations"] for i in self.data if i["model"] in self.classes["ViTs"]]
        data_1 = [item for sublist in data_1 for item in sublist]
        data_2 = [i["correlations"] for i in self.data if i["model"] in self.classes["conv"]]
        data_2 = [item for sublist in data_2 for item in sublist]
        return ttest_ind(data_1, data_2)

    def get_single_model_p_distribution(self):
        with open(Path("alexnet-same-sample.json")) as loader:
            self.coefficients = json.load(loader)
        combinations = itertools.permutations(self.coefficients, 2)
        combinations = list(combinations)
        p_values = []
        for i in combinations:
            p_values.append(ttest_ind(i[0], i[1]).pvalue)
        plt.hist(p_values)
        plt.savefig("foo-3.png")


def get_SE(model: str):
    with open(Path("bootstrap_output-1.json")) as loader:
        data = json.load(loader)
    data = [i["correlations"] for i in data if i["model"] == model][0]
    SE = np.std(data) / np.sqrt(len(data))
    SD = np.std(data)
    # [\Sum [(r*(i) – mean(r*))^2 / (1000 – 1) ]^1/2
    return SD.round(3)


def get_model_class(model):
    classes = {
        "ViTs": ["vit_b_16", "vit_l_32", "vit_b_32", "vit_l_16"],
        "Convolutional Models": ["efficientnet_b1", "resnet101", "resnet50", "vgg16", "densenet169", "vgg19", "alexnet"],
    }

    for key in classes.keys():
        if model in classes[key]:
            return key


def get_SE_data():
    with open(Path("bootstrap_output-1.json")) as loader:
        data = json.load(loader)
    with open(Path("bootstrap_output-priming-models.json")) as loader:
        data_priming_models = json.load(loader)

    labels = {
        "alexnet": "AlexNet",
        "densenet169": "DenseNet169",
        "efficientnet_b1": "EfficientNet-B1",
        "resnet50": "ResNet50",
        "resnet101": "ResNet101",
        "vgg16": "VGG16",
        "vgg19": "VGG19",
        "vit_b_16": "ViT-B/16",
        "vit_b_32": "ViT-B/32",
        "vit_l_16": "ViT-L/16",
        "vit_l_32": "ViT-L/32",
        "IA": "Interactive Activation Model",
        "SOLAR": "Spatial Coding Model",
    }

    SEs = []
    for d in data:
        for i in d["correlations"]:
            SEs.append({"class": get_model_class(d["model"]), "model": labels[d["model"]], "correlation": i})
    
    for d in data_priming_models:
        for i in d["correlations"]:
            SEs.append({"class": "Priming Models", "model": labels[d["model"]], "correlation": i})

    SEs.append({"class": "Baselines", "model": "pixCS", "correlation": 0.08})
    SEs.append({"class": "Coding Schemes", "model": "Absolute", "correlation": 0.53})
    SEs.append({"class": "Coding Schemes", "model": "Spatial Coding", "correlation": 0.56})
    SEs.append({"class": "Coding Schemes", "model": "Binary Open Bigram", "correlation": 0.67})
    SEs.append({"class": "Coding Schemes", "model": "Overlap Open Bigram", "correlation": 0.66})
    SEs.append({"class": "Coding Schemes", "model": "SERIOL Open Bigram", "correlation": 0.44})

    # SEs.append({"class": "Priming Models", "model": "Spatial Coding Model", "correlation": 0.68})
    # SEs.append({"class": "Priming Models", "model": "Interactive Activation Model", "correlation": 0.57})
    SEs.append({"class": "Priming Models", "model": "LTRS", "correlation": 0.7})

    SEs.append({"class": "Baselines", "model": "LevDist", "correlation": 0.69})

    return SEs


SEs = get_SE_data()

if __name__ == "__main__":
    # Bootstrap()
    # BootstrapJeff()
    # T_T()
    # print(get_SE("densenet169"))
    # get_SE_data("alexnet")
    Bootstrap_Priming_Models()


