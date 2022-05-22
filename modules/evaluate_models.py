import json
import logging
import os
import random
import sys
from itertools import product
from pathlib import Path

import psutil
import torch
import torchvision
from torchvision.datasets import ImageFolder as torch_image_folder
from tqdm import tqdm

from utils.data_generate.read_corpus import read_corpus
from utils.data_load.device_control import device_allocator
from utils.data_load.normalize import add_compute_stats
from utils.evaluate.evaluate_model import Evaluate


def restart_program():
    """Restarts the current program, with file objects and descriptors cleanup"""
    try:
        p = psutil.Process(os.getpid())
        for handler in p.open_files() + p.connections():
            os.close(handler.fd)
    except Exception as e:
        logging.error(e)

    python = sys.executable
    os.execl(python, python, *sys.argv)


class BatchEvaluate:
    """_summary_
    returns: dataframe -
    cols:
        model, word,
        prime_type_1 (ID), prime_type_2,

        cosine_similarity (penultimate),
        cosine_similarity (classification)

    create a list of dicts and transform into dataframe:
    https://stackoverflow.com/questions/20638006/convert-list-of-dictionaries-to-a-pandas-dataframe
    """

    def __init__(self, validate: False):
        self.analysis_settings = json.load(open(Path("analysis_settings.json"), "r"))
        self.path_output = Path("assets", "model_output", "normal.json")
        self.validate = validate
        if self.validate:
            self.load_validation_data()
        self.read_prime_data()
        self.read_words_and_primes()
        self.read_selected_models()
        self.make_combinations()
        self.load_models()
        self.main()
        self.save()

    def read_words_and_primes(self):
        self.targets_2014 = read_corpus(Path("assets", "2014-targets.txt"))
        self.prime_types_2014 = read_corpus(Path("assets", "2014-prime-types.txt"))

    def read_selected_models(self):
        self.selected_network_names = json.load(open(Path("assets", "selected_models.json"), "r"))

    def read_prime_data(self):
        self.prime_data = add_compute_stats(torch_image_folder)(
            root=str(Path("data") / self.analysis_settings["prime_data_folder"]),
            stats=json.load(open(Path("data", "normalization_stats.json"), "r")),
        )

    def make_combinations(self):
        if not os.path.exists(self.path_output):
            self.combinations = list(
                product(*[self.selected_network_names, self.targets_2014, ["ID"], self.prime_types_2014])
            )
            self.combinations = [{"model": c[0], "word": c[1], "primes": (c[2], c[3])} for c in self.combinations]
            self.save()
        else:
            self.combinations = json.load(open(self.path_output, "r"))

    def load_validation_data(self):
        normalization_stats = json.load(open(Path("data", "normalization_stats.json"), "r"))
        self.data_valid = add_compute_stats(torch_image_folder)(
            root=str(Path("data") / "data_valid"), stats=normalization_stats
        )

    def main(self):
        """_summary_
        Add similarity stats to the main dictionary.
        """
        try:
            for i in tqdm(range(len(self.combinations))):
                if "cosine_similarities" not in self.combinations[i].keys():
                    self.combinations[i]["cosine_similarities"] = Evaluate(
                        model=getattr(self, self.combinations[i]["model"]),
                        word=self.combinations[i]["word"],
                        prime_types=self.combinations[i]["primes"],
                        which_layer="all",
                        prime_data=self.prime_data,
                    ).compute_similarity_layer_wise()

                    if i % 128 == 0:
                        self.save()
                        print(f"\nSaved: {i}")

                    if i % 1024 == 0:
                        restart_program()

        except RuntimeError:
            restart_program()

    def save(self):
        json.dump(self.combinations, open(self.path_output, "w"))

    def validate_model_prediction(self, model):
        correct = 0
        for i in random.sample(range(len(self.data_valid)), 1000):
            output = model(self.data_valid[i][0].unsqueeze(0).cuda())
            _, prediction = torch.max(output, dim=1)
            if int(prediction) == self.data_valid[i][1]:
                correct += 1
        if correct < 950:
            print(f"Accuracy: {correct/10}%")
            raise Exception(f"{model} accuracy is lower than 95%")
        else:
            print(f"Accuracy: {correct/10}%")

    def load_models(self):
        self.alexnet = torchvision.models.alexnet(pretrained=False)
        self.densenet169 = torchvision.models.densenet169(pretrained=False)
        self.efficientnet_b1 = torchvision.models.efficientnet_b1(pretrained=False)
        self.resnet50 = torchvision.models.resnet50(pretrained=False)
        self.resnet101 = torchvision.models.resnet101(pretrained=False)
        self.vgg16 = torchvision.models.vgg16(pretrained=False)
        self.vgg19 = torchvision.models.vgg19(pretrained=False)
        self.vit_b_16 = torchvision.models.vit_b_16(pretrained=False)
        self.vit_b_32 = torchvision.models.vit_b_32(pretrained=False)
        self.vit_l_16 = torchvision.models.vit_l_16(pretrained=False)
        self.vit_l_32 = torchvision.models.vit_l_32(pretrained=False)

        for model in self.selected_network_names:
            # for model in ["vit_l_16"]:
            getattr(self, model).load_state_dict(torch.load(Path("params", f"{model}.pth")))
            getattr(self, model).eval()
            device_allocator(getattr(self, model))
            if self.validate:
                self.validate_model_prediction(getattr(self, model))
            print(f"State dict loaded: {model}")


if __name__ == "__main__":
    BatchEvaluate(validate=False)
