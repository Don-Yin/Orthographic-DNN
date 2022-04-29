import json
import os
import platform
from itertools import product
from pathlib import Path

import torch
from torchvision.datasets import ImageFolder as torch_image_folder
from tqdm import tqdm
from utils.data_generate.read_corpus import read_corpus
from utils.data_load.normalize import add_compute_stats
from utils.evaluate.cosine_similarity import compute_cosine_similarity


class MakeImageRawCosineSimilarityData:
    def __init__(self):
        self.analysis_settings = json.load(open(Path("analysis_settings.json"), "r"))
        self.path_output = Path("assets", "image_raw_cosine_similarity", self.analysis_settings["image_raw_similarity_name"])
        self.read_words_and_primes()
        self.read_prime_data()
        self.make_combinations()
        self.main()

    def read_words_and_primes(self):
        self.targets_2014 = read_corpus(Path("assets", "2014-targets.txt"))
        self.prime_types_2014 = read_corpus(Path("assets", "2014-prime-types.txt"))

    def read_prime_data(self):
        self.prime_data = add_compute_stats(torch_image_folder)(
            root=str(Path("data") / self.analysis_settings["prime_data_folder"]), stats=json.load(open(Path("data", "normalization_stats.json"), "r"))
        )

    def make_combinations(self):
        if not os.path.exists(self.path_output):
            self.combinations = list(product(*[self.targets_2014, ["ID"], self.prime_types_2014]))
            self.combinations = [{"word": c[0], "primes": (c[1], c[2])} for c in self.combinations]
            self.save()
        else:
            self.combinations = json.load(open(self.path_output, "r"))

    def save(self):
        json.dump(self.combinations, open(self.path_output, "w"))

    def main(self):
        for i in tqdm(range(len(self.combinations))):
            if any(["cosine_similarity" not in self.combinations[i].keys()]):
                self.combinations[i]["cosine_similarity"] = self.get_cosine_similarity(self.combinations[i])

                if i % 1024 == 0:
                    self.save()
                    print(f"\nSaved: {i}")

        self.save()
        print("\nSaved: All")

    def get_cosine_similarity(self, dict: dict):
        image_label_prime: list[list] = [
            i[0].replace(".png", "").split(("/" if platform.system() != "Windows" else "\\"))[2::]
            for i in self.prime_data.imgs
        ]
        image_indices: tuple[int] = (
            image_label_prime.index([dict["word"], dict["primes"][0]]),
            image_label_prime.index([dict["word"], dict["primes"][1]]),
        )

        tensors = (
            torch.flatten(self.prime_data[image_indices[0]][0].cuda().detach()),
            torch.flatten(self.prime_data[image_indices[1]][0].cuda().detach()),
        )

        return compute_cosine_similarity(tensors, dimension=0)


if __name__ == "__main__":
    MakeImageRawCosineSimilarityData()
