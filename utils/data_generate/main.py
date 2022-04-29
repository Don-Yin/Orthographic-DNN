import json
import math
import os
import random
import shutil
from itertools import product
from operator import mul
from pathlib import Path

import numpy as np
from numpy import arange
from PIL import Image
from PIL.Image import new
from PIL.ImageDraw import Draw
from PIL.ImageFont import truetype
from scipy import ndimage
from torchvision.datasets import ImageFolder as torch_image_folder
from tqdm import tqdm
from utils.data_generate.read_corpus import read_corpus
from utils.data_load.normalize import add_compute_stats


class CreateData:
    def __init__(self, mode: str):
        self.mode = mode
        self.width: int = 224
        self.height: int = 224
        self.center: tuple = (self.width / 2, self.height / 2)
        self.path_folder_data: Path = Path("data")
        self.ratio_train_valid_data: float = 4 / 1
        self.coefficient_data_valid: float = self.ratio_train_valid_data**-1
        self.coefficient_data_train: float = 1 - self.coefficient_data_valid
        self.analysis_settings = json.load(open(Path("analysis_settings.json"), "r"))

    def set_attributes(self, c: dict):
        if self.mode == "main":
            self.word = c["word"]

        elif self.mode == "prime":
            self.target = c["target"]
            self.prime_type = c["prime"]
            self.content = c["content"]

        self.create_main_folders()
        self.name_font = c["font"]
        self.size_font = c["size"]
        self.position = c["position"]
        self.index = c["index"]
        self.name_save = c["name_save"]

        try:
            self.font = truetype(os.path.abspath(Path("assets", "fonts") / self.name_font), self.size_font)
        except OSError:
            print(self.name_font)

    def set_configuration(self, configuration: dict):
        self.sd_rotation = configuration["standard_deviation_rotation"]
        self.variance_font = configuration["variance_font"]
        self.coefficient_space = configuration["coefficient_space"] + 1
        self.coefficient_translate = configuration["coefficient_translation"]

    def create_main_folders(self):
        if self.mode == "main":
            self.ensure_folder_exists(self.path_folder_data)
            self.ensure_folder_exists(self.path_folder_data / "data_all")
            self.ensure_folder_exists(self.path_folder_data / "data_all" / self.word)
            self.ensure_folder_exists(self.path_folder_data / "data_train")
            self.ensure_folder_exists(self.path_folder_data / "data_valid")
        elif self.mode == "prime":
            self.ensure_folder_exists(self.path_folder_data / self.analysis_settings["prime_data_folder"])
            self.ensure_folder_exists(self.path_folder_data / self.analysis_settings["prime_data_folder"] / self.target)

    def ensure_folder_exists(self, path: Path):
        if not os.path.exists(path):
            os.mkdir(path)

    def create_images(self):
        self.word = self.content if self.mode == "prime" else self.word
        self.word = self.word.upper()
        canvas = new("RGB", (self.width, self.height), color=(0, 0, 0))
        self.letters_size_font_shift: list = [
            random.randint(-self.variance_font, self.variance_font) for _ in range(0, len(self.word))
        ]
        self.width_letters_cumulative = self.get_width_letters(Draw(canvas))
        self.shape_half_word: tuple = (self.width_letters_cumulative[-1] / 2, self.get_max_height(Draw(canvas)) / 2)
        self.average_diagonal_length = self.get_average_diagonal_length(Draw(canvas))
        self.radius_translate = self.get_radius_translate()
        self.coords_after_move = self.get_coords_after_move()

        for i in range(0, len(self.word)):
            letter_font = self.get_zoomed_font(self.letters_size_font_shift[i])

            shape_letter = Draw(canvas).textsize(self.word[i], font=letter_font)
            canvas_letter = new("RGBA", shape_letter, color=(0, 0, 0))
            Draw(canvas_letter).text((0, 0), self.word[i], fill=(255, 255, 255, 255), font=letter_font)

            canvas_letter = self.rotate(canvas_letter, angle=np.random.normal(loc=0.0, scale=self.sd_rotation, size=None))

            canvas_letter.putdata(self.set_background_transparent(canvas_letter))
            canvas.paste(im=canvas_letter, box=self.get_final_position_letter(i), mask=canvas_letter)

        if self.mode == "main":
            canvas.save(self.path_folder_data / "data_all" / self.word.lower() / self.name_save)

        elif self.mode == "prime":
            canvas.save(self.path_folder_data / self.analysis_settings["prime_data_folder"] / self.target / self.name_save)

    def rotate(self, image, angle: int):
        image_array = np.array(image)
        rotated_array = ndimage.rotate(image_array, angle, cval=0.0, reshape=True, mode="constant", prefilter=True)
        return Image.fromarray(rotated_array)

    def zoom(self, image, factor: float):
        image_array = np.array(image).astype(np.uint8)
        zoomed_array = ndimage.zoom(input=image_array, zoom=factor, cval=0.0, mode="constant", prefilter=True, order=0)
        return Image.fromarray(zoomed_array)

    def get_radius_translate(self):
        return self.coefficient_translate * self.average_diagonal_length

    def get_width_letters(self, draw: Draw) -> list:
        shapes_letters: list = [
            draw.textsize(self.word[i], font=self.get_zoomed_font(self.letters_size_font_shift[i]))
            for i in range(0, len(self.word))
        ]
        width_letters: list = [i[0] * self.coefficient_space for i in shapes_letters]
        width_letters.insert(0, 0)
        return np.cumsum(width_letters)

    def get_average_diagonal_length(self, draw: Draw) -> float:
        shapes_letters: list = [
            draw.textsize(self.word[i], font=self.get_zoomed_font(self.letters_size_font_shift[i]))
            for i in range(0, len(self.word))
        ]
        return sum([math.sqrt(s[0] ** 2 + s[1] ** 2) for s in shapes_letters]) / len(shapes_letters)

    def get_max_height(self, draw: Draw) -> float:
        shapes_letters: list = [
            draw.textsize(
                self.word[i],
                font=truetype(
                    os.path.abspath(Path("assets", "fonts") / self.name_font),
                    self.size_font + self.letters_size_font_shift[i],
                ),
            )
            for i in range(0, len(self.word))
        ]
        return max([s[1] for s in shapes_letters])

    def get_zoomed_font(self, zoom: int):
        return truetype(os.path.abspath(Path("assets", "fonts") / self.name_font), self.size_font + zoom)

    def set_background_transparent(self, image) -> list:
        return [(lambda i: (0, 0, 0, 0) if i[:3] == (0, 0, 0) else i)(i) for i in image.getdata()]

    def get_translation_vector(self, radius: float) -> tuple:
        r = radius * math.sqrt(random.random())
        theta = random.random() * 2 * math.pi
        return (r * math.cos(theta), r * math.sin(theta))

    def get_coords_after_move(self) -> tuple:
        word_position_vector = tuple(
            map(
                mul,
                self.position,
                (self.width * 0.5 - self.shape_half_word[0], self.height * 0.5 - self.shape_half_word[1]),
            )
        )
        coords_centered = tuple(np.subtract(self.center, self.shape_half_word))
        return tuple(map(sum, zip(word_position_vector, coords_centered)))

    def get_final_position_letter(self, instance) -> tuple:
        position_letter: tuple = (
            self.coords_after_move[0] + self.width_letters_cumulative[instance],
            self.coords_after_move[1],
        )
        position_letter: tuple = tuple(map(sum, zip(self.get_translation_vector(self.radius_translate), position_letter)))
        return (int(position_letter[0]), int(position_letter[1]))

    def split_images_to_folders(self):
        list_folders_in_data_all: list[str] = os.listdir(self.path_folder_data / "data_all")

        for folder in list_folders_in_data_all:
            list_images_in_folder: list = os.listdir(self.path_folder_data / "data_all" / folder)
            self.ensure_folder_exists(self.path_folder_data / "data_train" / folder)
            self.ensure_folder_exists(self.path_folder_data / "data_valid" / folder)

            list_images_to_train_folder: list[str] = random.sample(
                list_images_in_folder, int(len(list_images_in_folder) * self.coefficient_data_train)
            )

            list_images_to_valid_folder: list[str] = [
                i for i in list_images_in_folder if i not in list_images_to_train_folder
            ]

            for image in list_images_to_train_folder:
                source: Path = self.path_folder_data / "data_all" / folder / image
                destiny: Path = self.path_folder_data / "data_train" / folder / image
                shutil.move(source, destiny)

            for image in list_images_to_valid_folder:
                source: Path = self.path_folder_data / "data_all" / folder / image
                destiny: Path = self.path_folder_data / "data_valid" / folder / image
                shutil.move(source, destiny)

        shutil.rmtree(self.path_folder_data / "data_all")


def init_create_train_data(dummy: bool = False):
    corpus: list[str] = read_corpus(Path("assets", "1000-corpus.txt"))

    fonts: list = [f for f in os.listdir(Path("assets", "fonts")) if f.endswith(".ttf")]
    sizes: list = list(arange(18, 38, 2)) if not dummy else list(arange(18, 20, 2))
    positions: list = [(0, 0)]

    configuration_variation: dict = {
        "standard_deviation_rotation": 8.0,  # sd of angle
        "coefficient_translation": 0.04,  # letter: coefficient * (diagonal len of average letter box in the word)
        "coefficient_space": 0.5,  # space between letters = previous letter width * this coefficient
        "variance_font": 3,  # variance of font size
    }

    combinations_variation: list = (
        list(product(*[corpus, fonts, sizes, positions])) * 80
        if not dummy
        else list(product(*[corpus, fonts, sizes, positions]))
    )

    for i in range(len(combinations_variation)):
        combinations_variation[i] += (i,)

    combinations_variation: list[dict] = [
        {
            "word": c[0],
            "font": c[1],
            "size": c[2],
            "position": c[3],
            "index": c[4],
            "name_save": "-".join(
                [
                    c[0].lower(),
                    c[1],
                    str(c[2]),
                    # f"{int(self.position[0]):.5f}",
                    # f"{int(self.position[1]):.5f}",
                    str(c[4]),
                    ".png",
                ]
            ),
        }
        for c in combinations_variation
    ]

    generated_files = [file.name for file in Path("data").rglob("*.png")]

    combinations_variation = [c for c in combinations_variation if c["name_save"] not in generated_files]

    create = CreateData(mode="main")

    print("\n")
    print("--------------------------------")
    print(f"Total images: {len(combinations_variation)}")
    print(f"Images in train set: {int(len(combinations_variation) * create.coefficient_data_train)}")
    print(f"Images in valid set: {int(len(combinations_variation) * create.coefficient_data_valid)}")
    print("--------------------------------")
    print(f"Words: {len(corpus)}")
    print(f"Fonts: {len(fonts)}")
    print(f"Level of sizes: {len(sizes)}")
    print(f"Random positions (1 == all centred): {len(positions)}")
    print("--------------------------------")
    print(f"Letter rotation angle sd: {configuration_variation['standard_deviation_rotation']}")
    print(f"Letter translation coefficient: {configuration_variation['coefficient_translation']}")
    print(f"Space between letters coefficient: {configuration_variation['coefficient_space']}")
    print(f"Font variation range: {configuration_variation['variance_font']}")
    print("\n")

    create.set_configuration(configuration_variation)
    for c in tqdm(combinations_variation):
        create.set_attributes(c)
        create.create_images()
    create.split_images_to_folders()

    stats = add_compute_stats(torch_image_folder)(root=str(Path("data") / "data_train")).stats
    stats = {"mean": list(stats["mean"]), "std": list(stats["std"])}
    json.dump(stats, open(Path("data", "normalization_stats.json"), "w"))


def get_position_from_type(position_correction: bool, type: str, target: str, content: str) -> tuple:
    if not position_correction:
        return (0, 0)
    elif type == "SUB3" and content == target[:3]:
        return (-0.415, 0)
    elif type == "SUB3" and content == target[3:]:
        return (0.415, 0)
    elif type == "DL-1F":
        return (-0.3, 0)
    elif type == "DL-2M":
        return (-0.34, 0)
    elif type == "IL-1M":
        return (0.345, 0)
    elif type == "IL-2M":
        return (0.3, 0)
    elif type == "IL-1I":
        return (-0.3, 0)
    elif type == "IL-1F":
        return (0.3, 0)
    else:
        return (0, 0)


def init_create_prime_data(position_correction: bool = False):
    targets: list[str] = read_corpus(Path("assets", "2014-targets.txt"))
    primes: list[str] = read_corpus(Path("assets", "2014-prime-types.txt"))
    prime_data = json.load(open(Path("assets", "2014-prime-data.json"), "r"))

    combinations_target_prime = list(product(*[targets, primes]))

    combinations_target_prime: list[dict] = [
        {
            "target": c[0],
            "prime": c[1],
            "content": [i[c[1]] for i in prime_data if i["ID"] == c[0]][0],
            "font": "arial.ttf",
            "size": 22,
            "position": get_position_from_type(
                position_correction=position_correction,
                type=c[1],
                target=c[0],
                content=[i[c[1]] for i in prime_data if i["ID"] == c[0]][0],
            ),
            "index": 0,
            "name_save": "".join([c[1], ".png"]),
        }
        for c in combinations_target_prime
    ]

    create = CreateData(mode="prime")

    print("\n")
    print("--------------------------------")
    print(f"Total images: {len(combinations_target_prime)}")
    print("--------------------------------")
    print(f"Words: {len(targets)}")
    print("--------------------------------")
    print("\n")

    configuration_variation: dict = {
        "standard_deviation_rotation": 0,  # sd of angle
        "coefficient_translation": 0,  # letter: coefficient * (diagonal len of average letter box in the word)
        "coefficient_space": 0.5,  # space between letters = previous letter width * this coefficient
        "variance_font": 0,  # variance of font size
    }

    create.set_configuration(configuration_variation)
    for c in tqdm(combinations_target_prime):
        create.set_attributes(c)
        create.create_images()


if __name__ == "__main__":
    init_create_train_data()
