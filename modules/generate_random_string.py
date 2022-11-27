"""Generate 1000 random strings of an arbitrary length."""


import json
import random
import string
from pathlib import Path


def random_word(length: int):
    """Generate a random string of fixed length and return it
    params: length (int): Length of the string to generate.
    return: str: Random string of length length.
    """
    letters = string.ascii_lowercase
    return "".join(random.choice(letters) for i in range(length))


if __name__ == "__main__":
    random_strings = [random_word(6) for _ in range(1000)]
    json.dump(random_strings, open(Path("random_strings.json"), "w"))
