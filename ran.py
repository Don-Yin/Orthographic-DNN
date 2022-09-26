import json
import random
import string
from pathlib import Path


def random_word(length):
    letters = string.ascii_lowercase
    return "".join(random.choice(letters) for i in range(length))


if __name__ == "__main__":
    random_strings = [random_word(6) for _ in range(1000)]
    json.dump(random_strings, open(Path("random_strings.json"), "w"))
