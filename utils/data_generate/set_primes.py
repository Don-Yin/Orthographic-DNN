import json
from pathlib import Path

import pandas


class SetPrimes:
    def __init__(self):
        self.path_output_prime_types = Path("assets", "2014-prime-types.txt")
        self.path_output_prime_data = Path("assets", "2014-prime-data.json")
        self.read_data_2014()
        self.read_prime_types()
        self.save()

    def read_data_2014(self):
        self.data_2014 = pandas.read_excel(Path("assets", "adelman.xlsx"), sheet_name=2)
        self.data_2014 = self.data_2014.rename(columns={"Target": "target"})
        self.prime_data = self.data_2014.to_dict("records")

    def read_prime_types(self):
        self.prime_types = list(self.data_2014.columns)[1::]

    def save(self):
        with open(self.path_output_prime_types, "w") as writer:
            writer.write("\n".join(self.prime_types))

        json.dump(self.prime_data, open(self.path_output_prime_data, "w"))


if __name__ == "__main__":
    SetPrimes()
