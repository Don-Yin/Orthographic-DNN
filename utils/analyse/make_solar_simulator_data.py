from pathlib import Path

import pandas
from utils.data_generate.read_corpus import read_corpus


class CreateSolarData:
    def __init__(self):
        self.read_targets()
        self.set_data_2014_to_csv()
        self.read_from_csv()

    def read_targets(self):
        self.targets: list[str] = read_corpus(Path("assets", "2014-targets.txt"))

    def set_data_2014_to_csv(self):
        data = pandas.read_excel(Path("assets", "adelman.xlsx"), sheet_name=2)
        data = data.loc[data["ID"].isin(self.targets)]
        data["Target"] = data["Target"].str.lower()

        data["TCon"] = data["ALD-ARB"]

        data = data[
            [
                "TCon",
                "Target",
                "ID",
                "TL12",
                "TL-I",
                "TL56",
                "NATL2",
                "NATL3",
                "DL-1M",
                "DL-1F",
                "DL-2M",
                "T-All",
                "TH",
                "SUB3",
                "RH",
                "IH",
                "RF",
                "SN-I",
                "SN-M",
                "SN-F",
                "N1R",
                "DSN-M",
                "IL-1M",
                "IL-2M",
                "EL",
                "IL-1I",
                "IL-1F",
                "IL-2MR",
                "ALD-PW",
                "ALD-ARB",
            ]
        ]
        data.to_csv(Path("assets", "data_2014_prime_and_target.csv"), index=False)

    def read_from_csv(self):
        data = open(Path("assets", "data_2014_prime_and_target.csv"), "r").read().replace(",", "	").replace(" ", "")
        open(Path("assets", "data_2014_prime_and_target_for_solar.txt"), "w").write(data)


if __name__ == "__main__":
    CreateSolarData()
