import json
from pathlib import Path

import pandas

analysis_settings = json.load(open(Path("analysis_settings.json"), "r"))
error_dict = json.load(open(Path("assets", "label_error.json"), "r"))
sorter = json.load(open(Path("assets", "sorter_human_data.json"), "r"))
duration = 2000


def read_data_SOLAR():
    data = open(Path("assets", "scm", f"SOLAR_result_target_duration_{duration}.txt"), "r").read()
    initial_lines = data.split("\n")[:2]
    initial_lines = [i.split("\t") for i in initial_lines]
    initial_lines = [[i for i in line if i] for line in initial_lines]

    for i in range(len(initial_lines[1])):
        if initial_lines[1][i] in error_dict.keys():
            initial_lines[1][i] = error_dict[initial_lines[1][i]]

    lines = [i.split("\t") for i in data.split("\n")[2:]]
    lines = [[i for i in line if i] for line in lines]
    lines = [i for i in lines if i]
    lines = [[int(i.strip("No")) if not i.isalpha() else i for i in line] for line in lines]
    for i in range(len(lines)):
        lines[i] = dict(zip(initial_lines[1], lines[i]))
        lines[i].pop("Prime", None)
        lines[i].pop("Target", None)

    content = []
    for line in lines:
        for key, value in line.items():
            content.append({"Primes": key, "Predicted RT": value})

    json.dump(content, open(Path("SOLAR_result.json"), "w"))



def read_data_IA():
    data = open(Path("assets", "ia", f"ia_target_duration_{duration}.txt"), "r").read()
    initial_lines = data.split("\n")[:2]
    initial_lines = [i.split("\t") for i in initial_lines]
    initial_lines = [[i for i in line if i] for line in initial_lines]

    for i in range(len(initial_lines[1])):
        if initial_lines[1][i] in error_dict.keys():
            initial_lines[1][i] = error_dict[initial_lines[1][i]]

    lines = [i.split("\t") for i in data.split("\n")[2:]]
    lines = [[i for i in line if i] for line in lines]
    lines = [i for i in lines if i]
    lines = [[int(i.strip("No")) if not i.isalpha() else i for i in line] for line in lines]
    for i in range(len(lines)):
        lines[i] = dict(zip(initial_lines[1], lines[i]))
        lines[i].pop("Prime", None)
        lines[i].pop("Target", None)

    content = []
    for line in lines:
        for key, value in line.items():
            content.append({"Primes": key, "Predicted RT": value})
    
    json.dump(content, open(Path("IA_result.json"), "w"))


if __name__ == "__main__":
    read_data_IA()
    # data = json.load(open(Path("SOLAR_result.json"), "r"))
    # data = 
    # data['Primes'] = data['Primes'].apply(lambda x: sorter[x])
