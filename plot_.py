from pathlib import Path
from turtle import width

import pandas
import seaborn as sns

data = pandas.read_excel(Path("assets", "data_for_bar_chart.xlsx"), sheet_name=0)


sns.set(style="whitegrid")

# g = sns.barplot(x="name", y="priming score", hue="class", data=data)

g = sns.catplot(
    data=data, kind="bar", x="Priming Score (τ)", y="Model", hue="Class", ci="sd", palette="dark", alpha=0.6, height=6
)

g.figure.savefig("test.png", dpi=400)

print(data)
