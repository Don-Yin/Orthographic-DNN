from pathlib import Path
from turtle import width

import pandas
import seaborn as sns

from bootstrap import SEs

# data = pandas.read_excel(Path("assets", "data_for_bar_chart.xlsx"), sheet_name=0)

sns.set(style="whitegrid")

# g = sns.catplot(data=data, kind="bar", x="Priming Score (τ)", y="Model", hue="Class", ci="sd", palette="dark", alpha=0.9, height=6, errorbar="sd")
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
}

SEs = pandas.DataFrame(SEs)

# print(sns.color_palette("hls", 1))

g = sns.factorplot(
    order=list(labels.values())
    + [
        "pixCS",
        "Absolute",
        "Spatial Coding",
        "Binary Open Bigram",
        "Overlap Open Bigram",
        "SERIOL Open Bigram",
        "Spatial Coding Model",
        "Interactive Activation Model",
        "LTRS",
        "LevDist",
    ],
    data=SEs,
    x="correlation",
    y="model",
    # col="class",
    # hue="class",
    kind="bar",
    ci="sd",
    orient="h",
    # hue_order=["Convolutional Models", "ViTs"],
    palette=[sns.xkcd_rgb["windows blue"]] * 7 + [sns.xkcd_rgb["amber"]] * 4 + [sns.xkcd_rgb["greyish"]] * 1 + [sns.xkcd_rgb["pale red"]] * 5 + [sns.xkcd_rgb["pale green"]] * 3 + [sns.xkcd_rgb["greyish"]] * 1,
)

g.figure.savefig("test.png", dpi=400)
