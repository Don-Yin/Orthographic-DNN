import json
from pathlib import Path

import torch
import torchvision
from torchvision.datasets import ImageFolder as torch_image_folder

from utils.data_load.device_control import device_allocator
from utils.data_load.normalize import add_compute_stats
from utils.evaluate.evaluate_model import Evaluate

analysis_settings = json.load(open(Path("analysis_settings.json"), "r"))

prime_data = add_compute_stats(torch_image_folder)(
    root=str(Path("data") / analysis_settings["prime_data_folder"]),
    stats=json.load(open(Path("data", "normalization_stats.json"), "r")),
)

alexnet = torchvision.models.alexnet(pretrained=False)
alexnet.load_state_dict(torch.load(Path("params", "alexnet.pth")))
alexnet.eval()

device_allocator(alexnet)

if __name__ == "__main__":
    tensor_1, tensor_2 = Evaluate(
        model=alexnet,
        word="abduct",
        prime_types=["ID", "SUB3"],
        which_layer="penultimate_visualizer",
        prime_data=prime_data,
    ).compute_similarity_layer_wise()

    torchvision.utils.save_image(tensor=tensor_1, fp=Path("tensor_1.png"))
    torchvision.utils.save_image(tensor=tensor_2, fp=Path("tensor_2.png"))

    open("tensor_1.json", "w").write(json.dumps(tensor_1.tolist()))
    open("tensor_2.json", "w").write(json.dumps(tensor_2.tolist()))

    print(tensor_1.shape, tensor_2.shape)
