import json
from pathlib import Path

import torch
import torchvision


class DepictModels:
    def __init__(self):
        self.analysis_settings = json.load(open(Path("analysis_settings.json"), "r"))
        self.read_selected_models()
        self.init_models()
        self.main()

    def main(self):
        for model_name in self.selected_network_names:
            self.generate_onnx(model_name)

    def read_selected_models(self):
        self.selected_network_names = json.load(open(Path("assets", "selected_models.json"), "r"))

    def init_models(self):
        self.alexnet = torchvision.models.alexnet(pretrained=False)
        self.densenet169 = torchvision.models.densenet169(pretrained=False)
        self.efficientnet_b1 = torchvision.models.efficientnet_b1(pretrained=False)
        self.resnet50 = torchvision.models.resnet50(pretrained=False)
        self.resnet101 = torchvision.models.resnet101(pretrained=False)
        self.vgg16 = torchvision.models.vgg16(pretrained=False)
        self.vgg19 = torchvision.models.vgg19(pretrained=False)
        self.vit_l_16 = torchvision.models.vit_l_16(pretrained=False)

    def generate_onnx(self, model_name):
        input_names = ["actual_input_1"] + ["learned_%d" % i for i in range(16)]
        output_names = ["output1"]

        torch.onnx.export(
            getattr(self, model_name),
            torch.randn(10, 3, 224, 224, device="cpu"),
            Path("results", self.analysis_settings["result"], "model_architectures", f"{model_name}.onnx"),
            verbose=True,
            input_names=input_names,
            output_names=output_names,
        )


if __name__ == "__main__":
    DepictModels()
