import os

import numpy as np
from sty import bg, ef, fg, rs
from torchvision.datasets import ImageFolder


class MyImageFolder(ImageFolder):
    """Selectively read training data:

    data_train = MyImageFolder(root=str(Path("data") / "data_train"), name_classes=corpus[:10])

    Args:
        ImageFolder (ImageFolder): ImageFolder class from torch vision
    """

    def __init__(self, name_classes=None, verbose=True, *args, **kwargs):
        print(fg.red + ef.inverse + "ROOT:  " + kwargs["root"] + rs.inverse + rs.fg)
        self.name_classes = np.sort(name_classes)
        self.verbose = verbose
        super().__init__(*args, **kwargs)

    def find_classes(self, dir: str):
        if self.name_classes is None:
            return super().find_classes(dir)
        else:
            classes = [d.name for d in os.scandir(dir) if d.is_dir() and (d.name in self.name_classes)]
            classes.sort()
            class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target
