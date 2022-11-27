import os
import pathlib
import pickle
from pathlib import Path
from time import time

import numpy as np
import torch
import torchvision
import torchvision.transforms as tf
from PIL import ImageStat
from sty import bg, ef, fg, rs




def compute_mean_and_std_from_dataset(dataset, dataset_path=None, max_iteration=100, data_loader=None, verbose=True):
    if max_iteration < 30:
        print("Max Iteration in Compute Mean and Std for dataset is lower than 30! This could create unrepresentative stats!") if verbose else None
    start = time()
    stats = {}
    transform_save = dataset.transform
    if data_loader is None:
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)

    statistics = None
    c = 0
    stop = False
    for data, _ in data_loader:
        for b in range(data.shape[0]):
            if c % 10 == 9 and verbose:
                print(f"{c}/{max_iteration}, m: {np.array(statistics.mean)/255}, std: {np.array(statistics.stddev)/255}")
            c += 1
            if statistics is None:
                statistics = Stats(tf.ToPILImage()(data[b]))
            else:
                statistics += Stats(tf.ToPILImage()(data[b]))
            if c > max_iteration:
                stop = True
                break
        if stop:
            break

    stats["time_one_iter"] = (time() - start) / max_iteration
    stats["mean"] = np.array(statistics.mean) / 255
    stats["std"] = np.array(statistics.stddev) / 255
    stats["iter"] = max_iteration
    print((fg.cyan + "mean: {}; std: {}, time: {}" + rs.fg).format(stats["mean"], stats["std"], stats["time_one_iter"])) if verbose else None
    if dataset_path is not None:
        print("Saving in {}".format(dataset_path))
        with open(dataset_path, "wb") as f:
            pickle.dump(stats, f)

    dataset.transform = transform_save
    return stats


class Stats(ImageStat.Stat):
    def __add__(self, other):
        return Stats(list(map(np.add, self.h, other.h)))

def add_compute_stats(obj_class):
    # global ComputeStatsUpdateTransform

    class ComputeStatsUpdateTransform(obj_class):
        ## This class basically is used for normalize Dataset Objects such as ImageFolder in order to be used in our more general framework
        def __init__(
            self,
            name_generator="dataset",
            add_PIL_transforms=None,  ## if resize add here
            add_tensor_transforms=None,
            num_image_calculate_mean_std=70,
            stats=None,
            save_stats_file=None,
            **kwargs,
        ):
            """

            @param add_tensor_transforms:
            @param stats: this can be a dict (previous stats, which will contain 'mean': [x, y, z] and 'std': [w, v, u], a path to a pickle file, or None
            @param save_stats_file:
            @param kwargs:
            """
            self.verbose = True
            print(fg.yellow + "\n**Creating Dataset [" + fg.cyan + f"{name_generator}" + fg.yellow + "]**" + rs.fg)
            super().__init__(**kwargs)
            self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

            if add_PIL_transforms is None:
                add_PIL_transforms = []
            if add_tensor_transforms is None:
                add_tensor_transforms = []

            self.transform = torchvision.transforms.Compose([*add_PIL_transforms, torchvision.transforms.ToTensor(), *add_tensor_transforms])

            self.name_generator = name_generator
            self.additional_transform = add_PIL_transforms
            self.num_image_calculate_mean_std = num_image_calculate_mean_std
            self.num_classes = len(self.classes)

            compute_stats = False

            if isinstance(stats, dict):
                self.stats = stats
                print(fg.red + "Using precomputed stats: " + fg.cyan + f"mean = {self.stats['mean']}, std = {self.stats['std']}" + rs.fg)

            elif isinstance(stats, str):
                if os.path.isfile(stats):
                    self.stats = pickle.load(open(stats, "rb"))
                    print(
                        fg.red
                        + f"Using stats from file [{Path(stats).name}]: "
                        + fg.cyan
                        + f"mean = {self.stats['mean']}, std = {self.stats['std']}"
                        + rs.fg
                    )
                    if stats == save_stats_file:
                        save_stats_file = None
                else:
                    print(fg.red + f"File [{Path(stats).name}] not found, stats will be computed." + rs.fg)
                    compute_stats = True

            if stats is None or compute_stats is True:
                self.stats = self.call_compute_stats()

            if save_stats_file is not None:
                print(f"Stats saved in {save_stats_file}")
                pathlib.Path(os.path.dirname(save_stats_file)).mkdir(parents=True, exist_ok=True)
                pickle.dump(self.stats, open(save_stats_file, "wb"))

            normalize = torchvision.transforms.Normalize(mean=self.stats["mean"], std=self.stats["std"])
            self.transform.transforms += [normalize]
            print(f"Map class_name -> labels: {self.class_to_idx}\n{len(self)} samples.") if self.verbose else None

        def call_compute_stats(self):
            return compute_mean_and_std_from_dataset(self, None, max_iteration=self.num_image_calculate_mean_std, verbose=self.verbose)

    return ComputeStatsUpdateTransform


if __name__ == "__main__":
    from subset import MyImageFolder

    data_train = MyImageFolder(root=str(Path("data") / "data_all"), name_classes=["and"])
    print(data_train)
