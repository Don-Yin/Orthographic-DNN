import gc
import json
from pathlib import Path

import torch
from torch.utils.data import random_split
from torch.utils.data.dataloader import DataLoader as data_loader
from torchsummary import summary as model_summary
from torchvision.datasets import ImageFolder as torch_image_folder
from utils.data_load.device_control import DeviceDataLoader, device_allocator
from utils.data_load.normalize import add_compute_stats
from utils.data_load.subset import MyImageFolder
from utils.routine import routine_settings
from utils.train.converging_average import ExpMovingAverage

# added detach() to prevent memory leak. Remove if errors arises


class LoopCounter:
    """
    Batch loop counter
    """

    def __init__(self, start=0):
        self.count = start

    def __call__(self):
        self.count += 1
        return self.count

    def clear(self):
        self.count = 0


class Train:
    def __init__(self, hyperparameters: dict):
        self.hyperparameters = hyperparameters
        self.path_data: Path = Path("data") / "data_train"
        routine_settings()

    def fit(self):
        self.load_data()
        self.set_batch_loaders()
        self.set_model()

        optimizer = self.hyperparameters["function_optimizer"](
            params=self.model.parameters(),
            lr=self.hyperparameters["rate_learning"],
            weight_decay=self.hyperparameters["weight_decay"],
        )

        moving_average = ExpMovingAverage(0)
        batch_loop_counter = LoopCounter(0)

        for epoch in range(self.hyperparameters["num_epochs"]):
            print(f"Epoch: {epoch}")
            self.model.train()
            batch_loop_counter.clear()
            for batch in self.data_train_batch_loader:
                if batch_loop_counter.count % self.hyperparameters["batch_report_every"] == 0:
                    print(f"Reporting batch: {batch_loop_counter.count}")
                    test_loss_and_accuracy = self.get_batch_loss_and_accuracy(next(iter(self.data_test_batch_loader)))
                    moving_average(test_loss_and_accuracy["batch_loss"])
                    gc.collect(generation=2)  # collect garbage

                    if moving_average.avg < self.hyperparameters["min_moving_average_threshold"]:
                        self.save_params()
                        raise ValueError(f"Training finished: {self.hyperparameters['model_tag']}")

                batch_loop_counter()
                loss = self.get_train_loss(batch)
                # train_losses_log.append(loss)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                if batch_loop_counter.count % self.hyperparameters["batch_save_every"] == 0:
                    # self.memory_tracker.print_diff()
                    self.save_params()
                    print(f"Params saved: {self.hyperparameters['model_tag']}")
                    batch_loop_counter.clear()

    def read_list_corpus(self):
        corpus: str = open(Path("assets", "1000-corpus.txt"), "r").read()
        return [w for w in corpus.split("\n") if w != ""]

    def load_data(self, whether_normalize: bool = True, subset: tuple = False):
        try:
            normalization_stats = json.load(open(Path("data", "normalization_stats.json"), "r"))
        except FileNotFoundError:
            normalization_stats = None

        if subset and whether_normalize:
            corpus: list[str] = self.read_list_corpus()
            classes = corpus[subset[0] : subset[1]]
            self.data = add_compute_stats(MyImageFolder)(root=str(self.path_data), name_classes=classes, stats=normalization_stats)
        elif subset and not whether_normalize:
            corpus: list[str] = self.read_list_corpus()
            classes = corpus[subset[0] : subset[1]]
            self.data = MyImageFolder(root=str(self.path_data), name_classes=classes)
        else:
            # when not using add_compute_stats, the data has to go through transforms.Compose([transforms.Resize(), other transforms stuff]), which has to be passed as one of the parameters as torch_image_folder(transform=transforms.Compose([transforms.Resize(), other transforms stuff]))
            self.data = add_compute_stats(torch_image_folder)(root=str(Path("data") / "data_train"), stats=normalization_stats)

    def set_batch_loaders(self):
        data_train, data_test = random_split(
            self.data,
            [
                int(len(self.data) * self.hyperparameters["ratio_train_test"][0]),
                int(len(self.data) * self.hyperparameters["ratio_train_test"][1]),
            ],
        )

        num_workers = 0  # cpu is somewhat not compatable with multiprocessing while using add_compute_stats

        self.data_train_batch_loader = DeviceDataLoader(
            data_loader(data_train, self.hyperparameters["size_batch"], shuffle=True, num_workers=num_workers, pin_memory=True)
        )

        self.data_test_batch_loader = DeviceDataLoader(
            data_loader(data_test, self.hyperparameters["size_batch"] * 2, num_workers=num_workers, pin_memory=True)
        )

    def set_model(self):
        self.model = self.hyperparameters["model"]
        device_allocator(self.model)

    def save_params(self):
        torch.save(self.model.state_dict(), f=Path("params", f"{self.hyperparameters['model_tag']}.pth"))

    def get_train_loss(self, batch):
        images, labels = batch
        output = self.model(images)
        loss = self.hyperparameters["function_loss"](output, labels)
        return loss

    def get_batch_loss_and_accuracy(self, batch):
        images, labels = batch
        outputs = self.model(images)
        loss = self.hyperparameters["function_loss"](outputs, labels)
        _, predictions_top_1 = torch.max(outputs, dim=1)
        _, predictions_top_5 = torch.topk(outputs, k=5, dim=1)
        predictions_top_5 = predictions_top_5.t()
        accuracy_top_1 = torch.tensor(torch.sum(predictions_top_1 == labels).detach().item() / len(labels))
        accuracy_top_5 = torch.tensor(torch.sum(torch.eq(predictions_top_5, labels).t()).detach().item() / len(labels))
        return {"batch_loss": loss.detach(), "batch_accuracy_top_1": accuracy_top_1, "batch_accuracy_top_5": accuracy_top_5}


if __name__ == "__main__":
    pass
