import gc
import json
from pathlib import Path

import neptune.new as neptune
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


class Train:
    def __init__(self, hyperparameters: dict):
        self.hyperparameters = hyperparameters
        self.path_data: Path = Path("data") / "data_train"
        # self.memory_tracker = SummaryTracker()  # track memory leak
        routine_settings()

    def fit(self):
        self.load_data()
        self.init_neptune()
        self.set_batch_loaders()
        self.set_model()

        optimizer = self.hyperparameters["function_optimizer"](
            params=self.model.parameters(),
            lr=self.hyperparameters["rate_learning"],
            weight_decay=self.hyperparameters["weight_decay"],
        )

        moving_average = ExpMovingAverage(0)

        for epoch in range(self.hyperparameters["num_epochs"]):
            print(f"Epoch: {epoch}")
            self.model.train()
            train_losses_log = []
            batch_num: int = 0
            for batch in self.data_train_batch_loader:
                if batch_num % self.hyperparameters["batch_report_every"] == 0:
                    print(f"Reporting batch: {batch_num}")
                    train_loss_and_accuracy = self.get_batch_loss_and_accuracy(batch)
                    self.neptune_run["train/accuracy_top_1"].log(train_loss_and_accuracy["batch_accuracy_top_1"])
                    self.neptune_run["train/accuracy_top_5"].log(train_loss_and_accuracy["batch_accuracy_top_5"])
                    self.neptune_run["train/loss"].log(train_loss_and_accuracy["batch_loss"])

                    test_loss_and_accuracy = self.get_batch_loss_and_accuracy(next(iter(self.data_test_batch_loader)))
                    self.neptune_run["test/accuracy_top_1"].log(test_loss_and_accuracy["batch_accuracy_top_1"])
                    self.neptune_run["test/accuracy_top_5"].log(test_loss_and_accuracy["batch_accuracy_top_5"])
                    self.neptune_run["test/loss"].log(test_loss_and_accuracy["batch_loss"])

                    moving_average(test_loss_and_accuracy["batch_loss"])
                    self.neptune_run["test/loss_moving_average"].log(moving_average.avg)

                    gc.collect(generation=2)  # collect garbage

                    if moving_average.avg < self.hyperparameters["min_moving_average_threshold"]:
                        self.save_params()
                        self.neptune_run.stop()
                        raise Exception(f"Training finished: {self.hyperparameters['model_tag']}")

                batch_num += 1
                loss = self.get_train_loss(batch)
                train_losses_log.append(loss)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                if batch_num % self.hyperparameters["batch_save_every"] == 0:
                    # self.memory_tracker.print_diff()
                    self.save_params()
                    print(f"Params saved: {self.hyperparameters['model_tag']}")
                    batch_num = 0

            summary_epoch = self.get_logger_test_loss_and_accuracy(self.data_test_batch_loader)
            summary_epoch["train_loss"] = torch.stack(train_losses_log).mean().detach().item()
            self.neptune_run["epoch/accuracy_top_1"].log(summary_epoch["test_accuracy_top_1"])
            self.neptune_run["epoch/loss"].log(summary_epoch["test_loss"])
            self.print_epoch(epoch, summary_epoch)
        self.neptune_run.stop()

    def read_list_corpus(self):
        corpus: str = open(Path("assets", "1000-corpus.txt"), "r").read()
        return [w for w in corpus.split("\n") if w != ""]

    def load_data(self, whether_normalize: bool = True, subset: tuple = False):
        normalization_stats = json.load(open(Path("data", "normalization_stats.json"), "r"))

        if subset and whether_normalize:
            corpus: list[str] = self.read_list_corpus()
            classes = corpus[subset[0] : subset[1]]
            self.data = add_compute_stats(MyImageFolder)(
                root=str(self.path_data), name_classes=classes, stats=normalization_stats
            )
        elif subset and not whether_normalize:
            corpus: list[str] = self.read_list_corpus()
            classes = corpus[subset[0] : subset[1]]
            self.data = MyImageFolder(root=str(self.path_data), name_classes=classes)
        else:
            self.data = add_compute_stats(torch_image_folder)(
                root=str(Path("data") / "data_train"), stats=normalization_stats
            )

    def init_neptune(self):        
        neptune_token = json.load(open(Path("assets", "neptune_token.json"), "r"))
        self.neptune_run = neptune.init(project=neptune_token["project"], api_token=neptune_token["api_token"],)
        self.neptune_run["hyperparameters"] = self.hyperparameters

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
            data_loader(
                data_train, self.hyperparameters["size_batch"], shuffle=True, num_workers=num_workers, pin_memory=True
            )
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

    def get_validation_accuracy(self, batch):
        images, labels = batch
        outputs = self.model(images)
        _, predictions_top_1 = torch.max(outputs, dim=1)
        accuracy_top_1 = torch.tensor(torch.sum(predictions_top_1 == labels).detach().item() / len(predictions_top_1))
        return accuracy_top_1

    def get_logger_test_loss_and_accuracy(self, loader_test_data):
        self.model.eval()
        log_validation_loss = [self.get_batch_loss_and_accuracy(batch) for batch in loader_test_data]
        epochLoss = torch.stack([i["batch_loss"] for i in log_validation_loss]).mean()  # Combine losses
        epochAccu = torch.stack([i["batch_accuracy_top_1"] for i in log_validation_loss]).mean()  # Combine accuracies
        self.model.train()
        return {"test_loss": epochLoss.detach().item(), "test_accuracy_top_1": epochAccu.detach().item()}

    def print_epoch(self, epoch, result):
        print(
            "Epoch [{}], train_loss: {:.4f}, test_loss: {:.4f}, test_accuracy_top_1: {:.4f}".format(
                epoch, result["train_loss"], result["test_loss"], result["test_accuracy_top_1"]
            )
        )


if __name__ == "__main__":
    # model_summary(pnasnet, input_size=(3, 224, 224))
    pass
