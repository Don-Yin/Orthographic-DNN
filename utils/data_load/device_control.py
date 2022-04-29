import torch


class DeviceAllocator:
    def __init__(self):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def to_device(self, data):
        if isinstance(data, (list, tuple)):
            return [self.to_device(i) for i in data]
        # print(f"Loading data to device: {self.device}")
        return data.to(self.device, non_blocking=True)


def device_allocator(data):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if isinstance(data, (list, tuple)):
        return [device_allocator(i) for i in data]
    # print(f"Loading data to device: {device}")
    return data.to(device, non_blocking=True)


class DeviceDataLoader:
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def to_device(self, data):
        if isinstance(data, (list, tuple)):
            return [self.to_device(i) for i in data]
        # print(f"Loading data to device: {self.device}")
        return data.to(self.device, non_blocking=True)

    def __iter__(self):
        for i in self.data_loader:  # here
            yield self.to_device(i)

    def __len__(self):
        return len(self.data_loader)
