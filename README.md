# Orthographic-DNN
Project objective: comparing human orthographic perception with visual DNN models.
Fonts are in assets/fonts.

python==3.10.4


## Full priming models to add
![LTRS](http://www.adelmanlab.org/ltrs/)
![IA](http://www.pc.rhul.ac.uk/staff/c.davis/SpatialCodingModel/)
![BayesianReader](https://www.mrc-cbu.cam.ac.uk/personal/dennis.norris/personal/BayesianReader/)
(https://www.mrc-cbu.cam.ac.uk/people/dennis.norris/personal/BayesianReader/documentation/)

---

## Install cuda + pytorch
1. install cuda driver
2. correct version of pytorch on pytorch.org - >= torch-1.11.0

## Run on server
1. build docker image: 
    tmux new -s {session name}
    tmux attach-session
    tmux attach-session -t {session name}
    ```docker build -t=don_vgg:0.1 .```
2. show image list in docker
    ```docker image ls```
3. double check mount volumes in run_docker.sh
4. run docker image
    create a new tmux session
    ```chmod +x run_docker.sh```

## tmux
previous session: tmux a
detach previous session: control b (pause) d

## Server
check current folder size: 
```du -sh .```

check free space available in current directory:
```df -Ph . | tail -1 | awk '{print $4}'```

check number of files in a folder
```find ~/Orthographic-DNN//data -type f | wc -l```

Once your model is running, run `nvidia-smi -lms 100` to check that the GPU is getting used

## Load validation data.
```python
    data_valid = add_compute_stats(torch_image_folder)(root=str(Path("data") / "data_valid"))
    data_valid_batch_loader = data_loader(data_valid, self.size_batch * 2, num_workers=4, pin_memory=True)
    data_valid_batch_loader = DeviceDataLoader(data_valid_batch_loader)
```

## Init neptune
```python
  self.neptune_run["parameters"] = {
      "function_optimizer": self.configuration["function_optimizer"],
      "rate_learning": self.configuration["rate_learning"],
      "function_loss": self.configuration["function_loss"],
      "size_batch": self.configuration["size_batch"],
      "num_epochs": self.configuration["num_epochs"],
  }
```

## change number of classes
```python
vit_l_16.heads.head = nn.Linear(vit_l_16.heads.head.in_features, 1000)
resnet.fc = nn.Linear(resnet.fc.in_features, 1000)
self.vgg19.classifier[-1] = nn.Linear(vgg19.classifier[-1].in_features, 5000)
```
