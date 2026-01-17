# SittingWatch

## Introduction

This repository contains a **YOLOv8-based model** for sitting pose detection, along with a simple **web backend** for [hunckbackDetect](https://github.com/ying-2626/hunckbackDetect). We also provide scripts for building the dataset.

For sitting pose detection, we first constructed **a brand new dataset** through **knowledge distillation** and **fine-tuning**. Specifically, we used [Google Mediapipe](https://github.com/google-ai-edge/mediapipe) to collect 300 images annotated with 3D keypoints. We then leveraged [hunckbackDetect](https://github.com/ying-2626/hunckbackDetect) to label each image as either `sitting_bad` or `sitting_good`. Since some images were incorrectly processed due to partial occlusion, we used `X-AnyLabeling` to manually correct the annotations. Then we **augmented** the data with **18 basic DIP methods**. Finally we trained a YOLOv8-based model using this dataset.

For the backend, we designed a simple structure that takes a **JPG image** as input and outputs the **sitting pose classification** along with its **confidence score**, to support [hunckbackDetect](https://github.com/ying-2626/hunckbackDetect).

**NOTE:** The directory `YOLOv8` was downloaded from `https://github.com/ultralytics/ultralytics`, which was then modified slightly.

## Project Structure

The structure of the project is described as follows.

```bash
.
├── buildDataset    # Scripts for building the dataset
├── image           # Static information for demonstration
├── logs            # Records of the backend service
├── README.md
├── runs            # Checkpoints of the pre-trained models
├── server          # Scripts for backend service
├── test            # Deprecated
├── YOLOv8          # Scripts for training, majorly derived from Ultralytics
└── yolov8n.pt      # Necessary dependency
```

## Installation Guide

First, create a virtual environment using Anaconda (recommended).

```bash
conda create -n sitting-watch python=3.10 -y
conda activate sitting-watch
```

Second, install PyTorch that suits your own device through `https://pytorch.org/get-started/locally/`. The code has been tested on version `2.7.1`.

Third, install the dependencies using the following commands.

```bash
cd path/to/SittingWatch
pip install -r requirements.txt
```

To use YOLO for inference, install `ultralytics` through `pip`.

```bash
pip install ultralytics
```

Also install `flask` and `gunicorn` for backend.

```bash
pip install flask gunicorn
```

## Start Backend Service

Use the following commands to start the backend service.

```bash
cd server
gunicorn --workers 1 --threads 1 --bind 0.0.0.0:6666 app:app
```

To verify the server, try the following command.

```bash
curl -X POST -F "image=@path/to/your/test.jpg" http://localhost:6666/detect
```

The result should be something like below.

```bash
{"class":"sitting_bad","conf":"0.7605"}
```

The server has been tested for [hunckbackDetect](https://github.com/ying-2626/hunckbackDetect).

## Further Details

### Testing Environment

The project has been tested in the following environment.

| Item | Detail |
| ---- | ---- |
| Platform | HP OMEN 17 2022 |
| OS | Ubuntu 22.04 LTS |
| CPU | Intel Core i7-12800HX |
| GPU | NVIDIA GeForce RTX 3080 Ti Laptop (16GB) |
| CUDA | 11.8 |
| Memory | 32GB RAM |
| Disk | 2TB SSD |

### API References

**POST** `/detect`

**Description**: Detects posture in a JPG image using the YOLOv8 model and returns classification and confidence.

| **Field**          | **Description**                                                                 |
|--------------------|---------------------------------------------------------------------------------|
| **Method**         | POST                                                                           |
| **URL**            | `/detect`                                                                      |
| **Request Body**   | `image`: JPG image file                                                        |
| **Status Codes**   | - 200: Success<br>- 400: Invalid input (missing image or unsupported format)    |
| **Response Body**  | ```json<br>{<br>  "class": "sitting_bad" \| "sitting_good" \| "busy" \| "nan",<br>  "conf": "<float between 0 and 1, formatted to .4f, 0.0 for 'nan' or 'busy'>"<br>}<br>``` |
| **Error Response** | ```json<br>{<br>  "error": "<error message>"<br>}<br>```                       |
| **Notes**          | - Processes one inference at a time using a queue.<br>- Returns `{"class": "busy", "conf": "0.0"}` if queue exceeds 50 requests.<br>- Saves annotated images in `/home/whs/PoseDetection/SittingWatch/logs/{date}` as `annotated_{HHMMSS}.jpg`. |

### Inference without Backend

Try the following commands.

```bash
cd server
python inference-test.py
```

### Data Preparation

Suppose that you have got a dataset in YOLO format. You can use the following commands to augment the data.

```bash
cd buildDataset
python augment_data.py
```

Detailed information of the **18 basic DIP methods** are listed below.

- Horizontal flip  
- Rotate +10°  
- Rotate -10°  
- Rotate +20°  
- Rotate -20°  
- Rotate +30°  
- Rotate -30°  
- Gaussian blur  
- Brightness increase (level 1)  
- Brightness increase (level 2)  
- Brightness decrease (level 1)  
- Brightness decrease (level 2)  
- Contrast increase (level 1)  
- Contrast increase (level 2)  
- Contrast decrease (level 1)  
- Contrast decrease (level 2)  
- Sharpening (level 1)  
- Sharpening (level 2)

### Training Guide

First, enter the `YOLOv8` directory.

```bash
cd YOLOv8
```

Second, place the datasets in `path/to/SittingWatch/YOLOv8/main/datasets/sitting_pose` and adjust `data.yaml` according to your local environment. Mind that the dataset should be in YOLO format, which is described below.

```
.
└── sitting_pose
    ├── data.yaml
    ├── test
    │   ├── images
    │   └── labels
    ├── train
    │   ├── images
    │   ├── labels
    └── valid
        ├── images
        └── labels
```

Then, select a set of appropriate hyperparameter and start training.

```bash
yolo detect train data=main/datasets/sitting_pose/data.yaml model=yolov8n.yaml pretrained=yolov8n.pt epochs=1000 batch=64 lr0=0.0001
```

The result should be something like below.

![training_example](image/training_example.png)

