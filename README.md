##  Official YOLOv8 训练自己的数据集并基于NVIDIA TensorRT部署

说明： 本项目支持YOLOv8的对应的package的版本是：[ultralytics-8.0.0](https://pypi.org/project/ultralytics/8.0.0/)

### 1. YOLOv8的相关资源

+ YOLOv8 Github: https://github.com/ultralytics/ultralytics
+ YOLOv8文档： https://v8docs.ultralytics.com/

### 2. YOLOv8环境安装

我们使用的是`ultralytics(8.0.0) python package`,其安装方式如下：

```shell
pip install ultralytics==8.0.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

你可以在`/usr/local/lib/pythonx.x/dist-packages/ultralytics `下找到安装包中的YOLOv8的源文件，进行魔改！

### 3. 构建自己训练集的配置文件和模型配置文件

+ 模型配置文件：

```yaml
#yolov8s.yaml
# Parameters
nc: 4  # number of classes
depth_multiple: 0.33  # scales module repeats
width_multiple: 0.50  # scales convolution channels

# YOLOv8.0s backbone
backbone:
  # [from, repeats, module, args]
  - [-1, 1, Conv, [64, 3, 2]]  # 0-P1/2
  - [-1, 1, Conv, [128, 3, 2]]  # 1-P2/4
  - [-1, 3, C2f, [128, True]]
  - [-1, 1, Conv, [256, 3, 2]]  # 3-P3/8
  - [-1, 6, C2f, [256, True]]
  - [-1, 1, Conv, [512, 3, 2]]  # 5-P4/16
  - [-1, 6, C2f, [512, True]]
  - [-1, 1, Conv, [1024, 3, 2]]  # 7-P5/32
  - [-1, 3, C2f, [1024, True]]
  - [-1, 1, SPPF, [1024, 5]]  # 9

# YOLOv8.0s head
head:
  - [-1, 1, nn.Upsample, [None, 2, 'nearest']]
  - [[-1, 6], 1, Concat, [1]]  # cat backbone P4
  - [-1, 3, C2f, [512]]  # 13

  - [-1, 1, nn.Upsample, [None, 2, 'nearest']]
  - [[-1, 4], 1, Concat, [1]]  # cat backbone P3
  - [-1, 3, C2f, [256]]  # 17 (P3/8-small)

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 12], 1, Concat, [1]]  # cat head P4
  - [-1, 3, C2f, [512]]  # 20 (P4/16-medium)

  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 9], 1, Concat, [1]]  # cat head P5
  - [-1, 3, C2f, [1024]]  # 23 (P5/32-large)

  - [[15, 18, 21], 1, Detect, [nc]]  # Detect(P3, P4, P5)

```

+ 数据集配置文件

```yaml
#score_data.yaml

# Train/val/test sets as 1) dir: path/to/imgs, 2) file: path/to/imgs.txt, or 3) list: [path/to/imgs1, path/to/imgs2, ..]
train: ./dataset/score/images/train # train images
val: ./dataset/score/images/val # val images
#test: ./dataset/score/images/test # test images (optional)

# Classes
names:
  0: person
  1: cat
  2: dog
  3: horse

```

+ 训练超参数配置文件

我们对训练的超参数进行了简单的修改，通过命令行参数传入，也可以通过配置文件进行配置。

```yaml
task: "detect" # choices=['detect', 'segment', 'classify', 'init'] # init is a special case. Specify task to run.
mode: "train" # choices=['train', 'val', 'predict'] # mode to run task in.

# Train settings -------------------------------------------------------------------------------------------------------
model: null # i.e. yolov8n.pt, yolov8n.yaml. Path to model file
data: null # i.e. coco128.yaml. Path to data file
epochs: 100 # number of epochs to train for
patience: 50  # TODO: epochs to wait for no observable improvement for early stopping of training
batch: 16 # number of images per batch
imgsz: 640 # size of input images
save: True # save checkpoints
cache: False # True/ram, disk or False. Use cache for data loading
device: '' # cuda device, i.e. 0 or 0,1,2,3 or cpu. Device to run on
workers: 8 # number of worker threads for data loading
project: null # project name
name: null # experiment name
exist_ok: False # whether to overwrite existing experiment
pretrained: False # whether to use a pretrained model
optimizer: 'SGD' # optimizer to use, choices=['SGD', 'Adam', 'AdamW', 'RMSProp']
...
```

### 4. YOLOv8目标检测任务训练
默认使用多卡, 单卡添加 device=0

```shell
yolo task=detect mode=train model=yolov8s.yaml data=yolo_data.yaml epochs=100 batch=64
```
or :
```shell
yolo task=detect mode=train model=ckpt/yolov8n.pt data=yolo_data.yaml epochs=100 batch=64

```

![](docs/train_log1.png)

### 5. YOLOv8推理Demo

```shell
yolo task=detect mode=predict model=runs/detect/train/weights/best.pt source=datasets/val/images/1.jpg
```
或
```shell
# 自己实现的推断程序
python3 inference.py
```
| ![](docs/cat1.jpg)   |

### 6. YOLOv8端到端模TensorRT模型加速

1. pth模型转onnx

```shell
#CLI
yolo task=detect mode=export model=./runs/detect/train/weights/last.pt format=onnx simplify=True opset=13

# python
from ultralytics import YOLO

model = YOLO("./runs/detect/train/weights/last.pt ")  # load a pretrained YOLOv8n model
model.export(format="onnx")  # export the model to ONNX format
```

2. 增加NMS Plugin 

执行`tensorrt/`下的如下代码，添加NMS到YOLOv8模型

+ 添加后处理

```shell
python3 yolov8_add_postprocess.py
```

+ 添加NMS plugin

```shell
python3 yolov8_add_nms.py
```

生成`last_1_nms.onnx`,打开该文件对比和原onnx文件的区别，发现增加了如下节点(完成了将NMS添加到onnx的目的）：

![](docs/nms.png)

3. onnx转trt engine

```shell
trtexec --onnx=last_1_nms.onnx --saveEngine=yolov8s.plan --workspace=3000 --verbose
```

![](docs/trt.png)

出现上述界面，onnx正常序列化为TRT engine.

4. TRT C++推断

在win 10下基于RTX 1060 TensorRT 8.2.1进行测试，我们的开发环境是VS2017,**所有C++代码已经存放在`tensorrt/`文件夹下**。其推断结果如下图所示（可以发现我们实现了YOLOv8的TensorRT端到端的推断，其推断结果与原训练框架保持一致）：

| ![](tensorrt/yolov8/yolov8/res/cat1.jpg)   |

### 参考文献：

+ https://github.com/ultralytics/ultralytics

+ https://mp.weixin.qq.com/s/_OvSTQZlb5jKti0JnIy0tQ
+ https://github.com/ultralytics/assets/releases
+ https://v8docs.ultralytics.com/
+ https://pypi.org/project/ultralytics/0.0.44/#description
+ https://mp.weixin.qq.com/s/-4pn--3kFI_J1oX6p5GWVQ
+ https://github.com/uyolo1314/ultralytics



