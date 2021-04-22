<a href="https://apps.apple.com/app/id1452689527" target="_blank">
<img src="https://user-images.githubusercontent.com/26833433/85940594-2d3f7d80-b8d2-11ea-809a-87b3bf6d968b.jpg" width="1000"></a>
&nbsp

This repo contains Ultralytics inference and training code for YOLOv3 in PyTorch. The code works on Linux, MacOS and Windows. Credit to Joseph Redmon for YOLO https://pjreddie.com/darknet/yolo/.

## Requirements

Python 3.8 or later with all [requirements.txt](https://github.com/ultralytics/yolov3/blob/master/requirements.txt) dependencies installed, including `torch>=1.6`. To install run:

```bash
$ pip install -r requirements.txt
```

## Tutorials

- [Notebook](https://github.com/ultralytics/yolov3/blob/master/tutorial.ipynb) <a href="https://colab.research.google.com/github/ultralytics/yolov3/blob/master/tutorial.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
- [Train Custom Data](https://github.com/ultralytics/yolov3/wiki/Train-Custom-Data) << highly recommended
- [GCP Quickstart](https://github.com/ultralytics/yolov3/wiki/GCP-Quickstart)
- [Docker Quickstart Guide](https://github.com/ultralytics/yolov3/wiki/Docker-Quickstart) ![Docker Pulls](https://img.shields.io/docker/pulls/ultralytics/yolov3?logo=docker)
- [A TensorRT Implementation of YOLOv3 and YOLOv4](https://github.com/wang-xinyu/tensorrtx/tree/master/yolov3-spp)

## Training

**Start Training:** `python3 train.py` to begin training after downloading COCO data with `data/get_coco2017.sh`. Each epoch trains on 117,263 images from the train and validate COCO sets, and tests on 5000 images from the COCO validate set.

**Resume Training:** `python3 train.py --resume` to resume training from `weights/last.pt`.

**Plot Training:** `from utils import utils; utils.plot_results()`

<img src="https://user-images.githubusercontent.com/26833433/78175826-599d4800-7410-11ea-87d4-f629071838f6.png" width="900">

### Image Augmentation

`datasets.py` applies OpenCV-powered (https://opencv.org/) augmentation to the input image. We use a **mosaic dataloader** to increase image variability during training.

<img src="https://user-images.githubusercontent.com/26833433/80769557-6e015d00-8b02-11ea-9c4b-69310eb2b962.jpg" width="900">

### Speed

https://cloud.google.com/deep-learning-vm/  
**Machine type:** preemptible [n1-standard-8](https://cloud.google.com/compute/docs/machine-types) (8 vCPUs, 30 GB memory)  
**CPU platform:** Intel Skylake  
**GPUs:** K80 ($0.14/hr), T4 ($0.11/hr), V100 ($0.74/hr) CUDA with [Nvidia Apex](https://github.com/NVIDIA/apex) FP16/32  
**HDD:** 300 GB SSD  
**Dataset:** COCO train 2014 (117,263 images)  
**Model:** `yolov3-spp.cfg`  
**Command:** `python3 train.py --data coco2017.data --img 416 --batch 32`

| GPU    | n      | `--batch-size`   | img/s          | epoch<br>time        | epoch<br>cost      |
| ------ | ------ | ---------------- | -------------- | -------------------- | ------------------ |
| K80    | 1      | 32 x 2           | 11             | 175 min              | $0.41              |
| T4     | 1<br>2 | 32 x 2<br>64 x 1 | 41<br>61       | 48 min<br>32 min     | $0.09<br>$0.11     |
| V100   | 1<br>2 | 32 x 2<br>64 x 1 | 122<br>**178** | 16 min<br>**11 min** | **$0.21**<br>$0.28 |
| 2080Ti | 1<br>2 | 32 x 2<br>64 x 1 | 81<br>140      | 24 min<br>14 min     | -<br>-             |

## Inference

```bash
python3 detect.py --source ...
```

- Image: `--source file.jpg`
- Video: `--source file.mp4`
- Directory: `--source dir/`
- Webcam: `--source 0`
- RTSP stream: `--source rtsp://170.93.143.139/rtplive/470011e600ef003a004ee33696235daa`
- HTTP stream: `--source http://112.50.243.8/PLTV/88888888/224/3221225900/1.m3u8`

**YOLOv3:** `python3 detect.py --cfg cfg/yolov3.cfg --weights yolov3.pt`  
<img src="https://user-images.githubusercontent.com/26833433/64067835-51d5b500-cc2f-11e9-982e-843f7f9a6ea2.jpg" width="500">

**YOLOv3-tiny:** `python3 detect.py --cfg cfg/yolov3-tiny.cfg --weights yolov3-tiny.pt`  
<img src="https://user-images.githubusercontent.com/26833433/64067834-51d5b500-cc2f-11e9-9357-c485b159a20b.jpg" width="500">

**YOLOv3-SPP:** `python3 detect.py --cfg cfg/yolov3-spp.cfg --weights yolov3-spp.pt`  
<img src="https://user-images.githubusercontent.com/26833433/64067833-51d5b500-cc2f-11e9-8208-6fe197809131.jpg" width="500">

## Pretrained Checkpoints

Download from: [https://drive.google.com/open?id=1LezFG5g3BCW6iYaV89B2i64cqEUZD7e0](https://drive.google.com/open?id=1LezFG5g3BCW6iYaV89B2i64cqEUZD7e0)

## Darknet Conversion

```bash
$ git clone https://github.com/ultralytics/yolov3 && cd yolov3

# convert darknet cfg/weights to pytorch model
$ python3  -c "from models import *; convert('cfg/yolov3-spp.cfg', 'weights/yolov3-spp.weights')"
Success: converted 'weights/yolov3-spp.weights' to 'weights/yolov3-spp.pt'

# convert cfg/pytorch model to darknet weights
$ python3  -c "from models import *; convert('cfg/yolov3-spp.cfg', 'weights/yolov3-spp.pt')"
Success: converted 'weights/yolov3-spp.pt' to 'weights/yolov3-spp.weights'
```

## mAP

| <i></i>                                                                                                                                 | Size | COCO mAP<br>@0.5...0.95          | COCO mAP<br>@0.5                 |
| --------------------------------------------------------------------------------------------------------------------------------------- | ---- | -------------------------------- | -------------------------------- |
| YOLOv3-tiny<br>YOLOv3<br>YOLOv3-SPP<br>**[YOLOv3-SPP-ultralytics](https://drive.google.com/open?id=1UcR-zVoMs7DH5dj3N1bswkiQTA4dmKF4)** | 320  | 14.0<br>28.7<br>30.5<br>**37.7** | 29.1<br>51.8<br>52.3<br>**56.8** |
| YOLOv3-tiny<br>YOLOv3<br>YOLOv3-SPP<br>**[YOLOv3-SPP-ultralytics](https://drive.google.com/open?id=1UcR-zVoMs7DH5dj3N1bswkiQTA4dmKF4)** | 416  | 16.0<br>31.2<br>33.9<br>**41.2** | 33.0<br>55.4<br>56.9<br>**60.6** |
| YOLOv3-tiny<br>YOLOv3<br>YOLOv3-SPP<br>**[YOLOv3-SPP-ultralytics](https://drive.google.com/open?id=1UcR-zVoMs7DH5dj3N1bswkiQTA4dmKF4)** | 512  | 16.6<br>32.7<br>35.6<br>**42.6** | 34.9<br>57.7<br>59.5<br>**62.4** |
| YOLOv3-tiny<br>YOLOv3<br>YOLOv3-SPP<br>**[YOLOv3-SPP-ultralytics](https://drive.google.com/open?id=1UcR-zVoMs7DH5dj3N1bswkiQTA4dmKF4)** | 608  | 16.6<br>33.1<br>37.0<br>**43.1** | 35.4<br>58.2<br>60.7<br>**62.8** |

- mAP@0.5 run at `--iou-thr 0.5`, mAP@0.5...0.95 run at `--iou-thr 0.7`
- Darknet results: https://arxiv.org/abs/1804.02767

```bash
$ python3 test.py --cfg yolov3-spp.cfg --weights yolov3-spp-ultralytics.pt --img 640 --augment

Namespace(augment=True, batch_size=16, cfg='cfg/yolov3-spp.cfg', conf_thres=0.001, data='coco2014.data', device='', img_size=640, iou_thres=0.6, save_json=True, single_cls=False, task='test', weights='weight
Using CUDA device0 _CudaDeviceProperties(name='Tesla V100-SXM2-16GB', total_memory=16130MB)

               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|█████████| 313/313 [03:00<00:00,  1.74it/s]
                 all     5e+03  3.51e+04     0.375     0.743      0.64     0.492

 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.456
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.647
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.496
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.263
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.501
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.596
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.361
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.597
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.666
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.492
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.719
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.810

Speed: 17.5/2.3/19.9 ms inference/NMS/total per 640x640 image at batch-size 16
```

<!-- Speed: 11.4/2.2/13.6 ms inference/NMS/total per 608x608 image at batch-size 1 -->

## Reproduce Our Results

Run commands below. Training takes about one week on a 2080Ti per model.

```bash
$ python train.py --data coco2014.data --weights '' --batch-size 16 --cfg yolov3-spp.cfg
$ python train.py --data coco2014.data --weights '' --batch-size 32 --cfg yolov3-tiny.cfg
```

<img src="https://user-images.githubusercontent.com/26833433/80831822-57a9de80-8ba0-11ea-9684-c47afb0432dc.png" width="900">

## Reproduce Our Environment

To access an up-to-date working environment (with all dependencies including CUDA/CUDNN, Python and PyTorch preinstalled), consider a:

- **GCP** Deep Learning VM with $300 free credit offer: See our [GCP Quickstart Guide](https://github.com/ultralytics/yolov3/wiki/GCP-Quickstart)
- **Google Colab Notebook** with 12 hours of free GPU time. <a href="https://colab.research.google.com/github/ultralytics/yolov3/blob/master/tutorial.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
- **Docker Image** https://hub.docker.com/r/ultralytics/yolov3. See [Docker Quickstart Guide](https://github.com/ultralytics/yolov3/wiki/Docker-Quickstart) ![Docker Pulls](https://img.shields.io/docker/pulls/ultralytics/yolov3?logo=docker)

## Citation

[![DOI](https://zenodo.org/badge/146165888.svg)](https://zenodo.org/badge/latestdoi/146165888)

## About Us

Ultralytics is a U.S.-based particle physics and AI startup with over 6 years of expertise supporting government, academic and business clients. We offer a wide range of vision AI services, spanning from simple expert advice up to delivery of fully customized, end-to-end production solutions, including:

- **Cloud-based AI** systems operating on **hundreds of HD video streams in realtime.**
- **Edge AI** integrated into custom iOS and Android apps for realtime **30 FPS video inference.**
- **Custom data training**, hyperparameter evolution, and model exportation to any destination.

For business inquiries and professional support requests please visit us at https://www.ultralytics.com.

## Contact

**Issues should be raised directly in the repository.** For business inquiries or professional support requests please visit https://www.ultralytics.com or email Glenn Jocher at glenn.jocher@ultralytics.com.

################################# 比特大陆：模型量化 ###########################################

## attention Bug

1. 在比特大陆转换 u/bmodel 模型时，配置文件.cfg 中的 batch = 1
2. 在比特大陆平台图片.JPG 不能读出，必须转化成.jpg 读出
3. 在平台上运行算法，模型必须先加密
4. 在比特大陆平台，多个 model 之间的名字不能一样 INT8.model/INT8A.model
5. cv::bmcv::uploadMat(cls_mat_roi); 保存图片才不会出错，针对 mat 对象，需要刷新
6. 在 git 上拖到 Windows 上.so 库会损坏，Windows 上无权限管理。分 2 次拖，Windows 上上传，linux 上修改测试。

公司 git 推送代码：
Hint: To automatically insert Change-Id, install the hook:
remote: gitdir=$(git rev-parse --git-dir); scp -p -P 29418 ganhaiyang@10.100.13.21:hooks/commit-msg ${gitdir}/hooks/
remote: And then amend the commit:
remote: git commit --amend
(1)走读-》Pubulish Edit
(2)添加修改点，测试点-》Commit Message

## 制作 mdb 数据集

convert*imageset --shuffle --resize_height=416 --resize_width=416 / 20210205jingdianmao_val.txt img_lmdb
file:///Z:/common2/%E6%A8%A1%E5%9E%8B%E9%87%8F%E5%8C%96/%E6%A8%A1%E5%9E%8B%E9%87%8F%E5%8C%96*%E6%89%93%E7%94%B5%E8%AF%9D%E6%A8%A1%E5%9E%8B(YOLOV3).html

################################# map 平台测试工具 ###########################################
Z:\common2\AI_Server\test_tool

## 模型发布流程：

1. 模型量化-in8 ——> 模型加密
   模型量化保存（-dump_dist）中间文件：
   calibration_use_pb release --model darknet_model_bmnetd_test_fp32.prototxt --weights darknet_model_bmnetd.fp32umodel --iterations 50 -dump_dist=last.th --bitwidth TO_INT8

模型量化加载（-load_dist)中间文件：
calibration_use_pb release --model darknet_model_bmnetd_test_fp32.prototxt --weights darknet_model_bmnetd.fp32umodel --iterations 1000 -load_dist=last.th --bitwidth TO_INT8

2.  模型路径：\\10.100.11.208\iot\AI 版本发布\算法集成\bit\vehicle\vehicle_V1.2_20210225_encrypt
    发布平台：Bitman
    应用场景：车辆检测相关

修改点：
存在问题：
本地服务器无法正常测试，出现很多 多余的 目标框，设备端验证不存在该问题。
配置文件：见 model_cfg.json
精确度指标：

3.平台端实例
Z:\common2\技术文档\场景文档\确认 OK 4.检测流程
Z:\likeliang\bitmain\bitmain_1209\bmnnsdk2-bm1684_v2.2.0\QK_AI_Box\external\aicore\core\object_detector

5.阈值

################################# 比特大陆测试数据 ###########################################
Z:\ganhaiyang\Alg_Proj\2.2.0_20201117_042200\bmnnsdk2\bmnnsdk2-bm1684_v2.2.0\QK_AI_Box\external\aicore\sample_test\build\sample_object_detector

模型发布：
\\10.100.11.208\iot\AI 版本发布\算法集成\bit\ele_cap_protection_shoes

模型备份：
Z:\common2\模型发布\模型发布-bit\ele_clothes

## ############# hand_smoke#####################################

yolo v4 是 5 层输出的，yolo v3 用三层输出。
Y:\jiadongfeng\AI\darknet_v4
./darknet detector test cfg/smoke.data cfg/smoke_5l_test.cfg backup/smoke/smoke_5l_10500.weights /home/os/window_share/common2/dataset/hand_smoke/test/images -thresh 0.3 -out /home/os/window_share/common2/dataset/hand_smoke/test/images_out/ | tee /home/os/window_share/common2/dataset/hand_smoke/test/smoke.txt

大版本：运行 frameworks：
（1）在 QK_AI_Box 上设置编译器，运行 source build/envsetup.sh
(2) 编译及安装库 make --》 make install 注意：编译大版本不能用 make -j16 编译，报错！
（3）盒子配置：http://192.168.3.13/#/login 用户名/密码：admin 123456

# ################ bitman

1. 测试例子
   Z:\ganhaiyang\Alg_Proj\2.2.0_20201117_042200\bmnnsdk2\bmnnsdk2-bm1684_v2.2.0\examples\YOLOv3_object\cpp_cv_bmcv_bmrt_postprocess\yolov3.cpp
   make -f Makefile.arm
2. AIScene 架构
   AIScene 是接口类。对外提供 AiScene。ComScene 是场景通用类，封装通用操作。
   集成关系：AiScene--->ComScene---->SmokeScene
3. 查看日志（gdb 调试)
   build/envsetup.sh --》export BUILD_DEBUG=no 改为 yes 可定位到具体行号
   （1）关掉服务： sudo systemctl stop AICoreDaemon.service(不用重启)
   （2）打开 root：sudo -i
   （3）设置路径：export LD_LIBRARY_PATH=/system/ai_monitor/lib:$LD_LIBRARY_PATH;export PATH=/system/ai_monitor/bin:$PATH
   （4）gdb ai_system -》r 按 Enter -》bt 按 Enter 到具体的行号

   sample_test 调试：
   （1）gdb --args ./slowfast_test image imgs.txt sf18_pytorch_cpu/compilation.bmodel （--args 运行秩序文件带参数）
   （2）-》r 运行； -》bt 按 Enter 到具体的行号； f 0 查看函数；p i； p inputs[0]

4. log 文件中建立.debug 文件，不会下载模型，要删除
   /data/ai_monitor/log
   /data/ai_monitor/simulate/video/channels 删除视频

5. 问题反馈
   \\10.100.11.208\iot\比特大陆\问题汇总

6. 刷新的 SDK 版本
   （1）参考文件 1：\\10.100.11.208\PUB_Tool\360OS 系统框架组文档目录\XT_IOTOS\BitMain\360OS\制作全量 SD 卡升级软件与 OTA 升级包
   制作全量 SD 卡升级软件与 OTA 升级包流程.html

（2）参考文件 2：\\10.100.11.208\PUB_Tool\360OS 系统框架组文档目录\XT_IOTOS\BitMain\360OS
BitMain 环境搭建.html
