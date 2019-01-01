# RCNN论文复现
R-CNN: Regions with Convolutional Neural Network Features

## 基本知识介绍
### 1,计算机视觉中的不同任务
![](https://github.com/bigbrother33/Deep-Learning/blob/master/photo/20190101200347.png)<br><br>
### 2,IOU的定义
简单介绍一下IOU。物体检测需要定位出物体的boundingbox，就像下面的图片一样，我们不仅要定位出车辆的bounding box 我们还要识别出bounding box 里面的物体就是车辆。对于boundingbox的定位精度，有一个很重要的概念，因为我们算法不可能百分百跟人工标注的数据完全匹配，因此就存在一个定位精度评价公式：`IOU`  <br>
![](https://github.com/bigbrother33/Deep-Learning/blob/master/photo/20160902124803660.png)   
IOU定义了两个bounding box的重叠度，如下图所示： 

![](https://github.com/bigbrother33/Deep-Learning/blob/master/photo/20160902124815518.png)  
矩形框A、B的一个重合度IOU计算公式为：<br>
<a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{100}&space;\LARGE&space;IOU=(A\cap&space;B)/(A\cup&space;B)" target="_blank"><img src="https://latex.codecogs.com/png.latex?\dpi{100}&space;\LARGE&space;IOU=(A\cap&space;B)/(A\cup&space;B)" title="\LARGE IOU=(A\cap B)/(A\cup B)" /></a><br>
