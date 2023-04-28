# Python_ConvolutionalNeuralNetwork_FaceRecognition_fatiguedriving
基于Python卷积神经网络人脸识别驾驶员疲劳检测与预警系统设计

开发技术环境： Pycharm + Python3.6 + PyQt5 + OpenCV + 卷积神经网络模型

  本文采用卷积神经算法对驾驶室内的驾驶员进行实时的面部图像抓拍，通过图像处理的技术分析人眼的闭合程度，从而判断驾驶员的疲劳程度。本文介绍了对目标图像进行人脸检测，然后在分割出的人脸图像中，对人脸图像进行水平投影，并根据水平投影得到的人眼上下眼睑，定位出人眼的位置，而且根据人眼的上下眼睑可以通过事先给出的一定判别标准，判断眼部是否处于疲劳状态，从而达到疲劳检测的目的。当检测出驾驶员处于疲劳时，系统会自动报警，使驾驶员恢复到正常状态，从而尽量规避了行车的安全隐患，并且系统做出预留功能，可以将驾驶员的疲劳状态图片发送给指定的服务器以备查询。因此组成本系统中系统模块如下：
（1）视频采集模块
（2）图像预处理模块
（3）人脸定位模块
（4）人眼定位模块
（5）疲劳程度判别模块
（6）报警模块
