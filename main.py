import datetime
import os
import sys
import threading
import math
import time

from PySide6.QtGui import QImage

import main_ui

from PySide6 import QtCore, QtGui, QtWidgets
import dlib
import numpy as np
import cv2
from scipy.spatial import distance as dist

from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils

import pygame

# pyside6-uic -o main_ui.py main.ui

class MainUI(QtWidgets.QWidget, main_ui.Ui_Form):
    # 信号，在UI线程中，不能在其他线程直接操作UI
    thread_signal = QtCore.Signal(dict)

    def __init__(self):
        super().__init__()
        self.setupUi(self)

        # 连接信号
        self.pushButton.clicked.connect(self.button_clicked)
        # self.thread_signal.connect(self.thread_singnal_slot)

        # 六个功能是否要用
        self.fun = [True for i in range(6)]
        self.checkBox_11.setChecked(self.fun[0])
        self.checkBox_12.setChecked(self.fun[1])
        self.checkBox_21.setChecked(self.fun[2])
        self.checkBox_22.setChecked(self.fun[3])
        self.checkBox_31.setChecked(self.fun[4])
        self.checkBox_32.setChecked(self.fun[5])

        self.checkBox_11.stateChanged.connect(self.select_changed)
        self.checkBox_12.stateChanged.connect(self.select_changed)
        self.checkBox_21.stateChanged.connect(self.select_changed)
        self.checkBox_22.stateChanged.connect(self.select_changed)
        self.checkBox_31.stateChanged.connect(self.select_changed)
        self.checkBox_32.stateChanged.connect(self.select_changed)

        # 阈值
        self.values = [3,2,3,5,2]
        self.spinBox_1.setValue(self.values[0])
        self.spinBox_2.setValue(self.values[1])
        self.spinBox_3.setValue(self.values[2])
        self.spinBox_4.setValue(self.values[3])
        self.spinBox_5.setValue(self.values[4])
        self.spinBox_1.valueChanged.connect(self.value_changed)
        self.spinBox_2.valueChanged.connect(self.value_changed)
        self.spinBox_3.valueChanged.connect(self.value_changed)
        self.spinBox_4.valueChanged.connect(self.value_changed)
        self.spinBox_5.valueChanged.connect(self.value_changed)
        #self.spinBox_6.valueChanged.connect(self.value_changed)

        self.thread_signal.connect(self.thread_singnal_slot)

        self.label_img.setScaledContents(True)

        self.plainTextEdit_tip.appendPlainText('等待开始\n')


        """参数"""
        # 默认为摄像头0
        self.VIDEO_STREAM = 0
        self.CAMERA_STYLE = False  # False未打开摄像头，True摄像头已打开
        # 闪烁阈值（秒）
        self.AR_CONSEC_FRAMES_check = 3
        self.OUT_AR_CONSEC_FRAMES_check = 5
        # 眼睛长宽比
        self.EYE_AR_THRESH = 0.2
        self.EYE_AR_CONSEC_FRAMES = self.AR_CONSEC_FRAMES_check
        # 打哈欠长宽比
        self.MAR_THRESH = 0.5
        self.MOUTH_AR_CONSEC_FRAMES = self.AR_CONSEC_FRAMES_check
        # 瞌睡点头
        self.HAR_THRESH = 0.3
        self.NOD_AR_CONSEC_FRAMES = self.AR_CONSEC_FRAMES_check

        """计数"""
        # 初始化帧计数器和眨眼总数
        self.COUNTER = 0
        self.TOTAL = 0
        # 初始化帧计数器和打哈欠总数
        self.mCOUNTER = 0
        self.mTOTAL = 0
        # 初始化帧计数器和点头总数
        self.hCOUNTER = 0
        self.hTOTAL = 0
        # 离职时间长度
        self.oCOUNTER = 0

        """姿态"""
        # 世界坐标系(UVW)：填写3D参考点，该模型参考http://aifi.isr.uc.pt/Downloads/OpenGL/glAnthropometric3DModel.cpp
        self.object_pts = np.float32([[6.825897, 6.760612, 4.402142],  #33左眉左上角
                                 [1.330353, 7.122144, 6.903745],  #29左眉右角
                                 [-1.330353, 7.122144, 6.903745], #34右眉左角
                                 [-6.825897, 6.760612, 4.402142], #38右眉右上角
                                 [5.311432, 5.485328, 3.987654],  #13左眼左上角
                                 [1.789930, 5.393625, 4.413414],  #17左眼右上角
                                 [-1.789930, 5.393625, 4.413414], #25右眼左上角
                                 [-5.311432, 5.485328, 3.987654], #21右眼右上角
                                 [2.005628, 1.409845, 6.165652],  #55鼻子左上角
                                 [-2.005628, 1.409845, 6.165652], #49鼻子右上角
                                 [2.774015, -2.080775, 5.048531], #43嘴左上角
                                 [-2.774015, -2.080775, 5.048531],#39嘴右上角
                                 [0.000000, -3.116408, 6.097667], #45嘴中央下角
                                 [0.000000, -7.415691, 4.070434]])#6下巴角

        # 相机坐标系(XYZ)：添加相机内参
        self.K = [6.5308391993466671e+002, 0.0, 3.1950000000000000e+002,
                 0.0, 6.5308391993466671e+002, 2.3950000000000000e+002,
                 0.0, 0.0, 1.0]# 等价于矩阵[fx, 0, cx; 0, fy, cy; 0, 0, 1]
        # 图像中心坐标系(uv)：相机畸变参数[k1, k2, p1, p2, k3]
        self.D = [7.0834633684407095e-002, 6.9140193737175351e-002, 0.0, 0.0, -1.3073460323689292e+000]

        # 像素坐标系(xy)：填写凸轮的本征和畸变系数
        self.cam_matrix = np.array(self.K).reshape(3, 3).astype(np.float32)
        self.dist_coeffs = np.array(self.D).reshape(5, 1).astype(np.float32)

        # 重新投影3D点的世界坐标轴以验证结果姿势
        self.reprojectsrc = np.float32([[10.0, 10.0, 10.0],
                                       [10.0, 10.0, -10.0],
                                       [10.0, -10.0, -10.0],
                                       [10.0, -10.0, 10.0],
                                       [-10.0, 10.0, 10.0],
                                       [-10.0, 10.0, -10.0],
                                       [-10.0, -10.0, -10.0],
                                       [-10.0, -10.0, 10.0]])
        # 绘制正方体12轴
        self.line_pairs = [[0, 1], [1, 2], [2, 3], [3, 0],
                          [4, 5], [5, 6], [6, 7], [7, 4],
                          [0, 4], [1, 5], [2, 6], [3, 7]]


        # 线程
        self.thread = None
        self.sound_thread = None


    def get_head_pose(self,shape):# 头部姿态估计
        # （像素坐标集合）填写2D参考点，注释遵循https://ibug.doc.ic.ac.uk/resources/300-W/
        # 17左眉左上角/21左眉右角/22右眉左上角/26右眉右上角/36左眼左上角/39左眼右上角/42右眼左上角/
        # 45右眼右上角/31鼻子左上角/35鼻子右上角/48左上角/54嘴右上角/57嘴中央下角/8下巴角
        image_pts = np.float32([shape[17], shape[21], shape[22], shape[26], shape[36],
                                shape[39], shape[42], shape[45], shape[31], shape[35],
                                shape[48], shape[54], shape[57], shape[8]])
        # solvePnP计算姿势——求解旋转和平移矩阵：
        # rotation_vec表示旋转矩阵，translation_vec表示平移矩阵，cam_matrix与K矩阵对应，dist_coeffs与D矩阵对应。
        _, rotation_vec, translation_vec = cv2.solvePnP(self.object_pts, image_pts, self.cam_matrix, self.dist_coeffs)
        # projectPoints重新投影误差：原2d点和重投影2d点的距离（输入3d点、相机内参、相机畸变、r、t，输出重投影2d点）
        reprojectdst, _ = cv2.projectPoints(self.reprojectsrc, rotation_vec, translation_vec, self.cam_matrix,self.dist_coeffs)
        reprojectdst = tuple(map(tuple, reprojectdst.reshape(8, 2)))# 以8行2列显示

        # 计算欧拉角calc euler angle
        # 参考https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html#decomposeprojectionmatrix
        rotation_mat, _ = cv2.Rodrigues(rotation_vec)#罗德里格斯公式（将旋转矩阵转换为旋转向量）
        pose_mat = cv2.hconcat((rotation_mat, translation_vec))# 水平拼接，vconcat垂直拼接
        # decomposeProjectionMatrix将投影矩阵分解为旋转矩阵和相机矩阵
        _, _, _, _, _, _, euler_angle = cv2.decomposeProjectionMatrix(pose_mat)

        pitch, yaw, roll = [math.radians(_) for _ in euler_angle]

        pitch = math.degrees(math.asin(math.sin(pitch)))
        roll = -math.degrees(math.asin(math.sin(roll)))
        yaw = math.degrees(math.asin(math.sin(yaw)))
        #print('pitch:{}, yaw:{}, roll:{}'.format(pitch, yaw, roll))

        return reprojectdst, euler_angle# 投影误差，欧拉角
    def eye_aspect_ratio(self,eye):
        # 垂直眼标志（X，Y）坐标
        A = dist.euclidean(eye[1], eye[5])# 计算两个集合之间的欧式距离
        B = dist.euclidean(eye[2], eye[4])
        # 计算水平之间的欧几里得距离
        # 水平眼标志（X，Y）坐标
        C = dist.euclidean(eye[0], eye[3])
        # 眼睛长宽比的计算
        ear = (A + B) / (2.0 * C)
        # 返回眼睛的长宽比
        return ear

    def mouth_aspect_ratio(self,mouth):# 嘴部
        A = np.linalg.norm(mouth[2] - mouth[9])  # 51, 59
        B = np.linalg.norm(mouth[4] - mouth[7])  # 53, 57
        C = np.linalg.norm(mouth[0] - mouth[6])  # 49, 55
        mar = (A + B) / (2.0 * C)
        return mar

    def _learning_face(self):
        """dlib的初始化调用"""
        # 使用人脸检测器get_frontal_face_detector
        self.detector = dlib.get_frontal_face_detector()
        # dlib的68点模型，使用作者训练好的特征预测器
        self.predictor = dlib.shape_predictor("./model/shape_predictor_68_face_landmarks.dat")
        data = {
            'type':'msg',
            'value':u"加载模型成功!!!\n"
        }
        self.thread_signal.emit(data)

        # 分别获取左右眼面部标志的索引
        (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        (mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

        # 建cv2摄像头对象，这里使用电脑自带摄像头，如果接了外部摄像头，则自动切换到外部摄像头
        self.cap = cv2.VideoCapture(self.VIDEO_STREAM)

        data = {
            'type': 'msg',
        }
        if self.cap.isOpened() == True:  # 返回true/false 检查初始化是否成功
            self.CAMERA_STYLE = True
            data['value'] = u"打开摄像头成功!!!"
        else:
            data['value'] = u"摄像头打开失败!!!"
        self.thread_signal.emit(data)
        # 所有结果
        res = []
        t_time = datetime.datetime.now()
        e_time = datetime.datetime.now()
        h_time = datetime.datetime.now()
        # 成功打开视频，循环读取视频流
        while (self.cap.isOpened()):
            start_time = datetime.datetime.now()
            res = ['-' for i in range(9)]
            # cap.read()
            # 返回两个值：
            #    一个布尔值true/false，用来判断读取视频是否成功/是否到视频末尾
            #    图像对象，图像的三维矩阵
            flag, im_rd = self.cap.read()
            # 取灰度
            img_gray = cv2.cvtColor(im_rd, cv2.COLOR_RGB2GRAY)

            # 使用人脸检测器检测每一帧图像中的人脸。并返回人脸数faces
            faces = self.detector(img_gray, 0)
            # 如果检测到人脸
            if (len(faces) != 0):
                res[0] = '识别到人脸'
                # enumerate方法同时返回数据对象的索引和数据，k为索引，d为faces中的对象
                for k, d in enumerate(faces):
                    # 用红色矩形框出人脸
                    cv2.rectangle(im_rd, (d.left(), d.top()), (d.right(), d.bottom()), (0, 0, 255), 1)
                    # 使用预测器得到68点数据的坐标
                    shape = self.predictor(im_rd, d)
                    # 圆圈显示每个特征点
                    for i in range(68):
                        cv2.circle(im_rd, (shape.part(i).x, shape.part(i).y), 2, (0, 255, 0), -1, 8)
                    # 将脸部特征信息转换为数组array的格式
                    shape = face_utils.shape_to_np(shape)

                    """
                    打哈欠
                    """
                    if self.fun[1]:
                        # 嘴巴坐标
                        mouth = shape[mStart:mEnd]
                        # 打哈欠
                        mar = self.mouth_aspect_ratio(mouth)
                        # 使用cv2.convexHull获得凸包位置，使用drawContours画出轮廓位置进行画图操作
                        mouthHull = cv2.convexHull(mouth)
                        cv2.drawContours(im_rd, [mouthHull], -1, (0, 255, 0), 1)
                        # 同理，判断是否打哈欠
                        if mar > self.MAR_THRESH:  # 张嘴阈值0.5
                            self.mCOUNTER += 1
                            res[4] = '张嘴'
                        else:
                            # 如果连续3次都小于阈值，则表示打了一次哈欠
                            if self.mCOUNTER >= self.MOUTH_AR_CONSEC_FRAMES:  # 阈值：3
                                self.mTOTAL += 1
                                # 显示
                                # cv2.putText(im_rd, "Yawning!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                                self.thread_signal.emit({'type':'msg','value':time.strftime('%Y-%m-%d %H:%M ', time.localtime()) + u"打哈欠"})
                            # 重置嘴帧计数器
                            self.mCOUNTER = 0
                            res[4] = '闭嘴'
                        # cv2.putText(im_rd, "COUNTER: {}".format(self.mCOUNTER), (150, 60), cv2.FONT_HERSHEY_SIMPLEX,
                        #             0.7, (0, 0, 255), 2)
                        # cv2.putText(im_rd, "MAR: {:.2f}".format(mar), (300, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        #             (0, 0, 255), 2)
                        # cv2.putText(im_rd, "Yawning: {}".format(self.mTOTAL), (450, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        #             (255, 255, 0), 2)
                    else:
                        pass
                    """
                    眨眼
                    """
                    if self.fun[0]:
                        # 提取左眼和右眼坐标
                        leftEye = shape[lStart:lEnd]
                        rightEye = shape[rStart:rEnd]
                        # 构造函数计算左右眼的EAR值，使用平均值作为最终的EAR
                        leftEAR = self.eye_aspect_ratio(leftEye)
                        rightEAR = self.eye_aspect_ratio(rightEye)
                        ear = (leftEAR + rightEAR) / 2.0
                        leftEyeHull = cv2.convexHull(leftEye)
                        rightEyeHull = cv2.convexHull(rightEye)
                        # 使用cv2.convexHull获得凸包位置，使用drawContours画出轮廓位置进行画图操作
                        cv2.drawContours(im_rd, [leftEyeHull], -1, (0, 255, 0), 1)
                        cv2.drawContours(im_rd, [rightEyeHull], -1, (0, 255, 0), 1)
                        # 循环，满足条件的，眨眼次数+1
                        if ear < self.EYE_AR_THRESH:  # 眼睛长宽比：0.2
                            self.COUNTER += 1
                            res[5] = '闭眼'
                        else:
                            # 如果连续3次都小于阈值，则表示进行了一次眨眼活动
                            if self.COUNTER >= self.EYE_AR_CONSEC_FRAMES:  # 阈值：3
                                self.TOTAL += 1
                                self.thread_signal.emit({'type':'msg','value':time.strftime('%Y-%m-%d %H:%M ', time.localtime()) + u"眨眼"})
                            # 重置眼帧计数器
                            self.COUNTER = 0
                            res[5] = '睁眼'
                        # 第十四步：进行画图操作，同时使用cv2.putText将眨眼次数进行显示
                        # cv2.putText(im_rd, "Faces: {}".format(len(faces)), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        #             (0, 0, 255), 2)
                        # cv2.putText(im_rd, "COUNTER: {}".format(self.COUNTER), (150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        #             (0, 0, 255), 2)
                        # cv2.putText(im_rd, "EAR: {:.2f}".format(ear), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        #             (0, 0, 255), 2)
                        # cv2.putText(im_rd, "Blinks: {}".format(self.TOTAL), (450, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        #             (255, 255, 0), 2)
                    else:
                        pass
                    """
                    瞌睡点头
                    """
                    if self.fun[2]:
                        # 获取头部姿态
                        reprojectdst, euler_angle = self.get_head_pose(shape)
                        har = euler_angle[0, 0]  # 取pitch旋转角度
                        if har > self.HAR_THRESH:  # 点头阈值0.3
                            self.hCOUNTER += 1
                            res[3] = '斜'
                        else:
                            # 如果连续3次都小于阈值，则表示瞌睡点头一次
                            if self.hCOUNTER >= self.NOD_AR_CONSEC_FRAMES:  # 阈值：3
                                self.hTOTAL += 1
                                self.thread_signal.emit({'type': 'msg', 'value': time.strftime('%Y-%m-%d %H:%M ',
                                                                                                   time.localtime()) + u"瞌睡点头"})
                            # 重置点头帧计数器
                            self.hCOUNTER = 0
                            res[3] = '正'
                        # 绘制正方体12轴(视频流尺寸过大时，reprojectdst会超出int范围，建议压缩检测视频尺寸)
                        # for start, end in self.line_pairs:
                        #     x1, y1 = reprojectdst[start]
                        #     x2, y2 = reprojectdst[end]
                            #cv2.line(im_rd, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255))
                        # 显示角度结果
                        # cv2.putText(im_rd, "X: " + "{:7.2f}".format(euler_angle[0, 0]), (10, 90),
                        #             cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), thickness=2)  # GREEN
                        # cv2.putText(im_rd, "Y: " + "{:7.2f}".format(euler_angle[1, 0]), (150, 90),
                        #             cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), thickness=2)  # BLUE
                        # cv2.putText(im_rd, "Z: " + "{:7.2f}".format(euler_angle[2, 0]), (300, 90),
                        #             cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), thickness=2)  # RED
                        # cv2.putText(im_rd, "Nod: {}".format(self.hTOTAL), (450, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        #             (255, 255, 0), 2)
                    else:
                        pass

                # print('嘴巴实时长宽比:{:.2f} '.format(mar)+"\t是否张嘴："+str([False,True][mar > self.MAR_THRESH]))
                # print('眼睛实时长宽比:{:.2f} '.format(ear)+"\t是否眨眼："+str([False,True][self.COUNTER>=1]))
                # 是否瞌睡
                # 点头次数
                res[6] = str(self.TOTAL)
                # 哈欠次数
                res[7] = str(self.mTOTAL)
                # 点头次数
                res[8] = str(self.hTOTAL)
                res[1] = '正常'
                if self.TOTAL >= self.values[0]:
                    res[1] = '轻微疲劳'
                    pass
                if self.hTOTAL >= self.values[2]:
                    if self.TOTAL >= self.values[0]:
                        res[1] = '瞌睡'
                    else:
                        res[1] = '轻微疲劳'
                    pass
                if self.mTOTAL >= self.values[4]:
                    if self.TOTAL >= self.values[0]:
                        res[1] = '瞌睡'
                    else:
                        res[1] = '轻微疲劳'
                    pass
                if (datetime.datetime.now() - t_time).seconds >= 10:
                    self.TOTAL = 0
                    self.mTOTAL = 0
                    self.hTOTAL = 0
                    t_time = datetime.datetime.now()
                if res[3] == '斜' and res[5]=='闭眼':
                    if (datetime.datetime.now() - h_time).seconds>=self.values[3]:
                        res[1] = '瞌睡'
                else:
                    h_time = datetime.datetime.now()
                if res[5] == '闭眼':
                    if (datetime.datetime.now() - e_time).seconds>=self.values[1]:
                        res[1] = '瞌睡'
                else:
                    e_time = datetime.datetime.now()


            else:
                res[0] = '未识别到'
                # 没有检测到人脸
                self.oCOUNTER += 1
                cv2.putText(im_rd, "No Face", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA)
                if self.oCOUNTER >= self.OUT_AR_CONSEC_FRAMES_check:
                    self.thread_signal.emit({'type': 'msg', 'value': time.strftime('%Y-%m-%d %H:%M ',
                                                                                       time.localtime()) + u"没有识别到人脸"})
                    self.oCOUNTER = 0

                self.TOTAL = 0
                self.mTOTAL = 0
                self.hTOTAL = 0

            # 确定疲劳提示:眨眼50次，打哈欠15次，瞌睡点头30次
            # if self.TOTAL >= 50 or self.mTOTAL >= 15 or self.hTOTAL >= 30:
            #     cv2.putText(im_rd, "SLEEP!!!", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
                # self.m_textCtrl3.AppendText(u"疲劳")

            # opencv中imread的图片内部是BGR排序，wxPython的StaticBitmap需要的图片是RGB排序，不转换会出现颜色变换
            height, width = im_rd.shape[:2]
            RGBImg = cv2.cvtColor(im_rd, cv2.COLOR_BGR2RGB)

            data = {'type':'img','value':RGBImg}
            self.thread_signal.emit(data)

            end_time = datetime.datetime.now()

            # 帧数
            res[2] = str(int(1 / (end_time- start_time).microseconds * 1000000))

            data = {'type': 'res', 'value': res}
            self.thread_signal.emit(data)

        # 释放摄像头
        self.cap.release()

    def value_changed(self):
        self.values[0] = self.spinBox_1.value()
        self.values[1] = self.spinBox_2.value()
        self.values[2] = self.spinBox_3.value()
        self.values[3] = self.spinBox_4.value()
        self.values[4] = self.spinBox_5.value()
        pass

    def select_changed(self):
        self.fun[0] = self.checkBox_11.isChecked()
        self.fun[1] = self.checkBox_12.isChecked()
        self.fun[2] = self.checkBox_21.isChecked()
        self.fun[3] = self.checkBox_22.isChecked()
        self.fun[4] = self.checkBox_31.isChecked()
        self.fun[5] = self.checkBox_32.isChecked()
        pass

    def button_clicked(self):
        if self.thread is not None and self.thread.is_alive():
            self.plainTextEdit_tip('已经开始')
        else:
            self.thread = threading.Thread(target=self._learning_face,daemon=True)
            self.thread.start()
        pass

    def thread_sound(self):
        pygame.mixer.init()
        pygame.mixer.music.load('1.mp3')
        pygame.mixer.music.play()
        time.sleep(15)
        pygame.mixer.music.stop()

    def paly_sound(self):
        if self.sound_thread is not None and self.sound_thread.is_alive():
            # self.plainTextEdit_tip('播放声音中')
            pass
        else:
            self.plainTextEdit_tip.appendPlainText('疲劳驾驶 播放声音')
            self.sound_thread = threading.Thread(target=self.thread_sound,daemon=True)
            self.sound_thread.start()
        pass

    def thread_singnal_slot(self, d):
        if d['type']=='img':
            RGBImg = d['value']
            # 将图片转化成Qt可读格式   QImage
            qimage = QImage(RGBImg, RGBImg.shape[1], RGBImg.shape[0], QImage.Format_RGB888)
            piximage = QtGui.QPixmap(qimage)
            # 显示图片
            self.label_img.setPixmap(piximage)
            #pic_show_label.setScaledContents(True)
        elif d['type'] == 'msg':
            self.plainTextEdit_tip.appendPlainText(d['value'])
        elif d['type'] == 'res':
            self.label_11.setText(d['value'][0])
            self.label_12.setText(d['value'][1])
            self.label_13.setText(d['value'][2])
            self.label_21.setText(d['value'][3])
            self.label_22.setText(d['value'][4])
            self.label_23.setText(d['value'][5])
            self.label_31.setText(d['value'][6])
            self.label_32.setText(d['value'][7])
            self.label_33.setText(d['value'][8])
            if d['value'][1] == '轻微疲劳':
                self.label_12.setStyleSheet("color:yellow;")
            elif  d['value'][1] == '瞌睡':
                self.label_12.setStyleSheet("color:red;")
                self.paly_sound()
            else:
                self.label_12.setStyleSheet("color:black;")

        pass

    def close(self) -> bool:
        self.cap.release()
        super(MainUI, self).close()



if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = MainUI()
    window.show()
    sys.exit(app.exec())
