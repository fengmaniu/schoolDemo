from keras_face.library.siamese import SiameseFaceNet
from PyQt5.QtWidgets import QApplication, QWidget
import sys
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from cv2 import *
import threading
import time
from PyQt5.QtGui import  QPixmap
from PyQt5.QtWidgets import QWidget, QApplication, QGroupBox, QPushButton, QLabel, QHBoxLayout,  QVBoxLayout, QGridLayout, QFormLayout, QLineEdit, QTextEdit
from MTCNN import create_Kao_Onet, create_Kao_Rnet, create_Kao_Pnet
import tools_matrix as tools
import numpy as np
from PIL import Image
import imutils

class VideoBox(QWidget):
    VIDEO_TYPE_OFFLINE = 0
    VIDEO_TYPE_REAL_TIME = 1

    STATUS_INIT = 0
    STATUS_PLAYING = 1
    STATUS_PAUSE = 2

    video_url = ""
    progress = 0
    def __init__(self, video_url="", video_type=VIDEO_TYPE_OFFLINE, auto_play=False):
        super(VideoBox, self).__init__()
        self.createGridGroupBox()
        self.creatVboxGroupBox()
        self.preTime = 0
        mainLayout = QVBoxLayout()
        hboxLayout = QHBoxLayout()
        hboxLayout.addStretch()
        hboxLayout.addWidget(self.gridGroupBox)
        hboxLayout.addWidget(self.vboxGroupBox)
        mainLayout.addLayout(hboxLayout)
        self.setLayout(mainLayout)
        self.threshold = [0.6, 0.6, 0.7]

        self.video_url = video_url
        self.video_type = video_type  # 0: offline  1: realTime
        self.auto_play = auto_play
        self.status = self.STATUS_INIT  # 0: init 1:playing 2: pause
        self.timer = VideoTimer()
        self.timer.timeSignal.signal[str].connect(self.show_video_images)
        # video 初始设置
        self.playCapture = VideoCapture()
        if self.video_url != "":
            self.set_timer_fps()
            if self.auto_play:
                self.switch_video()
        self.thread2 = threading.Thread(target=self.update_timer)
        self.thread2.setDaemon(True)
        self.thread2.start()

    def initNet(self,Pnet,Rnet,Onet,lock):
        self.Pnet=Pnet
        self.Rnet = Rnet
        self.Onet = Onet
        self.lock = lock

    def createGridGroupBox(self):
        self.gridGroupBox = QGroupBox("Grid layout")
        layout = QGridLayout()
        self.pictureLabel = QLabel()
        init_image = QPixmap("data/001.png").scaled(1000, 700)
        self.pictureLabel.setPixmap(init_image)
        self.threadId = 0
        self.playButton = QPushButton()
        self.playButton.setEnabled(True)
        self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.playButton.clicked.connect(self.switch_video)

        control_box = QHBoxLayout()
        control_box.setContentsMargins(0, 0, 0, 0)
        control_box.addWidget(self.playButton)
        llayout = QVBoxLayout()
        llayout.addWidget(self.pictureLabel)
        llayout.addLayout(control_box)
        layout = QHBoxLayout()
        layout.addLayout(llayout)
        self.gridGroupBox.setLayout(layout)
        self.setWindowTitle('Basic Layout')

    def creatVboxGroupBox(self):
        self.vboxGroupBox = QGroupBox("Vbox layout")
        layout = QVBoxLayout()
        self.imgeLabel = QLabel()
        init_image = QPixmap("data/001.png").scaled(300, 300)
        self.imgeLabel.setPixmap(init_image)

        imgeLabel_1 = QLabel()
        pixMap_1 = QPixmap("data/001.png")
        imgeLabel_1.setPixmap(pixMap_1)

        imgeLabel_2 = QLabel()
        pixMap_2 = QPixmap("data/001.png")
        imgeLabel_2.setPixmap(pixMap_2)

        layout.addWidget(imgeLabel_2)
        layout.addWidget(imgeLabel_1)
        layout.addWidget(self.imgeLabel)
        self.vboxGroupBox.setLayout(layout)

    def reset(self):
        self.timer.stop()
        self.playCapture.release()
        self.status = VideoBox.STATUS_INIT
        self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))

    def set_timer_fps(self):
        self.playCapture.open(self.video_url)
        fps = self.playCapture.get(CAP_PROP_FPS)
        self.timer.set_fps(fps)
        self.playCapture.release()

    def set_video(self, url, video_type=VIDEO_TYPE_OFFLINE, auto_play=False):
        self.reset()
        self.video_url = url
        self.video_type = video_type
        self.auto_play = auto_play
        self.set_timer_fps()
        if self.auto_play:
            self.switch_video()

    def play(self):
        if self.video_url == "" or self.video_url is None:
            return
        if not self.playCapture.isOpened():
            self.playCapture.open(self.video_url)
        self.timer.start()
        self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
        self.status = VideoBox.STATUS_PLAYING

    def stop(self):
        if self.video_url == "" or self.video_url is None:
            return
        if self.playCapture.isOpened():
            self.timer.stop()
            if self.video_type is VideoBox.VIDEO_TYPE_REAL_TIME:
                self.playCapture.release()
            self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.status = VideoBox.STATUS_PAUSE

    def re_play(self):
        if self.video_url == "" or self.video_url is None:
            return
        self.playCapture.release()
        self.playCapture.open(self.video_url)
        self.timer.start()
        self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
        self.status = VideoBox.STATUS_PLAYING

    def detectFace(self,img, threshold):

        caffe_img = (img.copy() - 127.5) / 127.5
        origin_h, origin_w, ch = caffe_img.shape
        scales = tools.calculateScales(img)
        out = []
        t0 = time.time()
        # del scales[:4]

        for scale in scales:
            hs = int(origin_h * scale)
            ws = int(origin_w * scale)
            scale_img = cv2.resize(caffe_img, (ws, hs))
            input = scale_img.reshape(1, *scale_img.shape)
            ouput = Pnet.predict(input)  # .transpose(0,2,1,3) should add, but seems after process is wrong then.
            out.append(ouput)
        image_num = len(scales)
        rectangles = []
        for i in range(image_num):
            cls_prob = out[i][0][0][:, :,
                       1]  # i = #scale, first 0 select cls score, second 0 = batchnum, alway=0. 1 one hot repr
            roi = out[i][1][0]
            out_h, out_w = cls_prob.shape
            out_side = out_w
            if out_h>out_w:
                out_side = out_h
            #out_side = max(out_h, out_w)
            # print('calculating img scale #:', i)
            cls_prob = np.swapaxes(cls_prob, 0, 1)
            roi = np.swapaxes(roi, 0, 2)
            rectangle = tools.detect_face_12net(cls_prob, roi, out_side, 1 / scales[i], origin_w, origin_h,
                                                threshold[0])
            rectangles.extend(rectangle)
        rectangles = tools.NMS(rectangles, 0.7, 'iou')

        t1 = time.time()
        print('time for 12 net is: ', t1 - t0)

        if len(rectangles) == 0:
            return rectangles

        crop_number = 0
        out = []
        predict_24_batch = []
        for rectangle in rectangles:
            crop_img = caffe_img[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]
            scale_img = cv2.resize(crop_img, (24, 24))
            predict_24_batch.append(scale_img)
            crop_number += 1

        predict_24_batch = np.array(predict_24_batch)

        out = Rnet.predict(predict_24_batch)

        cls_prob = out[0]  # first 0 is to select cls, second batch number, always =0
        cls_prob = np.array(cls_prob)  # convert to numpy
        roi_prob = out[1]  # first 0 is to select roi, second batch number, always =0
        roi_prob = np.array(roi_prob)
        rectangles = tools.filter_face_24net(cls_prob, roi_prob, rectangles, origin_w, origin_h, threshold[1])
        t2 = time.time()
        print('time for 24 net is: ', t2 - t1)

        if len(rectangles) == 0:
            return rectangles

        crop_number = 0
        predict_batch = []
        for rectangle in rectangles:
            # print('calculating net 48 crop_number:', crop_number)
            crop_img = caffe_img[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]
            scale_img = cv2.resize(crop_img, (48, 48))
            predict_batch.append(scale_img)
            crop_number += 1

        predict_batch = np.array(predict_batch)

        output = Onet.predict(predict_batch)
        cls_prob = output[0]
        roi_prob = output[1]
        pts_prob = output[2]  # index
        # rectangles = tools.filter_face_48net_newdef(cls_prob, roi_prob, pts_prob, rectangles, origin_w, origin_h,
        #                                             threshold[2])
        rectangles = tools.filter_face_48net(cls_prob, roi_prob, pts_prob, rectangles, origin_w, origin_h, threshold[2])
        t3 = time.time()
        print('time for 48 net is: ', t3 - t2)

        return rectangles

    def rectangleDraw(self , rectangles, img):
        draw = img.copy()
        for rectangle in rectangles:
            if rectangle is not None:
                W = -int(rectangle[0]) + int(rectangle[2])
                H = -int(rectangle[1]) + int(rectangle[3])
                paddingH = 0.01 * W
                paddingW = 0.02 * H
                crop_img = img[int(rectangle[1] + paddingH):int(rectangle[3] - paddingH),
                           int(rectangle[0] - paddingW):int(rectangle[2] + paddingW)]
                crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
                if crop_img is None:
                    continue
                if crop_img.shape[0] < 0 or crop_img.shape[1] < 0:
                    continue
                cv2.rectangle(draw, (int(rectangle[0]), int(rectangle[1])), (int(rectangle[2]), int(rectangle[3])),
                              (255, 0, 0), 1)
                crop_img = imutils.resize(crop_img, width=100)
                height, width = crop_img.shape[:2]
                # if crop_img.ndim == 3:
                #     rgb = cvtColor(crop_img, COLOR_BGR2RGB)
                # elif crop_img.ndim == 2:
                #     rgb = cvtColor(crop_img, COLOR_GRAY2BGR)
                temp_image = QImage(crop_img.flatten(), width, height, QImage.Format_RGB888)
                temp_pixmap = QPixmap.fromImage(temp_image)
                self.imgeLabel.setPixmap(temp_pixmap)
        return draw

    def show_video_images(self):

        if self.playCapture.isOpened():
            success, frame = self.playCapture.read()
            if success:
                start = time.time()
                frame = imutils.resize(frame, width=1000)
                # if start - self.preTime > 1:
                #     self.preTime = start
                #     rectangles = self.detectFace(frame, threshold)
                #     frame = self.rectangleDraw(rectangles, frame)
                thread1 = myThread(self.threadId ,frame,self.Pnet,self.Rnet,self.Onet,lock,self.imgeLabel)
                self.threadId = self.threadId + 1
                thread1.start()
                #rectangles = self.detectFace(frame, threshold)
                #frame = self.rectangleDraw(rectangles, frame)
                end = time.time()
                #print(end - start)
                height, width = frame.shape[:2]
                if frame.ndim == 3:
                    rgb = cvtColor(frame, COLOR_BGR2RGB)
                elif frame.ndim == 2:
                    rgb = cvtColor(frame, COLOR_GRAY2BGR)

                temp_image = QImage(rgb.flatten(), width, height, QImage.Format_RGB888)
                temp_pixmap = QPixmap.fromImage(temp_image)
                self.pictureLabel.setPixmap(temp_pixmap)
                #self.preTime
                # frame = cv2.resize(frame, (1000, 700))
                # rectangles = self.detectFace(frame, threshold)
                # draw = frame.copy()
                # for rectangle in rectangles:
                #     if rectangle is not None:
                #         W = -int(rectangle[0]) + int(rectangle[2])
                #         H = -int(rectangle[1]) + int(rectangle[3])
                #         paddingH = 0.01 * W
                #         paddingW = 0.02 * H
                #         crop_img = frame[int(rectangle[1] + paddingH):int(rectangle[3] - paddingH),
                #                    int(rectangle[0] - paddingW):int(rectangle[2] + paddingW)]
                #         crop_img = cv2.cvtColor(crop_img, cv2.COLOR_RGB2GRAY)
                #         if crop_img is None:
                #             continue
                #         if crop_img.shape[0] < 0 or crop_img.shape[1] < 0:
                #             continue
                #         cv2.rectangle(draw, (int(rectangle[0]), int(rectangle[1])),
                #                       (int(rectangle[2]), int(rectangle[3])),
                #                       (255, 0, 0), 1)
                #         # cv2.imwrite("a.bmp", crop_img)
                #         #self.imgeLabel.setPixmap(crop_img)
                #         height, width = crop_img.shape[:2]
                #         if crop_img.ndim == 3:
                #             rgb = cvtColor(crop_img, COLOR_BGR2RGB)
                #         elif crop_img.ndim == 2:
                #             rgb = cvtColor(crop_img, COLOR_GRAY2BGR)
                #         temp_image = QImage(rgb.flatten(), width, height, QImage.Format_RGB888)
                #         temp_pixmap = QPixmap.fromImage(temp_image).scaled(200, 200)
                #         self.imgeLabel.setPixmap(temp_pixmap)
                #
                # height, width = draw.shape[:2]
                # if draw.ndim == 3:
                #     rgb = cvtColor(draw, COLOR_BGR2RGB)
                # elif draw.ndim == 2:
                #     rgb = cvtColor(draw, COLOR_GRAY2BGR)
                #
                # temp_image = QImage(rgb.flatten(), width, height, QImage.Format_RGB888)
                # temp_pixmap = QPixmap.fromImage(temp_image).scaled(1000, 700)
                # self.pictureLabel.setPixmap(temp_pixmap)

            else:
                print("read failed, no frame data")
                success, frame = self.playCapture.read()
                if not success and self.video_type is VideoBox.VIDEO_TYPE_OFFLINE:
                    print("play finished")  # 判断本地文件播放完毕
                    self.reset()
                    self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaStop))
                return
        else:
            print("open file or capturing device error, init again")
            self.reset()

    def update_timer(self):
        while (True):
            if self.status is VideoBox.STATUS_PLAYING:
                self.progress = self.progress + 1
                time.sleep(0.4)
                if (self.progress == 15):
                    self.progress = 0

    def switch_video(self):
        if self.video_url == "" or self.video_url is None:
            return
        if self.status is VideoBox.STATUS_INIT:
            self.playCapture.open(self.video_url)
            self.timer.start()
            self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
        elif self.status is VideoBox.STATUS_PLAYING:
            self.timer.stop()
            if self.video_type is VideoBox.VIDEO_TYPE_REAL_TIME:
                self.playCapture.release()
            self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        elif self.status is VideoBox.STATUS_PAUSE:
            if self.video_type is VideoBox.VIDEO_TYPE_REAL_TIME:
                self.playCapture.open(self.video_url)
            self.timer.start()
            self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))

        self.status = (VideoBox.STATUS_PLAYING,
                       VideoBox.STATUS_PAUSE,
                       VideoBox.STATUS_PLAYING)[self.status]


class Communicate(QObject):
    signal = pyqtSignal(str)


class myThread(threading.Thread, VideoBox):  # 继承父类threading.Thread
    def __init__(self, threadID,frame,Pnet,Rnet,Onet,lock,imgeLabel):
        threading.Thread.__init__(self)
        #VideoBox.__init__(self)
        self.threadID = threadID
        self.frame = frame
        self.preTime = 0
        self.Pnet = Pnet
        self.Rnet = Rnet
        self.Onet = Onet
        self.lock = lock
        self.threshold = [0.6, 0.6, 0.7]
        self.imgeLabel =imgeLabel

    def run(self):  # 把要执行的代码写到run函数里面 线程在创建后会直接运行run函数
        self.lock.acquire()
        rectangles = self.detectFace(self.frame, self.threshold)
        frame = self.rectangleDraw(rectangles, self.frame)
        self.lock.release()
    def detectFace(self,img, threshold):

        caffe_img = (img.copy() - 127.5) / 127.5
        origin_h, origin_w, ch = caffe_img.shape
        scales = tools.calculateScales(img)
        out = []
        t0 = time.time()
        # del scales[:4]

        for scale in scales:
            hs = int(origin_h * scale)
            ws = int(origin_w * scale)
            scale_img = cv2.resize(caffe_img, (ws, hs))
            input = scale_img.reshape(1, *scale_img.shape)
            ouput = self.Pnet.predict(input)  # .transpose(0,2,1,3) should add, but seems after process is wrong then.
            out.append(ouput)
        image_num = len(scales)
        rectangles = []
        for i in range(image_num):
            cls_prob = out[i][0][0][:, :,
                       1]  # i = #scale, first 0 select cls score, second 0 = batchnum, alway=0. 1 one hot repr
            roi = out[i][1][0]
            out_h, out_w = cls_prob.shape
            out_side = out_w
            if out_h>out_w:
                out_side = out_h
            #out_side = max(out_h, out_w)
            # print('calculating img scale #:', i)
            cls_prob = np.swapaxes(cls_prob, 0, 1)
            roi = np.swapaxes(roi, 0, 2)
            rectangle = tools.detect_face_12net(cls_prob, roi, out_side, 1 / scales[i], origin_w, origin_h,
                                                threshold[0])
            rectangles.extend(rectangle)
        rectangles = tools.NMS(rectangles, 0.7, 'iou')

        t1 = time.time()
        print('time for 12 net is: ', t1 - t0)

        if len(rectangles) == 0:
            return rectangles

        crop_number = 0
        out = []
        predict_24_batch = []
        for rectangle in rectangles:
            crop_img = caffe_img[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]
            scale_img = cv2.resize(crop_img, (24, 24))
            predict_24_batch.append(scale_img)
            crop_number += 1

        predict_24_batch = np.array(predict_24_batch)

        out = self.Rnet.predict(predict_24_batch)

        cls_prob = out[0]  # first 0 is to select cls, second batch number, always =0
        cls_prob = np.array(cls_prob)  # convert to numpy
        roi_prob = out[1]  # first 0 is to select roi, second batch number, always =0
        roi_prob = np.array(roi_prob)
        rectangles = tools.filter_face_24net(cls_prob, roi_prob, rectangles, origin_w, origin_h, threshold[1])
        t2 = time.time()
        print('time for 24 net is: ', t2 - t1)

        if len(rectangles) == 0:
            return rectangles

        crop_number = 0
        predict_batch = []
        for rectangle in rectangles:
            # print('calculating net 48 crop_number:', crop_number)
            crop_img = caffe_img[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]
            scale_img = cv2.resize(crop_img, (48, 48))
            predict_batch.append(scale_img)
            crop_number += 1

        predict_batch = np.array(predict_batch)

        output = self.Onet.predict(predict_batch)
        cls_prob = output[0]
        roi_prob = output[1]
        pts_prob = output[2]  # index
        # rectangles = tools.filter_face_48net_newdef(cls_prob, roi_prob, pts_prob, rectangles, origin_w, origin_h,
        #                                             threshold[2])
        rectangles = tools.filter_face_48net(cls_prob, roi_prob, pts_prob, rectangles, origin_w, origin_h, threshold[2])
        t3 = time.time()
        print('time for 48 net is: ', t3 - t2)

        return rectangles

    def rectangleDraw(self , rectangles, img):
        draw = img.copy()
        for rectangle in rectangles:
            if rectangle is not None:
                W = -int(rectangle[0]) + int(rectangle[2])
                H = -int(rectangle[1]) + int(rectangle[3])
                paddingH = 0.01 * W
                paddingW = 0.02 * H
                crop_img = img[int(rectangle[1] + paddingH):int(rectangle[3] - paddingH),
                           int(rectangle[0] - paddingW):int(rectangle[2] + paddingW)]
                crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
                if crop_img is None:
                    continue
                if crop_img.shape[0] < 0 or crop_img.shape[1] < 0:
                    continue
                cv2.rectangle(draw, (int(rectangle[0]), int(rectangle[1])), (int(rectangle[2]), int(rectangle[3])),
                              (255, 0, 0), 1)
                crop_img = imutils.resize(crop_img, width=100)
                height, width = crop_img.shape[:2]
                # if crop_img.ndim == 3:
                #     rgb = cvtColor(crop_img, COLOR_BGR2RGB)
                # elif crop_img.ndim == 2:
                #     rgb = cvtColor(crop_img, COLOR_GRAY2BGR)
                temp_image = QImage(crop_img.flatten(), width, height, QImage.Format_RGB888)
                temp_pixmap = QPixmap.fromImage(temp_image)
                self.imgeLabel.setPixmap(temp_pixmap)
                #cv2.imwrite('data/' + str(self.threadID) + 'test.jpg', crop_img)
        return draw


class VideoTimer(QThread):

    def __init__(self, frequent=20):
        QThread.__init__(self)
        self.stopped = False
        self.frequent = frequent
        self.timeSignal = Communicate()
        self.mutex = QMutex()

    def run(self):
        with QMutexLocker(self.mutex):
            self.stopped = False
        while True:
            if self.stopped:
                return
            self.timeSignal.signal.emit("1")
            time.sleep(1 / self.frequent)

    def stop(self):
        with QMutexLocker(self.mutex):
            self.stopped = True

    def is_stopped(self):
        with QMutexLocker(self.mutex):
            return self.stopped

    def set_fps(self, fps):
        self.frequent = fps

class PredictImg():
    def __init__(self):
        self.data = ""

    def predict(self):
        fnet = SiameseFaceNet()
        model_dir_path = './models_1'
        image_dir_path = "./data/images"
        fnet.load_model(model_dir_path)
        database = dict()
        database["aipengfei"] = [fnet.img_to_encoding(image_dir_path + "/aipengfei.png")]
        database["anyaru"] = [fnet.img_to_encoding(image_dir_path + "/anyaru.png")]
        database["baozhiqian"] = [fnet.img_to_encoding(image_dir_path + "/baozhiqian.png")]
        print("-------------------------")
        fnet.verify(image_dir_path + "/001.png", "aipengfei", database)
        fnet.verify(image_dir_path + "/002.png", "aipengfei", database)
        fnet.who_is_it(image_dir_path + "/001.png", database)
        fnet.who_is_it(image_dir_path + "/002.png", database)
        fnet.who_is_it(image_dir_path + "/003.png", database)
if __name__ == "__main__":
    mapp = QApplication(sys.argv)
    Pnet = create_Kao_Pnet(r'12net.h5')
    Rnet = create_Kao_Rnet(r'24net.h5')
    Onet = create_Kao_Onet(r'48net.h5')  # will not work. caffe and TF incompatible
    img = cv2.imread('0001.png')
    scale_img = cv2.resize(img, (100, 100))
    input = scale_img.reshape(1, *scale_img.shape)
    Pnet.predict(input)
    img = cv2.imread('0001.png')
    scale_img = cv2.resize(img, (24, 24))
    input = scale_img.reshape(1, *scale_img.shape)
    Rnet.predict(input)
    img = cv2.imread('0001.png')
    scale_img = cv2.resize(img, (48, 48))
    input = scale_img.reshape(1, *scale_img.shape)
    Onet.predict(input)
    video_path = 'east.mp4'
    cap = cv2.VideoCapture(video_path)
    mw = VideoBox()
    lock = threading.Lock()
    mw.initNet(Pnet,Rnet,Onet,lock)
    mw.set_video("east.mp4", VideoBox.VIDEO_TYPE_OFFLINE, False)
    mw.show()

    # predict =PredictImg()
    # predict.predict()
    sys.exit(mapp.exec_())