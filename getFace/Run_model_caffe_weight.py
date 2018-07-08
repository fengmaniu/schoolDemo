import sys
import tools_matrix as tools
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from MTCNN import create_Kao_Onet, create_Kao_Rnet, create_Kao_Pnet
from imutils.video import FPS
import imutils
from threading import Thread
import threading


# will not work. caffe and TF incompatible


def detectFace(img, threshold):

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
        out_side = max(out_h, out_w)
        # print('calculating img scale #:', i)
        cls_prob = np.swapaxes(cls_prob, 0, 1)
        roi = np.swapaxes(roi, 0, 2)
        rectangle = tools.detect_face_12net(cls_prob, roi, out_side, 1 / scales[i], origin_w, origin_h, threshold[0])
        rectangles.extend(rectangle)
    rectangles = tools.NMS(rectangles, 0.7, 'iou')

    t1 = time.time()
    print ('time for 12 net is: ', t1-t0)

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
    print ('time for 24 net is: ', t2-t1)


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
    print ('time for 48 net is: ', t3-t2)

    return rectangles

def rectangleDraw(rectangles,img):
    draw = img.copy()
    for rectangle in rectangles:
        if rectangle is not None:
            W = -int(rectangle[0]) + int(rectangle[2])
            H = -int(rectangle[1]) + int(rectangle[3])
            paddingH = 0.01 * W
            paddingW = 0.02 * H
            crop_img = img[int(rectangle[1] + paddingH):int(rectangle[3] - paddingH),
                       int(rectangle[0] - paddingW):int(rectangle[2] + paddingW)]
            crop_img = cv2.cvtColor(crop_img, cv2.COLOR_RGB2GRAY)
            if crop_img is None:
                continue
            if crop_img.shape[0] < 0 or crop_img.shape[1] < 0:
                continue
            cv2.rectangle(draw, (int(rectangle[0]), int(rectangle[1])), (int(rectangle[2]), int(rectangle[3])),
                          (255, 0, 0), 1)

            for i in range(5, 15, 2):
                cv2.circle(draw, (int(rectangle[i + 0]), int(rectangle[i + 1])), 2, (0, 255, 0))
    return  draw


class myThread(threading.Thread):  # 继承父类threading.Thread

    def __init__(self, threadID,frame,lock,Pnet,Rnet,Onet):
        threading.Thread.__init__(self)
        self.lock = lock
        self.threadID = threadID
        self.frame = frame
        self.threshold = [0.6,0.6,0.7]
        self.Pnet = Pnet
        self.Rnet = Rnet
        self.Onet = Onet

    def run(self):  # 把要执行的代码写到run函数里面 线程在创建后会直接运行run函数
        self.lock.acquire()
        print(".........."+self.threadID+"..........")
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
                #temp_image = QImage(crop_img.flatten(), width, height, QImage.Format_RGB888)
                #temp_pixmap = QPixmap.fromImage(temp_image)
                #self.imgeLabel.setPixmap(temp_pixmap)

                cv2.imwrite('data/'+str(self.threadID)+'test.jpg', crop_img)
        return draw

if __name__ == "__main__":
    video_path = 'east.mp4'
    Pnet = create_Kao_Pnet(r'12net.h5')
    Rnet = create_Kao_Rnet(r'24net.h5')
    Onet = create_Kao_Onet(r'48net.h5')
    cap = cv2.VideoCapture(video_path)
    preTime = 0
    lock = threading.Lock()
    index = 0
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
    while (True):
        # ret,img = cap.read()
        # img = cv2.resize(img, (700, 400))
        # rectangles = detectFace(img, threshold)
        # draw = img.copy()
        # draw = rectangleDraw(rectangles,draw)
        # cv2.imshow("test", draw)
        #
        # c = cv2.waitKey(1) & 0xFF
        # if c == 27 or c == ord('q'):
        #     break

        start = time.time()

        # grab the frame from the threaded video file stream
        (grabbed, frame) = cap.read()
        # if the frame was not grabbed, then we have reached the end
        # of the stream
        if not grabbed:
            break
        # resize the frame and convert it to grayscale (while still
        # retaining 3 channels)
        frame = imutils.resize(frame, width=750)
        #thread1 = myThread(index, "Thread-1", 1, frame, threshold,lock,Pnet,Rnet,Onet)
        thread1 = myThread("Thread-"+str(index), frame, lock,Pnet,Rnet,Onet)
        thread1.start()
        index = index +1
        # if start - preTime > 1:
        #     preTime = start
        #     #需要加线程处理
        #rectangles = detectFace(frame, threshold)
        #frame = rectangleDraw(rectangles, frame)
            #cv2.imwrite('test.jpg', frame)
        # display a piece of text to the frame (so we can benchmark
        # end = time.time()
        # seconds = end - start
        # fps = 1 / seconds;
        # cv2.putText(frame,str(fps) , (10, 30),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow("Frame", frame)
        cv2.waitKey(1)



