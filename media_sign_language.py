import cv2
import mediapipe as mp
import time
import numpy as np
import math
from PIL import ImageFont, ImageDraw, Image


class sign_language():
    def __init__(self):
        super(sign_language, self).__init__()
        mpPose = mp.solutions.pose  # 姿态识别方法
        self.pose = mpPose.Pose(static_image_mode=True,  # 静态图模式，False代表置信度高时继续跟踪，True代表实时跟踪检测新的结果
                           upper_body_only=True,  # 是否只检测上半身
                           smooth_landmarks=True,  # 平滑，一般为True
                           min_detection_confidence=0.1,  # 检测置信度
                           min_tracking_confidence=0.1)  # 跟踪置信度
        # 检测置信度大于0.5代表检测到了，若此时跟踪置信度大于0.5就继续跟踪，小于就沿用上一次，避免一次又一次重复使用模型
        mpHands = mp.solutions.hands
        self.hands = mpHands.Hands()
        self.mpDraw = mp.solutions.drawing_utils
        self.lad_hand_L2 = np.empty((21, 2))
        self.figure_hands = np.zeros(5)

    def putChtext(self, image, text, pos, color, size, font='simsun.ttc'):

        font = ImageFont.truetype(font, 45)
        image = Image.fromarray(image)
        canvas = ImageDraw.Draw(image)
        canvas.text(pos, text, fill=color, font=font)
        return np.array(image)

    def finger_stretch_detect(self, point1, point2, point3):
        # print(point1, point2, point3, type(point2))
        result = 0
        # 计算向量的L2范数
        dist1 = np.linalg.norm((point2 - point1), ord=2)
        dist2 = np.linalg.norm((point3 - point1), ord=2)
        if dist2 > dist1:
            result = 1
        return result

    def Key_Points_L2_One_base(self, point1, points):
        """
        判断第一个点与其他点之间的距离（L2范数），数据类型为np.array
        :param point1: 一个基准点
        :param points: 其余点
        :return: 返回和其余点个数相同的序号，最靠近的point1的为0
        """
        # point1 = np.array([100, 21])
        # points = [np.array([1.0, 2.0]), np.array([100.0, 22.0]), np.array([1000.0, 220.0])]
        n = len(points)
        result = []
        for i in range(n):
            result.append(np.linalg.norm((points[i] - point1), ord=2))
        x = sorted(result)
        output = []
        for i in result:
            a = x.index(i)
            output.append(a)
        return output

    def Key_Points_L2_twos(self, points1, points2):
        """
        判断两组点之间的距离（L2范数）关系，数据类型为np.array
        :param points1: 一组比较点
        :param points2: 对应的另一组
        :return: 返回和点组长度相同的序号，最靠近的point1的为0
        """
        n = len(points1)
        result = []
        for i in range(n):
            result.append(np.linalg.norm((points1[i], points2[i]), ord=2))
        x = sorted(result)
        output = []
        for i in result:
            a = x.index(i)
            output.append(a)
        return output

    def direct_vector(self, p1, p2):
        return [int(p2[0]-p1[0]), int(p2[1]-p1[1])]

    def Angle_count(self, point_base, point_1, point_2):
        """
        计算两点和基准点之间的角度，锐角直接显示，钝角则小于0
        :param point_base: 基准点
        :param point_1: 点1
        :param point_2: 点2
        :return: 返回计算的角度，以int角度制返回，钝角小于0，锐角大于0
        """
        p1 = self.direct_vector(point_base, point_1)
        p2 = self.direct_vector(point_base, point_2)
        try:
            angR = (p1[0]*p2[0] + p1[1]*p2[1])/((math.sqrt(p1[0]**2 + p1[1]**2)*(math.sqrt(p2[0]**2 + p2[1]**2))))
        except:
            angR = 0
        angD = round(math.degrees(angR))
        return angD

    def detect_hands_gesture(self, result, angle=0):
        if (result[0] == 1) and (result[1] == 0) and (result[2] == 0) and (result[3] == 0) and (result[4] == 0):
            gesture = "好的"
        elif (result[0] == 0) and (result[1] == 1) and (result[2] == 0) and (result[3] == 0) and (result[4] == 0):
            gesture = "壹"
        elif (result[0] == 0) and (result[1] == 0) and (result[2] == 1) and (result[3] == 0) and (result[4] == 0):
            gesture = "不要竖中指"
        elif (result[0] == 0) and (result[1] == 1) and (result[2] == 1) and (result[3] == 0) and (result[4] == 0):
            gesture = "贰"
        elif (result[0] == 0) and (result[1] == 1) and (result[2] == 1) and (result[3] == 1) and (result[4] == 0):
            gesture = "叁"
        elif (result[0] == 0) and (result[1] == 1) and (result[2] == 1) and (result[3] == 1) and (result[4] == 1):
            gesture = "肆"
        elif (result[0] == 1) and (result[1] == 1) and (result[2] == 1) and (result[3] == 1) and (result[4] == 1):
            gesture = "伍"
        elif (result[0] == 1) and (result[1] == 0) and (result[2] == 0) and (result[3] == 0) and (result[4] == 1):
            gesture = "陆"
        elif (result[0] == 1) and (result[1] == 1) and (result[2] == 1) and (result[3] == 0) and (result[4] == 0):
            gesture = "柒"
        elif (result[0] == 1) and (result[1] == 1) and (result[2] == 0) and (result[3] == 0) and (result[4] == 0):
            gesture = "捌"
        elif (result[0] == 0) and (result[1] == 0) and (result[2] == 0) and (result[3] == 0) and (
                result[4] == 0) and (angle < 0):
            gesture = "玖"
        elif (result[0] == 0) and (result[1] == 0) and (result[2] == 1) and (result[3] == 1) and (result[4] == 1):
            gesture = "OK"
        elif (result[0] == 0) and (result[1] == 0) and (result[2] == 0) and (result[3] == 0) and (
                result[4] == 0) and (angle > 0):
            gesture = "零"
        else:
            gesture = None

        return gesture

    def detect_father(self, img, pb_20, pb_14, pb_12, pb_10, lad_hand_L2, list_text):
        hand = self.Key_Points_L2_One_base(pb_10, [pb_12, pb_14, pb_20])
        # print(pb_10)
        # cv2.circle(img, (int(pb_10[0]), int(pb_10[1])), 3, (0, 255, 0), cv2.FILLED)
        # cv2.circle(img, (int(pb_12[0]), int(pb_12[1])), 3, (0, 255, 0), cv2.FILLED)
        # cv2.circle(img, (int(pb_14[0]), int(pb_14[1])), 3, (0, 255, 0), cv2.FILLED)
        # cv2.circle(img, (int(pb_20[0]), int(pb_20[1])), 3, (0, 255, 0), cv2.FILLED)
        # print(hand)
        if hand == [1, 2, 0] and "好的" in list_text:
            return True
        return False

    def detect_human(self, img, pb_20, pb_14, pb_19, pb_13, lad_hand_L2, list_text):
        hand = self.Key_Points_L2_twos([pb_20, pb_14], [pb_19, pb_13])
        # cv2.circle(img, (int(pb_19[0]), int(pb_19[1])), 3, (0, 255, 0), cv2.FILLED)
        # cv2.circle(img, (int(pb_13[0]), int(pb_13[1])), 3, (0, 255, 0), cv2.FILLED)
        # cv2.circle(img, (int(pb_14[0]), int(pb_14[1])), 3, (0, 255, 0), cv2.FILLED)
        # cv2.circle(img, (int(pb_20[0]), int(pb_20[1])), 3, (0, 255, 0), cv2.FILLED)
        # print(hand)
        if hand == [0, 1] and "壹" in list_text:
            return True
        return False

    def detect_mother(self, img, pb_20, pb_14, pb_12, pb_10, lad_hand_L2, list_text):
        hand = self.Key_Points_L2_One_base(pb_10, [pb_12, pb_14, pb_20])
        # print(pb_10)
        # cv2.circle(img, (int(pb_10[0]), int(pb_10[1])), 3, (0, 255, 0), cv2.FILLED)
        # cv2.circle(img, (int(pb_12[0]), int(pb_12[1])), 3, (0, 255, 0), cv2.FILLED)
        # cv2.circle(img, (int(pb_14[0]), int(pb_14[1])), 3, (0, 255, 0), cv2.FILLED)
        # cv2.circle(img, (int(pb_20[0]), int(pb_20[1])), 3, (0, 255, 0), cv2.FILLED)
        # print(hand)
        if hand == [1, 2, 0] and "壹" in list_text:
            return True
        return False

    def detect_you(self, lad_hand_L2, list_text):
        hand = self.Key_Points_L2_One_base(lad_hand_L2[5], [lad_hand_L2[6], lad_hand_L2[7], lad_hand_L2[8], lad_hand_L2[13]])
        if hand[-1] == 3:
            return True
        return False

    def run(self, img):
        # （2）处理每一帧图像
        # 接收图片是否导入成功、帧图像
        # success, img = cap.read()
        # 将导入的BGR格式图像转为RGB格式
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results_hands = self.hands.process(imgRGB)
        # 将图像传给姿态识别模型
        results = self.pose.process(imgRGB)
        lm_list_body = np.empty((25, 2))
        text = ''
        list_text = []
        # 如果检测到体态就执行下面内容，没检测到就不执行
        if results.pose_landmarks:
            # 绘制姿态坐标点，img为画板，传入姿态点坐标，坐标连线
            for index, lm in enumerate(results.pose_landmarks.landmark):
                h, w, _ = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list_body[index, :] = [cx, cy]
                # cv2.circle(img, (cx, cy), 2, (0, 255, 0), cv2.FILLED)
        # 绘制手部关键点
        if results_hands.multi_hand_landmarks:
            for i, handLms in enumerate(results_hands.multi_hand_landmarks):
                # i是手的索引，不分左右手
                for id, lm in enumerate(handLms.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lad_ = [cx, cy]
                    self.lad_hand_L2[id, :] = lad_
                    for k in range(5):
                        if k == 0:
                            figure_ = self.finger_stretch_detect(self.lad_hand_L2[17], self.lad_hand_L2[4 * k + 2], self.lad_hand_L2[4 * k + 4])
                        else:
                            figure_ = self.finger_stretch_detect(self.lad_hand_L2[0], self.lad_hand_L2[4 * k + 2], self.lad_hand_L2[4 * k + 4])
                        self.figure_hands[k] = figure_
                    angle = self.Angle_count(self.lad_hand_L2[5], self.lad_hand_L2[8], self.lad_hand_L2[0])
                    gesture_hands_result = self.detect_hands_gesture(self.figure_hands, angle)
                list_text.append(gesture_hands_result)
                print(list_text)
                print(i)
                # cv2.putText(img, f"{gesture_hands_result}", (20, 40 * (i+1)), cv2.FONT_HERSHEY_COMPLEX, 1,
                #            (255, 255, 0), 2)

            if results.pose_landmarks:
                if self.detect_you(self.lad_hand_L2, list_text):
                    text = '你'
                elif self.detect_human(img, lm_list_body[20], lm_list_body[14], lm_list_body[19], lm_list_body[13], self.lad_hand_L2, list_text):
                    text = '人'
                elif self.detect_father(img, lm_list_body[20], lm_list_body[14], lm_list_body[12], lm_list_body[10], self.lad_hand_L2, list_text):
                    text = '父亲'
                elif self.detect_mother(img, lm_list_body[20], lm_list_body[14], lm_list_body[12], lm_list_body[10], self.lad_hand_L2, list_text):
                    text = '母亲'

            for i, j in enumerate(list_text):
                if j != None:
                    img = self.putChtext(img, f"{j}", (20, 40 * (i+1)),
                               (255, 0, 0), 2)

            if text != None:
                img = self.putChtext(img, text, (20, 40), (255, 0, 0), 2)

            # cv2.putText(img, text, (20, 40), cv2.FONT_HERSHEY_COMPLEX, 1,
            #             (255, 255, 255), 2)
            # 绘制手部特征点：
            self.mpDraw.draw_landmarks(img, handLms, mp.solutions.hands.HAND_CONNECTIONS)
        print('识别完成——————————————————————')
        return img
        # 显示图像，输入窗口名及图像数据
        # cv2.imshow('image', img)
        # cv2.waitKey(1)
        # 释放视频资源
        # cap.release()
        # cv2.destroyAllWindows()

if __name__ == "__main__":
    sign_lang = sign_language()
    start = time.time()
    cap = cv2.VideoCapture(0)
    success, img = cap.read()
    # cap.release()
    while success:
        img = sign_lang.run(img)
        cv2.resize(img, (480, 640))
        print('识别完成*******')

        end = time.time()
        seconds = end - start
        fps = 1 / seconds
        start = time.time()
        cv2.putText(img, str(fps), (550, 40), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('jjj', img)
        cv2.waitKey(1)
        success, img = cap.read()

