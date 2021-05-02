import glob
import os
import sys

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================


import carla

from carla import ColorConverter as cc
import cv2
import argparse
import collections
import datetime
import logging
import math
import random
import re
import weakref

class CameraManager(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self.surface = None
        self._parent = parent_actor
        self.hud = hud
        self.recording = False
        self._camera_transforms = [
            carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15)),
            carla.Transform(carla.Location(x=1.6, z=1.7))]
        self.transform_index = 1
        self.sensors = [
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB'],
            ['sensor.camera.depth', cc.Raw, 'Camera Depth (Raw)'],
            ['sensor.camera.depth', cc.Depth, 'Camera Depth (Gray Scale)'],
            ['sensor.camera.depth', cc.LogarithmicDepth, 'Camera Depth (Logarithmic Gray Scale)'],
            ['sensor.camera.semantic_segmentation', cc.Raw, 'Camera Semantic Segmentation (Raw)'],
            ['sensor.camera.semantic_segmentation', cc.CityScapesPalette,
             'Camera Semantic Segmentation (CityScapes Palette)'],
            ['sensor.lidar.ray_cast', None, 'Lidar (Ray-Cast)']]
        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()
        for item in self.sensors:
            bp = bp_library.find(item[0])
            if item[0].startswith('sensor.camera'):
                bp.set_attribute('image_size_x', str(hud.dim[0]))
                bp.set_attribute('image_size_y', str(hud.dim[1]))
            elif item[0].startswith('sensor.lidar'):
                bp.set_attribute('range', '5000')
            item.append(bp)
        self.index = None

    def toggle_camera(self):
        self.transform_index = (self.transform_index + 1) % len(self._camera_transforms)
        self.sensor.set_transform(self._camera_transforms[self.transform_index])

    def set_sensor(self, index, notify=True):
        index = index % len(self.sensors)
        needs_respawn = True if self.index is None \
            else self.sensors[index][0] != self.sensors[self.index][0]
        if needs_respawn:
            if self.sensor is not None:
                self.sensor.destroy()
                self.surface = None
            self.sensor = self._parent.get_world().spawn_actor(
                self.sensors[index][-1],
                self._camera_transforms[self.transform_index],
                attach_to=self._parent)
            # We need to pass the lambda a weak reference to self to avoid
            # circular reference.
            weak_self = weakref.ref(self)
            self.sensor.listen(lambda image: CameraManager._parse_image(weak_self, image))
        if notify:
            self.hud.notification(self.sensors[index][2])
        self.index = index

    def next_sensor(self):
        self.set_sensor(self.index + 1)

    def toggle_recording(self):
        self.recording = not self.recording
        self.hud.notification('Recording %s' % ('On' if self.recording else 'Off'))

    def render(self, display):
        if self.surface is not None:
            display.blit(self.surface, (0, 0))

    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self()
        if not self:
            return
        if self.sensors[self.index][0].startswith('sensor.lidar'):
            points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
            points = np.reshape(points, (int(points.shape[0] / 3), 3))
            lidar_data = np.array(points[:, :2])
            lidar_data *= min(self.hud.dim) / 100.0
            lidar_data += (0.5 * self.hud.dim[0], 0.5 * self.hud.dim[1])
            lidar_data = np.fabs(lidar_data)  # pylint: disable=E1111
            lidar_data = lidar_data.astype(np.int32)
            lidar_data = np.reshape(lidar_data, (-1, 2))
            lidar_img_size = (self.hud.dim[0], self.hud.dim[1], 3)
            lidar_img = np.zeros(lidar_img_size)
            lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
            self.surface = pygame.surfarray.make_surface(lidar_img)
        else:
            # Original code Don't touch
            image.convert(self.sensors[self.index][1])
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))

            #################################################
            # it's my code
            pt1_sum_ri = (0, 0)
            pt2_sum_ri = (0, 0)
            pt1_avg_ri = (0, 0)
            count_posi_num_ri = 0

            pt1_sum_le = (0, 0)
            pt2_sum_le = (0, 0)
            pt1_avg_le = (0, 0)

            count_posi_num_le = 0

            test_im = np.array(image.raw_data)
            test_im = test_im.copy()
            test_im = test_im.reshape((image.height, image.width, 4))
            test_im = test_im[:, :, :3]
            # copy_im = test_im[:, :, :3]
            # test_im = copy_im.copy()
            #################################################

            #################################################
            # Now image resolution is 720x1280x3
            size_im = cv2.resize(test_im, dsize=(640, 480))  # VGA resolution
            # size_im = cv2.resize(test_im, dsize=(800, 600))  # SVGA resolution
            # size_im = cv2.resize(test_im, dsize=(1028, 720))  # HD resolution
            # size_im = cv2.resize(test_im, dsize=(1920, 1080))  # Full-HD resolution
            # cv2.imshow("size_im", size_im)
            #################################################

            #################################################
            # ROI Coordinates Set-up
            # roi = size_im[320:480, 213:426]  # [380:430, 330:670]   [y:y+b, x:x+a]
            # roi_im = cv2.resize(roi, (213, 160))  # x,y
            # cv2.imshow("roi_im", roi_im)
            roi = size_im[240:480, 108:532]  # [380:430, 330:670]   [y:y+b, x:x+a]
            roi_im = cv2.resize(roi, (424, 240))  # (a of x, b of y)
            # cv2.imshow("roi_im", roi_im)
            #################################################

            #################################################
            # Gaussian Blur Filter
            Blur_im = cv2.bilateralFilter(roi_im, d=-1, sigmaColor=3, sigmaSpace=3)
            #################################################

            #################################################
            # Canny edge detector
            edges = cv2.Canny(Blur_im, 50, 100)
            # cv2.imshow("edges", edges)
            #################################################

            #################################################
            # Hough Transformation
            # lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180.0, threshold=80, minLineLength=30, maxLineGap=50)
            # rho, theta는 1씩 변경하면서 검출하겠다는 의미, np.pi/180 라디안 = 1'
            # threshold 숫자가 작으면 정밀도↓ 직선검출↑, 크면 정밀도↑ 직선검출↓
            # min_line_len 선분의 최소길이
            # max_line,gap 선분 사이의 최대 거리
            lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180.0, threshold=25, minLineLength=10, maxLineGap=20)

            N = lines.shape[0]

            '''
            print('range_N=',range(N))
            if range(N) == 0 :
                print("bad")
            elif range(N) == 0 : print("good")
            '''

            for line in range(N):
                # for line in lines:

                # x1, y1, x2, y2 = line[0]

                x1 = lines[line][0][0]
                y1 = lines[line][0][1]
                x2 = lines[line][0][2]
                y2 = lines[line][0][3]

                if x2 == x1:
                    a = 1
                else:
                    a = x2 - x1

                b = y2 - y1

                radi = b / a  # 라디안 계산
                # print('radi=', radi)

                theta_atan = math.atan(radi) * 180.0 / math.pi
                # print('theta_atan=', theta_atan)

                pt1_ri = (x1 + 108, y1 + 240)
                pt2_ri = (x2 + 108, y2 + 240)
                pt1_le = (x1 + 108, y1 + 240)
                pt2_le = (x2 + 108, y2 + 240)

                if theta_atan > 30.0 and theta_atan < 80.0:
                    # cv2.line(size_im, (x1+108, y1+240), (x2+108, y2+240), (0, 255, 0), 2)
                    # print('live_atan=', theta_atan)

                    count_posi_num_ri += 1

                    pt1_sum_ri = sumMatrix(pt1_ri, pt1_sum_ri)
                    # pt1_sum = pt1 + pt1_sum
                    # print('pt1_sum=', pt1_sum)

                    pt2_sum_ri = sumMatrix(pt2_ri, pt2_sum_ri)
                    # pt2_sum = pt2 + pt2_sum
                    # print('pt2_sum=', pt2_sum)

                if theta_atan < -30.0 and theta_atan > -80.0:
                    # cv2.line(size_im, (x1+108, y1+240), (x2+108, y2+240), (0, 0, 255), 2)
                    # print('live_atan=', theta_atan)

                    count_posi_num_le += 1

                    pt1_sum_le = sumMatrix(pt1_le, pt1_sum_le)
                    # pt1_sum = pt1 + pt1_sum
                    # print('pt1_sum=', pt1_sum)

                    pt2_sum_le = sumMatrix(pt2_le, pt2_sum_le)
                    # pt2_sum = pt2 + pt2_sum
                    # print('pt2_sum=', pt2_sum)

            # print('pt1_sum=', pt1_sum_ri)
            # print('pt2_sum=', pt2_sum_ri)
            # print('count_posi_num_ri=', count_posi_num_ri)
            # print('count_posi_num_le=', count_posi_num_le)

            # testartu = pt1_sum / np.array(count_posi_num)
            # print(tuple(testartu))

            pt1_avg_ri = pt1_sum_ri // np.array(count_posi_num_ri)
            pt2_avg_ri = pt2_sum_ri // np.array(count_posi_num_ri)
            pt1_avg_le = pt1_sum_le // np.array(count_posi_num_le)
            pt2_avg_le = pt2_sum_le // np.array(count_posi_num_le)

            # print('pt1_avg_ri=', pt1_avg_ri)
            # print('pt2_avg_ri=', pt2_avg_ri)
            # print('pt1_avg_le=', pt1_avg_le)
            # print('pt2_avg_le=', pt2_avg_le)

            # print('pt1_avg=', pt1_avg_ri)
            # print('pt2_avg=', pt2_avg_ri)
            # print('np_count_posi_num=', np.array(count_posi_num))

            # line1_ri = tuple(pt1_avg_ri)
            # line2_ri = tuple(pt2_avg_ri)
            # line1_le = tuple(pt1_avg_le)
            # line2_le = tuple(pt2_avg_le)
            # print('line1=', line1_ri)
            # print('int2=', int2)

            #################################################
            # 차석인식의 흔들림 보정
            # right-----------------------------------------------------------
            x1_avg_ri, y1_avg_ri = pt1_avg_ri
            # print('x1_avg_ri=', x1_avg_ri)
            # print('y1_avg_ri=', y1_avg_ri)
            x2_avg_ri, y2_avg_ri = pt2_avg_ri
            # print('x2_avg_ri=', x2_avg_ri)
            # print('y2_avg_ri=', y2_avg_ri)

            a_avg_ri = ((y2_avg_ri - y1_avg_ri) / (x2_avg_ri - x1_avg_ri))
            b_avg_ri = (y2_avg_ri - (a_avg_ri * x2_avg_ri))
            # print('a_avg_ri=', a_avg_ri)
            # print('b_avg_ri=', b_avg_ri)

            pt2_y2_fi_ri = 480

            # pt2_x2_fi_ri = ((pt2_y2_fi_ri - b_avg_ri) // a_avg_ri)

            if a_avg_ri > 0:
                pt2_x2_fi_ri = int((pt2_y2_fi_ri - b_avg_ri) // a_avg_ri)
            else:
                pt2_x2_fi_ri = 0

            # print('pt2_x2_fi_ri=', pt2_x2_fi_ri)
            pt2_fi_ri = (pt2_x2_fi_ri, pt2_y2_fi_ri)
            # pt2_fi_ri = (int(pt2_x2_fi_ri), pt2_y2_fi_ri)
            # print('pt2_fi_ri=', pt2_fi_ri)

            # left------------------------------------------------------------
            x1_avg_le, y1_avg_le = pt1_avg_le
            x2_avg_le, y2_avg_le = pt2_avg_le
            # print('x1_avg_le=', x1_avg_le)
            # print('y1_avg_le=', y1_avg_le)
            # print('x2_avg_le=', x2_avg_le)
            # print('y2_avg_le=', y2_avg_le)

            a_avg_le = ((y2_avg_le - y1_avg_le) / (x2_avg_le - x1_avg_le))
            b_avg_le = (y2_avg_le - (a_avg_le * x2_avg_le))
            # print('a_avg_le=', a_avg_le)
            # print('b_avg_le=', b_avg_le)

            pt1_y1_fi_le = 480
            if a_avg_le < 0:
                pt1_x1_fi_le = int((pt1_y1_fi_le - b_avg_le) // a_avg_le)
            else:
                pt1_x1_fi_le = 0
            # pt1_x1_fi_le = ((pt1_y1_fi_le - b_avg_le) // a_avg_le)
            # print('pt1_x1_fi_le=', pt1_x1_fi_le)

            pt1_fi_le = (pt1_x1_fi_le, pt1_y1_fi_le)
            # print('pt1_fi_le=', pt1_fi_le)

            # print('pt1_avg_ri=', pt1_sum_ri)
            # print('pt2_fi_ri=', pt2_fi_ri)
            # print('pt1_fi_le=', pt1_fi_le)
            # print('pt2_avg_le=', pt2_sum_le)
            #################################################

            #################################################
            # lane painting
            # right-----------------------------------------------------------
            # cv2.line(size_im, tuple(pt1_avg_ri), tuple(pt2_avg_ri), (0, 255, 0), 2) # right lane
            cv2.line(size_im, tuple(pt1_avg_ri), tuple(pt2_fi_ri), (0, 255, 0), 2)  # right lane
            # left-----------------------------------------------------------
            # cv2.line(size_im, tuple(pt1_avg_le), tuple(pt2_avg_le), (0, 255, 0), 2) # left lane
            cv2.line(size_im, tuple(pt1_fi_le), tuple(pt2_avg_le), (0, 255, 0), 2)  # left lane
            # center-----------------------------------------------------------
            cv2.line(size_im, (320, 480), (320, 360), (0, 228, 255), 1)  # middle lane
            #################################################

            #################################################
            # possible lane
            # FCP = np.array([pt1_avg_ri, pt2_avg_ri, pt1_avg_le, pt2_avg_le])
            # cv2.fillConvexPoly(size_im, FCP, color=(255, 242, 213)) # BGR
            #################################################
            FCP_img = np.zeros(shape=(480, 640, 3), dtype=np.uint8) + 0
            # FCP = np.array([pt1_avg_ri, pt2_avg_ri, pt1_avg_le, pt2_avg_le])
            # FCP = np.array([(100,100), (100,200), (200,200), (200,100)])
            FCP = np.array([pt2_avg_le, pt1_fi_le, pt2_fi_ri, pt1_avg_ri])
            cv2.fillConvexPoly(FCP_img, FCP, color=(255, 242, 213))  # BGR
            alpha = 0.9
            size_im = cv2.addWeighted(size_im, alpha, FCP_img, 1 - alpha, 0)

            # alpha = 0.4
            # size_im = cv2.addWeighted(size_im, alpha, FCP, 1 - alpha, 0)
            #################################################

            #################################################
            # lane center 및 steering 계산 (320, 360)
            lane_center_y_ri = 360
            if a_avg_ri > 0:
                lane_center_x_ri = int((lane_center_y_ri - b_avg_ri) // a_avg_ri)
            else:
                lane_center_x_ri = 0

            lane_center_y_le = 360
            if a_avg_le < 0:
                lane_center_x_le = int((lane_center_y_le - b_avg_le) // a_avg_le)
            else:
                lane_center_x_le = 0

            # caenter left lane (255, 90, 185)
            cv2.line(size_im, (lane_center_x_le, lane_center_y_le - 10), (lane_center_x_le, lane_center_y_le + 10),
                     (0, 228, 255), 1)
            # caenter right lane
            cv2.line(size_im, (lane_center_x_ri, lane_center_y_ri - 10), (lane_center_x_ri, lane_center_y_ri + 10),
                     (0, 228, 255), 1)
            # caenter middle lane
            lane_center_x = ((lane_center_x_ri - lane_center_x_le) // 2) + lane_center_x_le
            cv2.line(size_im, (lane_center_x, lane_center_y_ri - 10), (lane_center_x, lane_center_y_le + 10),
                     (0, 228, 255), 1)

            # print('lane_center_x=', lane_center_x)

            text_left = 'Turn Left'
            text_right = 'Turn Right'
            text_center = 'Center'
            text_non = ''
            org = (320, 440)
            font = cv2.FONT_HERSHEY_SIMPLEX

            if 0 < lane_center_x <= 318:
                cv2.putText(size_im, text_left, org, font, 0.7, (0, 0, 255), 2)
            elif 318 < lane_center_x < 322:
                # elif lane_center_x > 318 and lane_center_x < 322 :
                cv2.putText(size_im, text_center, org, font, 0.7, (0, 0, 255), 2)
            elif lane_center_x >= 322:
                cv2.putText(size_im, text_right, org, font, 0.7, (0, 0, 255), 2)
            elif lane_center_x == 0:
                cv2.putText(size_im, text_non, org, font, 0.7, (0, 0, 255), 2)
            #################################################

            global test_con
            test_con = 1
            # print('test_con=', test_con)

            # 변수 초기화
            count_posi_num_ri = 0

            pt1_sum_ri = (0, 0)
            pt2_sum_ri = (0, 0)
            pt1_avg_ri = (0, 0)
            pt2_avg_ri = (0, 0)

            count_posi_num_le = 0

            pt1_sum_le = (0, 0)
            pt2_sum_le = (0, 0)
            pt1_avg_le = (0, 0)
            pt2_avg_le = (0, 0)

            cv2.imshow('frame_size_im', size_im)
            cv2.waitKey(1)
            # cv2.imshow("test_im", test_im) # original size image
            # cv2.waitKey(1)

        if self.recording:
            image.save_to_disk('_out/%08d' % image.frame_number)