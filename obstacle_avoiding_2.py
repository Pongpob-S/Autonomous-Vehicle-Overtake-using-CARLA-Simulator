

import pygame
import glob
import os
import sys
import numpy as np
import cv2
import queue
import argparse
import collections
import datetime
import logging
import math
import random
import re
import weakref
import csv

try:
	import pygame
	from pygame.locals import KMOD_CTRL
	from pygame.locals import KMOD_SHIFT
	from pygame.locals import K_0
	from pygame.locals import K_9
	from pygame.locals import K_BACKQUOTE
	from pygame.locals import K_BACKSPACE
	from pygame.locals import K_COMMA
	from pygame.locals import K_DOWN
	from pygame.locals import K_ESCAPE
	from pygame.locals import K_F1
	from pygame.locals import K_LEFT
	from pygame.locals import K_PERIOD
	from pygame.locals import K_RIGHT
	from pygame.locals import K_SLASH
	from pygame.locals import K_SPACE
	from pygame.locals import K_TAB
	from pygame.locals import K_UP
	from pygame.locals import K_a
	from pygame.locals import K_b
	from pygame.locals import K_c
	from pygame.locals import K_d
	from pygame.locals import K_g
	from pygame.locals import K_h
	from pygame.locals import K_i
	from pygame.locals import K_l
	from pygame.locals import K_m
	from pygame.locals import K_n
	from pygame.locals import K_p
	from pygame.locals import K_q
	from pygame.locals import K_r
	from pygame.locals import K_s
	from pygame.locals import K_v
	from pygame.locals import K_w
	from pygame.locals import K_x
	from pygame.locals import K_z
	from pygame.locals import K_MINUS
	from pygame.locals import K_EQUALS
except ImportError:
	raise RuntimeError('cannot import pygame, make sure pygame package is installed')


from image_functions import * # lane detection image processing functions
from laneDetect import * # lane detection algorithm

# init pygame
pygame.init()


""" PROGRAM BEGINS """
try:
	sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
		sys.version_info.major,
		sys.version_info.minor,
		'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
	pass

import carla

import random
import time
import threading

""" CONSTANTS """
IM_WIDTH    = 1280
IM_HEIGHT   = 720
GLOBAL_FONT = cv2.FONT_HERSHEY_SIMPLEX

# car control object
car_control = carla.VehicleControl()

# list to store actor in the simulation
actor_list = []

# create pygame screen
screen_size = width, height = (IM_WIDTH, IM_HEIGHT)
screen = pygame.display.set_mode(screen_size)
pygame.display.set_caption("FRONT CAMERA")

VID_NAME = 'Scenario.avi'
WRITE_VID = True
if WRITE_VID:
	out = cv2.VideoWriter(VID_NAME,cv2.VideoWriter_fourcc('M','J','P','G'), 10, (IM_WIDTH, IM_HEIGHT))

""" FNS """
def get_image(image):
	i = np.array(image.raw_data)
	i2 = i.reshape((IM_HEIGHT, IM_WIDTH, 4))
	i3 = i2[:, :, :3]

	return i3

def get_carla_status(vehicle_actor, image):
	#get carla status
	#original spawn point
	spawnpt = [-347.8, 33.6, 1]

	vehicle_tf = vehicle_actor.get_transform()
	vehicle_loc = [round(vehicle_tf.location.x - spawnpt[0], 2),
					round(vehicle_tf.location.y - spawnpt[1], 2),
					round(vehicle_tf.location.z, 2)]

	#display
	custom_org = (35, 50)
	custom_fontscale = 0.7
	custom_color = (255, 150, 120) # blue
	custom_thickness = 2

	image = cv2.UMat(image)
	res = cv2.putText(image, 
						f"location:(x={vehicle_loc[0]},y={vehicle_loc[1]},z={vehicle_loc[2]})",
						custom_org,
						GLOBAL_FONT,
						custom_fontscale,
						custom_color,
						custom_thickness)
	return res

def carlaToFile(actor, filename):
	ego_velocity      = actor.get_velocity()
	ego_velocity      = np.array([ego_velocity.x, ego_velocity.y, ego_velocity.z])
	absolute_velocity = round(np.linalg.norm(ego_velocity, ord=2), 3)
	absolute_front_velocity = relative_velocity + absolute_velocity

	vehicle_path = actor.get_transform()

	veh_x = round(vehicle_path.location.x, 2)
	veh_y = round(vehicle_path.location.y, 2)
	veh_z = round(vehicle_path.location.z, 2)
				
	fieldnames = ['x', 'y', 'z']
	rows = [{'x': veh_x,'y': veh_y, 'z': veh_z}]

	with open(filename, 'a', encoding='UTF8', newline ='') as fd:
		csv_writer = csv.DictWriter(fd, fieldnames = fieldnames)
		#csv_writer.writeheader()
		csv_writer.writerows(rows) 


	print(rows)

relative_velocity = 0

def get_driving_status(vehicle_actor, image):
	# get driving status

	global absolute_velocity

	vehicle_tf    = vehicle_actor.get_transform()
	vehicle_angle = round(vehicle_tf.rotation.yaw, 3) # degree
	drive_mode    = car_control.reverse
	hand_brake    = car_control.hand_brake

	acceleration  = vehicle_actor.get_acceleration() # m/s^2
	acceleration  = np.array([acceleration.x, acceleration.y, acceleration.z])
	absolute_acceleration = round(np.linalg.norm(acceleration, ord=2), 3)

	velocity      = vehicle_actor.get_velocity()
	velocity      = np.array([velocity.x, velocity.y, velocity.z])
	absolute_velocity = round(np.linalg.norm(velocity, ord=2), 3)
	absolute_front_velocity = relative_velocity + absolute_velocity

	# display
	custom_org = (1000, 600)
	custom_fontscale = 0.8
	custom_color = (200, 50, 50) # green
	custom_thickness = 2

	# drive mode string
	drive_mode_str = "REV" if drive_mode else "FWD"
	image = cv2.putText(image, f"drive mode: {drive_mode_str}", (900, 570), GLOBAL_FONT, custom_fontscale, custom_color, custom_thickness)

	# brake
	hand_brake_str = "YES" if hand_brake else "NO"
	image = cv2.putText(image, f"brake: {hand_brake_str}", (900, 600), GLOBAL_FONT, custom_fontscale, custom_color, custom_thickness)

	# acceleration
	image = cv2.putText(image, f"acceleration: {absolute_acceleration} m/s^2", (900, 630), GLOBAL_FONT, custom_fontscale, custom_color, custom_thickness)

	# velocity
	image = cv2.putText(image, f"velocity: {absolute_velocity} m/s", (900, 660), GLOBAL_FONT, custom_fontscale, custom_color, custom_thickness)

	image = cv2.putText(image, f"front car velocity: {absolute_front_velocity} m/s", (900, 540), GLOBAL_FONT, custom_fontscale, custom_color, custom_thickness)

	image = cv2.putText(image, f"lane positioning: {current_lane}", (900, 510), GLOBAL_FONT, custom_fontscale, custom_color, custom_thickness) 
	# angle
	res = cv2.putText(image, f"steering angle: {vehicle_angle} deg", (900, 690), GLOBAL_FONT, custom_fontscale, custom_color, custom_thickness)

	return res


def show_image(image):
	cv2.imshow("", image)
	cv2.waitKey(1)

### OBSTACLE DETECTOR FNS ###
obstacle_detector_spawnpoints = {
	"front": carla.Transform(carla.Location(x=2.5, z=1.3)),
	"back": carla.Transform(carla.Location(x=-2.5, z=1.3), carla.Rotation(yaw=180)),
	"left": carla.Transform(carla.Location(x= 2.0, y=-1, z=1.3), carla.Rotation(yaw=-90)),
	"right": carla.Transform(carla.Location(x= 2.0, y=1, z=1.3), carla.Rotation(yaw=90)),
	"frontleft" : carla.Transform(carla.Location(x= 2.0, y=-1, z=1.3), carla.Rotation(yaw=-15)),
	"frontright" : carla.Transform(carla.Location(x= 2.0, y=1, z=1.3), carla.Rotation(yaw=15))
}

# variables used for storing sensor instances
front_detect = ""
back_detect = ""
left_detect = ""
right_detect = ""
frontleft_detect = ""
frontright_detect = ""

# used for notifying
detector_notify = {
"front": 0,
"back": 0,
"left": 0,
"right": 0,
"frontleft": 0,
"frontright": 0
}


def spawn_one_obstacle_detector(blueprint, spawnpoint, side):
	# set detection range
	detectRange = "1"
	if side == "front":
		detectRange = "10"
	elif side == "frontleft":
		detectRange = "10"
	elif side == "frontright":
		detectRange = "10"
	elif side == "back":
		detectRange = "3"
	else:
		detectRange = "3"
	blueprint.set_attribute("distance", detectRange) 

	blueprint.set_attribute("sensor_tick", "0.2")
	blueprint.set_attribute("debug_linetrace", "False")
	blueprint.set_attribute("hit_radius", "0.5")
		
	sensor = world.try_spawn_actor(obstacle_detector_bp, spawnpoint, attach_to=vehicle)
	actor_list.append(sensor)
	
	return sensor

def spawn_all_obstacle_detector():
	front_detect = spawn_one_obstacle_detector(obstacle_detector_bp, obstacle_detector_spawnpoints["front"], "front")
	back_detect = spawn_one_obstacle_detector(obstacle_detector_bp, obstacle_detector_spawnpoints["back"], "back")
	left_detect = spawn_one_obstacle_detector(obstacle_detector_bp, obstacle_detector_spawnpoints["left"], "left")
	right_detect = spawn_one_obstacle_detector(obstacle_detector_bp, obstacle_detector_spawnpoints["right"], "right")
	frontleft_detect = spawn_one_obstacle_detector(obstacle_detector_bp, obstacle_detector_spawnpoints["frontleft"], "frontleft")
	frontright_detect = spawn_one_obstacle_detector(obstacle_detector_bp, obstacle_detector_spawnpoints["frontright"], "frontright")

	front_detect.listen(lambda detect: parseDetect(detect, "front"))
	back_detect.listen(lambda detect: parseDetect(detect, "back"))
	left_detect.listen(lambda detect: parseDetect(detect, "left"))
	right_detect.listen(lambda detect: parseDetect(detect, "right"))
	frontleft_detect.listen(lambda detect: parseDetect(detect, "frontleft"))
	frontright_detect.listen(lambda detect: parseDetect(detect, "frontright"))

def parseDetect(detect, side):
	# print(f"{side} detect: {detect.other_actor} at: {detect.distance} m")
	detector_notify[side] = round(detect.distance, 2)

def drawDetectorStatus(image):
	# obstacle notification notify
	custom_org = (200, 200)
	custom_fontscale = 2
	custom_color = (0, 150, 250) # orange
	custom_thickness = 2

	image = cv2.UMat(image)

	# draw side detect
	# front
	custom_fontscale = 2
	if detector_notify["front"] != 0 or detector_notify["back"] != 0 or detector_notify["left"] != 0 or detector_notify["right"] != 0:
		
		image = cv2.putText(image, f"WARNING", (50, 450), GLOBAL_FONT, 2, custom_color, custom_thickness)
		image = cv2.putText(image, f"obstacle nearby", (50, 480), GLOBAL_FONT, 1, custom_color, custom_thickness)
	
	custom_fontscale = 0.8
	image = cv2.putText(image, f"Obstacle detector", (50, 530), GLOBAL_FONT, custom_fontscale, custom_color, custom_thickness)
	image = cv2.putText(image, f"front: {detector_notify['front']} m", (50, 560), GLOBAL_FONT, custom_fontscale, custom_color, custom_thickness)
	image = cv2.putText(image, f"back: {detector_notify['back']} m", (50, 590), GLOBAL_FONT, custom_fontscale, custom_color, custom_thickness)
	image = cv2.putText(image, f"left: {detector_notify['left']} m", (50, 620), GLOBAL_FONT, custom_fontscale, custom_color, custom_thickness)
	image = cv2.putText(image, f"right: {detector_notify['right']} m", (50, 650), GLOBAL_FONT, custom_fontscale, custom_color, custom_thickness)
	image = cv2.putText(image, f"frontleft: {detector_notify['frontleft']} m", (50, 680), GLOBAL_FONT, custom_fontscale, custom_color, custom_thickness)
	image = cv2.putText(image, f"frontright: {detector_notify['frontright']} m", (50, 710), GLOBAL_FONT, custom_fontscale, custom_color, custom_thickness)

	detector_notify["front"] = 0
	detector_notify["back"] = 0
	detector_notify["left"] = 0
	detector_notify["right"] = 0
	detector_notify["frontleft"] = 0
	detector_notify["frontright"] = 0

	return image

def spawnCar(car_name, lx, ly, lz, rx, ry, rz):

	bp = blueprint_library.filter(car_name)[0]
	print(bp)
	print(car_name + " spawned!")
	location = carla.Location(lx, ly, lz)
	rotation = carla.Rotation(rx, ry, rz)
	spawn_point = carla.Transform(location, rotation)
	car_name = world.spawn_actor(bp, spawn_point)
	actor_list.append(car_name)

	return car_name

def spawnRandomCar():
	
	bp = random.choice(blueprint_library.filter('vehicle.*.*'))
	spawn_point = random.choice(world.get_map().get_spawn_points())
	npc = world.spawn_actor(bp, spawn_point)
	npc.set_autopilot(True)
	actor_list.append(npc)
	


radar_sensor = None


def toggle_radar():
		global radar_sensor

		if radar_sensor is None:
		   radar_sensor = RadarSensor(vehicle)
		elif radar_sensor.sensor is not None:
			radar_sensor.sensor.destroy()
			radar_sensor = None 

def destroy():
		global radar_sensor
		if radar_sensor is not None:
		   toggle_radar()

class RadarSensor(object):
	def __init__(self, parent_actor):

		global sensorbackright, sensorbackleft

		self.sensor = None
		self.sensorbackright = None
		self.sensorbackleft = None
		self._parent = parent_actor
		self.velocity_range = 7.5 # m/s
		#world = self._parent.get_world()
		self.debug = world.debug
		bp = world.get_blueprint_library().find('sensor.other.radar')
		bp.set_attribute('horizontal_fov', str(20))
		bp.set_attribute('vertical_fov', str(5))
		bp.set_attribute('range', str(10))
		self.sensor = world.spawn_actor(bp, carla.Transform(carla.Location(x=2.3, z=1.0), carla.Rotation(pitch=5)), attach_to=self._parent)
		self.sensorbackright = world.spawn_actor(bp, carla.Transform(carla.Location(x=2.3, y=1.0, z=1.0), carla.Rotation(yaw=165,)), attach_to=self._parent)
		self.sensorbackleft = world.spawn_actor(bp, carla.Transform(carla.Location(x=2.3, y=-1.0, z=1.0), carla.Rotation(yaw=-165)), attach_to=self._parent) 
		# We need a weak reference to self to avoid circular reference.
		weak_self = weakref.ref(self)
		self.sensor.listen(lambda radar_data: RadarSensor._Radar_callback(weak_self, radar_data))
		self.sensorbackright.listen(lambda radar_dataright: RadarSensor._Radar_callback(weak_self, radar_dataright))
		self.sensorbackleft.listen(lambda radar_dataleft: RadarSensor._Radar_callback(weak_self, radar_dataleft))




	@staticmethod
	def _Radar_callback(weak_self, radar_data):
		self = weak_self()
		if not self:
			return
		# To get a numpy [[vel, altitude, azimuth, depth],...[,,,]]:
		points = np.frombuffer(radar_data.raw_data, dtype=np.dtype('f4'))
		points = np.reshape(points, (len(radar_data), 4))

		pointsright = np.frombuffer(radar_data.raw_data, dtype=np.dtype('f4'))
		pointsright = np.reshape(pointsright, (len(radar_data), 4))

		pointsleft = np.frombuffer(radar_data.raw_data, dtype=np.dtype('f4'))
		pointsleft = np.reshape(pointsleft, (len(radar_data), 4))

		global detectright, detectleft, detect


		current_rot = radar_data.transform.rotation
		for detect in radar_data:
			azi = math.degrees(detect.azimuth)
			alt = math.degrees(detect.altitude)
			# The 0.25 adjusts a bit the distance so the dots can
			# be properly seen
			fw_vec = carla.Vector3D(x=detect.depth - 0.25)
			carla.Transform(
				carla.Location(),
				carla.Rotation(
					pitch=current_rot.pitch + alt,
					yaw=current_rot.yaw + azi,
					roll=current_rot.roll)).transform(fw_vec)

		current_rotright = radar_data.transform.rotation
		for detectright in radar_data:
			azi_right = math.degrees(detectright.azimuth)
			alt_right = math.degrees(detectright.altitude)
			# The 0.25 adjusts a bit the distance so the dots can
			# be properly seen
			fw_vec_right = carla.Vector3D(x=detectright.depth - 0.25)
			carla.Transform(
				carla.Location(),
				carla.Rotation(
					pitch=current_rotright.pitch + alt_right,
					yaw=current_rotright.yaw + azi_right,
					roll=current_rotright.roll)).transform(fw_vec_right)

		current_rotleft = radar_data.transform.rotation
		for detectleft in radar_data:
			azi_left = math.degrees(detectleft.azimuth)
			alt_left = math.degrees(detectleft.altitude)
			# The 0.25 adjusts a bit the distance so the dots can
			# be properly seen
			fw_vec_left = carla.Vector3D(x=detectleft.depth - 0.25)
			carla.Transform(
				carla.Location(),
				carla.Rotation(
					pitch=current_rotleft.pitch + alt_left,
					yaw=current_rotleft.yaw + azi_left,
					roll=current_rotleft.roll)).transform(fw_vec_left)

			def clamp(min_v, max_v, value):
				return max(min_v, min(value, max_v))

			norm_velocity = detect.velocity / self.velocity_range # range [-1, 1]
			r = int(clamp(0.0, 1.0, 1.0 - norm_velocity) * 255.0)
			g = int(clamp(0.0, 1.0, 1.0 - abs(norm_velocity)) * 255.0)
			b = int(abs(clamp(- 1.0, 0.0, - 1.0 - norm_velocity)) * 255.0)
			self.debug.draw_point(
				radar_data.transform.location + fw_vec,
				size=0.075,
				life_time=0.06,
				persistent_lines=False,
				color=carla.Color(r, g, b))

		global relative_velocity 
		relative_velocity = detect.velocity

		return relative_velocity



""" PROGRAM BEGINS """
try:
	client = carla.Client('localhost', 2000)
	client.set_timeout(10.0)
	print(f"Connected to server.")
	
	# get world
	world = client.load_world('Town04_Opt', carla.MapLayer.Buildings | carla.MapLayer.ParkedVehicles)
	world.unload_map_layer(carla.MapLayer.Buildings)

	# speed limit autopilot
	traffic_manager = client.get_trafficmanager(8000)

	
	# set weather

	weather = carla.WeatherParameters(
		cloudiness=0.0,
		precipitation=0.0,
		sun_altitude_angle=0.0)
	world.set_weather(weather)

	#world.set_weather(carla.WeatherParameters.ClearSunset)
	print(f"Weather set.")
	

	# change the view of the camera to the place
	spectator = world.get_spectator()
	spectator.set_transform(carla.Transform(carla.Location(-355, 33.6, 1)))

	# destroy all vehices and sensors to prepare the environment
	for x in list(world.get_actors()):
		if 'vehicle' in x.type_id or 'sensor' in x.type_id:
			x.destroy()

	# spawn a car
	blueprint_library = world.get_blueprint_library()
	bp = blueprint_library.filter('model3')[0]
	print(bp)
	print("Vehicle Spawned!")
	spawn_point = carla.Transform(carla.Location(-347.8, 33.6, 1),carla.Rotation(0,0,0)) #33.6 #30.1
	vehicle = world.spawn_actor(bp, spawn_point)
	actor_list.append(vehicle)


	#radar_spawn_point = carla.Transform(carla.Location(x=2.5, z=0.5))
	#radar = world.try_spawn_actor(radar_bp, radar_spawn_point, attach_to=vehicle)
	#detect = carla.RadarDetection()

	#radar_spawn_backright = carla.Transform(carla.Location(x=-2.5, y=1 , z=0.5), carla.Rotation(yaw=135))
	#radarbackright = world.try_spawn_actor(radar_bp, radar_spawn_backright, attach_to=vehicle)
	#detectbackright = carla.RadarDetection()
#
	#radar_spawn_backleft = carla.Transform(carla.Location(x=-2.5, y=-1, z=0.5), carla.Rotation(yaw=-135))
	#radarbackleft = world.try_spawn_actor(radar_bp, radar_spawn_backleft, attach_to=vehicle)
	#detectbackleft = carla.RadarDetection()


	## scenario 1 / obstacle (front, left, right)

	# Spawn an obstacle
	# scenario 1 / carla.Location(60,205.6,1)
	# scenario 2 / carla.Location(60,205.6,1)
	bp1 = blueprint_library.filter('cybertruck')[0]
	print(bp1)
	print("obstacle Spawned!")
	spawn_point1 = carla.Transform(carla.Location(-331.1, 33.5, 1),carla.Rotation(0,0,0))
	obstacle = world.spawn_actor(bp1, spawn_point1)
	actor_list.append(obstacle)
	#obstacle.set_autopilot(True)
	
	# Spawn an obstacle2 right
	# scenario 1 / carla.Location(45,208.5,1)
	# scenario 2 / carla.Location(60,209.1,1) 
	bp2 = blueprint_library.filter('cybertruck')[0]
	print(bp2)
	print("obstacle2 Spawned!")
	spawn_point2 = carla.Transform(carla.Location(-331.1, 37, 1),carla.Rotation(0,0,0))
	obstacle2 = world.spawn_actor(bp2, spawn_point2)
	actor_list.append(obstacle2)
	#obstacle2.set_autopilot(True)

	# Spawn an obstacle3 left
	# scenario 1 / carla.Location(45,202,1)
	# scenario 2 / carla.Location(60,202.1,1)
	bp3 = blueprint_library.filter('cybertruck')[0]
	print(bp3)
	print("obstacle3 Spawned!")
	spawn_point3 = carla.Transform(carla.Location(-331.1, 30, 2),carla.Rotation(0,0,0))
	obstacle3 = world.spawn_actor(bp3, spawn_point3)
	actor_list.append(obstacle3)
	#obstacle3.set_autopilot(True)

	#Spawn an obstacle4 backright
	#bp4 = blueprint_library.filter('cybertruck')[0]
	#print(bp4)
	#spawn_point4 = carla.Transform(carla.Location(-277, 37, 2), carla.Rotation(0,0,0))
	#obstacle4 = world.spawn_actor(bp4, spawn_point4)
	#actor_list.append(obstacle4)
	#obstacle4.apply_control(carla.VehicleControl(brake = 1.0))
#
	#bp5 = blueprint_library.filter('cybertruck')[0]
	#print(bp5)
	#spawn_point5 = carla.Transform(carla.Location(-250.0, 30, 3), carla.Rotation(0,0,0))
	#obstacle5 = world.spawn_actor(bp5, spawn_point5)
	#actor_list.append(obstacle5)
	#obstacle5.apply_control(carla.VehicleControl(brake = 1.0))
##
	#bp6 = blueprint_library.filter('cybertruck')[0]
	#print(bp6)
	#spawn_point6 = carla.Transform(carla.Location(-250, 33.5, 3), carla.Rotation(0,0,0))
	#obstacle6 = world.spawn_actor(bp6, spawn_point6)
	#actor_list.append(obstacle6)
	#obstacle6.apply_control(carla.VehicleControl(brake = 1.0))
##
	#bp7 = blueprint_library.filter('cybertruck')[0]
	#print(bp7)
	#spawn_point7 = carla.Transform(carla.Location(-250, 27, 4), carla.Rotation(0,0,0))
	#obstacle7 = world.spawn_actor(bp7, spawn_point7)
	#actor_list.append(obstacle7)
	#obstacle7.apply_control(carla.VehicleControl(brake = 1.0))
	'''
	# Spawm an obstacle3
	bp3 = blueprint_library.filter('mini')[0]
	print(bp3)
	print("obstacle3 Spawned!")
	spawn_point3 = carla.Transform(carla.Location(80,205.6,1),carla.Rotation(0,0,0))
	obstacle3 = world.spawn_actor(bp3, spawn_point3)
	actor_list.append(obstacle3)
	'''

	# get rgb camera blueprint and set it's attributes
	camera_bp = blueprint_library.find("sensor.camera.rgb")

	# spawn camera
	camera_bp.set_attribute("image_size_x", f"{IM_WIDTH}")
	camera_bp.set_attribute("image_size_y", f"{IM_HEIGHT}")
	camera_bp.set_attribute("fov", "80")
	camera_bp.set_attribute("lens_circle_falloff", "5.0")
	# Set the time in seconds between sensor captures
	camera_bp.set_attribute("sensor_tick", "0.2")
	# adjust sensor relative to vehicle
	camera_spawn_point = carla.Transform(carla.Location(x=2.5, z=1.5), carla.Rotation(pitch=-2.0))
	camera_spawn_point_3rd = carla.Transform(carla.Location(x=-7.0, z=2.5), carla.Rotation(pitch=-2.0))
	camera_spawn_point_left = carla.Transform(carla.Location(x=2.5, y=-1.5, z=0.5), carla.Rotation(pitch=-2.0))
	camera_spawn_point_right = carla.Transform(carla.Location(x=2.5, y=1.5, z=0.5), carla.Rotation(pitch=-2.0))
	# spawn the sensor and attach to the vehicle
	sensor = world.try_spawn_actor(camera_bp, camera_spawn_point, attach_to=vehicle)
	sensor_3rd = world.try_spawn_actor(camera_bp, camera_spawn_point_3rd, attach_to=vehicle)
	sensor_left = world.try_spawn_actor(camera_bp, camera_spawn_point_left, attach_to=vehicle)
	sensor_right = world.try_spawn_actor(camera_bp, camera_spawn_point_right, attach_to=vehicle)

	assert sensor is not None, "Camera cannot be spawned!"
	print(f"Camera spawned.")
	actor_list.append(sensor)
	actor_list.append(sensor_3rd)

	# add image to queue when captured
	image_queue = queue.Queue()
	sensor.listen(image_queue.put)

	image_queue_3rd = queue.Queue()
	sensor_3rd.listen(image_queue_3rd.put)

	image_queue_left = queue.Queue()
	sensor_left.listen(image_queue_left.put)

	image_queue_right = queue.Queue()
	sensor_right.listen(image_queue_right.put)

	### OBSTACLE DETECTOR SPAWNING ###
	obstacle_detector_bp = blueprint_library.find("sensor.other.obstacle")
	spawn_all_obstacle_detector()


	GameDone = False
	use_lane_detect = True

	# auto_steer used for lane toggling 
	# True for car already in left lane
	# False for car already in right lane
	current_lane = "" 
	steer_timer = 0
	start_steer_timer = False

	# begin pid timer
	pid_timer = time.time()

	#15 indexes of array storing pixel from left and right camera
	LaneMarkPixel_Right = [None] * 15
	LaneMarkPixel_Left = [None] * 15

	
	"""GAME START HERE"""
	while not GameDone:

		obstacle.set_autopilot(True)
		obstacle2.set_autopilot(True)
		obstacle3.set_autopilot(True)

		vehicle.apply_control(carla.VehicleControl(brake = 1.0))

		carlaToFile(obstacle, 'mid_path.csv')
		carlaToFile(obstacle2, 'right_path.csv')
		carlaToFile(obstacle3, 'left_path.csv')



		world.tick() #signal the server to update

		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				GameDone = True

		# get image
		img = image_queue.get()
		img = get_image(img)

		img_3rd = image_queue_3rd.get()
		img_3rd = get_image(img_3rd)

		img_left = image_queue_left.get()
		img_left = get_image(img_left)

		img_right = image_queue_right.get()
		img_right = get_image(img_right)	

		#monitor left lane
		left_frame = img_left.copy()
		left_warped, M_left, Minv_left = perspectiveTransform(left_frame)
		bin_warped_left = process_threshold(left_warped)

		#monitor right lane
		right_frame = img_right.copy()
		right_warped, M_right, Minv_right = perspectiveTransform(right_frame)
		bin_warped_right = process_threshold(right_warped)

		#catch pixel from process_threshold (black and white) for left cam
		nonzero_left = bin_warped_left.nonzero()
		nonzerox_left = np.argmax(nonzero_left[1])

		#catch pixel from process_threshold (black and white) for right cam
		nonzero_right = bin_warped_right.nonzero()
		nonzerox_right = np.argmax(nonzero_right[1])

		if use_lane_detect:

			#insert value of new pixel into the left index of the array
			LaneMarkPixel_Right.insert(0, nonzerox_right)
			#delete the right value of the array
			LaneMarkPixel_Right = LaneMarkPixel_Right[:-1]

			#insert value of new pixel into the left index of the array
			LaneMarkPixel_Left.insert(0, nonzerox_left)
			#delete the right value of the array
			LaneMarkPixel_Left = LaneMarkPixel_Left[:-1]

			#declare lane type
			RightLaneType = ""
			LeftLaneType = ""

			#read player velocity prevent the car from stopping on the lane for a long time
			player_velocity      = vehicle.get_velocity()
			player_velocity      = np.array([player_velocity.x, player_velocity.y, player_velocity.z])
			player_absolute_velocity = round(np.linalg.norm(player_velocity, ord=2), 3)
			#print(player_absolute_velocity)


			#check lane detection
			if 1 in LaneMarkPixel_Right:
				RightLaneType = "Broken"
			elif player_absolute_velocity < 0.1:
				#use_lane_detect = False
				#car_control.brake = 1.0
				RightLaneType = "Broken"
			else:
				RightLaneType = "Solid"
	
			if 1 in LaneMarkPixel_Left:
				LeftLaneType = "Broken"
			elif player_absolute_velocity < 0.1:
				#use_lane_detect = False
				#car_control.brake = 1.0
				LeftLaneType = "Broken"
			else:
				LeftLaneType = "Solid"
	
			#declare current lane
			if RightLaneType == "Broken":
				if LeftLaneType == "Broken":
					current_lane = "changable"
				else:
					current_lane = "left"
			else:
				current_lane = "right"
	
			print(current_lane)

		#print(LaneMarkPixel_Left)
		#print(LaneMarkPixel_Right)
		#print("Right Lane is " + RightLaneType)
		#print("Left Lane is " + LeftLaneType)

		cv2.imshow("right_lane", bin_warped_right)

		# toggle lane detection
		if use_lane_detect:
			# lane detection
			frame = img.copy()
			frame = mask_lane_center(frame)
			warped, M, Minv = perspectiveTransform(frame)
			bin_warped      = process_threshold(warped)
			left_fit, right_fit, slidingWin = laneWindowSearch(bin_warped)
			unwarped = cv2.warpPerspective(slidingWin, Minv, (IM_WIDTH, IM_HEIGHT))
			dist, side = getLaneOffset(unwarped, left_fit, right_fit)
			img = cv2.addWeighted(img, 1, unwarped, 1, 0)

			frame_3rd = img_3rd.copy()
			frame_3rd = mask_lane_center(frame_3rd)
			warped_3rd, M_3rd, Minv_3rd = perspectiveTransform(frame_3rd)
			bin_warped_3rd      = process_threshold(warped_3rd)
			
			#  เปิดหน้าต่างแสดงผล 3rd person
			cv2.imshow("3rd Person View", img_3rd)

		#cv2.imshow("bin_warped", bin_warped)

		### AUTO PILOT BEGINS ###
		Kp = 0.1; Kd = 0.01; Ki = 0.02
		pid_elapsed = time.time() - pid_timer
		pid_timer   = time.time()

		try:
			err = dist

			# anti windup add saturation
			if err > 1:
				err = 0.8

			if side == "left":
				err *= -1

			diff_err = err/pid_elapsed
			inte_err = err*pid_elapsed

			steering = (Kp*err) - (Kd*diff_err) + (Ki*err) 

			car_control.steer = steering

			#front_car_speed = obstacle5.get_velocity()
			#left_car_speed = obstacle3.get_velocity()
			#right_car_speed = obstacle2.get_velocity()

			distance_right = 15


			if detector_notify["front"] != 0:

				toggle_radar()

				#print(detectright.depth)

				print(detectright.velocity + absolute_velocity)
				print(detectleft.velocity + absolute_velocity)



				if current_lane == "changable":
					if detector_notify["frontright"] != 0 or detector_notify["right"] != 0:
						if detector_notify["frontleft"] != 0 or detector_notify["left"] !=0:
							vehicle.enable_constant_velocity(front_car_speed)
							#print("1. vehicle follows front car")
							#car_control.brake = 1.0
							#current_lane = "center"
						else:
							use_lane_detect = False
							car_control.steer = -0.3
							car_control.brake = 0.0
							vehicle.disable_constant_velocity()
							#print("1.1 vehicle shifting left")
							#current_lane = "left"
					else:
						use_lane_detect = False
						car_control.steer = 0.3
						car_control.brake = 0.0
						vehicle.disable_constant_velocity()
						#print("1.2 vehicle shifting right")
						#current_lane = "right"
				
				if current_lane == "left":
					if detector_notify["frontright"] != 0 or detector_notify["right"] != 0:
						vehicle.enable_constant_velocity(left_car_speed)
						use_lane_detect = True
						#print("2. vehicle follows front car")
						#current_lane = "left"
					else:
						use_lane_detect = False
						car_control.steer = 0.3
						car_control.brake = 0.0
						vehicle.disable_constant_velocity()
						#print("2.1 vehicle shifting right")
						#current_lane = "center"

				elif current_lane == "right":
					if detector_notify["frontleft"] != 0 or detector_notify["left"] != 0: 
						vehicle.enable_constant_velocity(right_car_speed)
						#print("3. vehicle follows front car")
						#current_lane = "right"
					else:
						use_lane_detect = False
						car_control.steer = -0.3
						car_control.brake = 0.0
						vehicle.disable_constant_velocity()
						#print("3.1 vehicle shifting left")
						#current_lane = "center"


				start_steer_timer = True
				steer_timer = time.time()

			else :
				vehicle.disable_constant_velocity()
				use_lane_detect = True
				car_control.brake = 0.0

		except:
			car_control.steer = 0
			#print("something wrong with err")



		# throtling control
		car_control.throttle = 0.2
		vehicle.apply_control(car_control)

		#car_control.throttle = 0.1
		#obstacle.apply_control(car_control)
		#obstacle.set_autopilot(True)

		'''
		# integrate object detection control
		if detector_notify["front"] != 0:
			use_lane_detect = False

			if current_lane == "left":
				car_control.steer = 0.3
			elif current_lane == "right":
				car_control.steer = -0.3

			start_steer_timer = True
			steer_timer = time.time()
		'''

		if start_steer_timer:
			if time.time() - steer_timer > 0.7:
				start_steer_timer = False # stop timer
				steer_timer = 0 # reset timer

				#if current_lane == "center":
				#	if detector_notify["right"] != 0 or detector_notify["frontright"] != 0:
				#		current_lane = "left"
				#	elif detector_notify["left"] !=0 or detector_notify["frontleft"] !=0:
				#		current_lane = "right"
				#elif current_lane == "right":
				#	current_lane = "center"
				#elif current_lane == "left":
				#	current_lane = "center"
			#print(current_lane)

		img = get_carla_status(vehicle, img)
		img = get_driving_status(vehicle, img)

		# draw obstacle detector status
		img = drawDetectorStatus(img)

		# use pygame to show image
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # BGR -> RGB colorspace
		out.write(img)
		img = cv2.flip(img, 1, img) # flip image
		img = img.get() # get np array
		pygame_img = np.rot90(img) # rotate array by 90 deg
		frame = pygame.surfarray.make_surface(pygame_img) # make surface
		screen.blit(frame, (0, 0)) # blit
		pygame.display.flip() # update screen
		
		# scenario 1 / obstacle (front, left, right)
		keys = pygame.key.get_pressed()
		#if keys[K_a]:
		#	print("left vehicle is moving")
		#	obstacle3.set_autopilot(True)
		#	traffic_manager.vehicle_percentage_speed_difference(obstacle3, 90)
		#elif keys[K_w]:
		#	print("front vehicle is moving")
		#	obstacle.set_autopilot(True)
		#	traffic_manager.vehicle_percentage_speed_difference(obstacle, 80)
		#elif keys[K_d]:
		#	print("right vehicle is moving")
		#	obstacle2.set_autopilot(True) 
		#	traffic_manager.vehicle_percentage_speed_difference(obstacle2, 70)
		#elif keys[K_g]:
		#	toggle_radar()

		



finally:
	print(f"release video writer...")
	out.release()


	print('destroying actors')
	for actor in actor_list:
		actor.destroy()
	print('Program Done.')


	


if __name__ == '__main__':

	try:
		main()
	except KeyboardInterrupt:
		print('\nCancelled by user. Bye!')