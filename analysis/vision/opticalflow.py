import cv2 
import numpy as np
import argparse

import pandas as pd
import datetime

import thread
import json

from multiprocessing import Process, Pool

class OpticalFlow:

	def __init__(self, filepath, outpath):

		self.frames = []
		self.filepath = filepath
		self.outpath = outpath
		self.thread_process_count = 1

		self.threads_finished = 0
		self.all_done = False

	def load_frames(self, filepath):

		cap = cv2.VideoCapture(filepath)

		while True:

			frame = cap.read()[1]

			if frame is None: break
			
			self.frames.append(frame)

		cap.release()
		del cap

		print("Total number of frames: %s" % len(self.frames))

	def optical_flow_video(self, frames, first_frame_idx):

		thread_no = (first_frame_idx/self.step_size)

		flows = pd.DataFrame(columns=['x', 'y', 'frame'])
		for i in range(1,len(frames)):

			print("Thread no. %s reporting, first_frame: %s, total frames:%s and working on frame:%s\n" 
				% (thread_no, first_frame_idx, len(frames), i))

			curr = cv2.cvtColor(frames[i],cv2.COLOR_BGR2GRAY)
			prev = cv2.cvtColor(frames[i-1],cv2.COLOR_BGR2GRAY)

			flow = cv2.calcOpticalFlowFarneback(prev, curr, None, 0.5, 3, 15, 3, 5, 1.2, 0)

			df = pd.DataFrame()
			df['x'] = [round(item, 3) for item in flow[:,:,0].flatten()]
			df['y'] = [round(item, 3) for item in flow[:,:,1].flatten()]
			df['frame'] = first_frame_idx+i

			flows = flows.append(df)

		flows.to_csv("%s_%s" % (self.outpath, thread_no))

		print("Thread no.%s finished" % (thread_no))

	def start(self, thread_count=1):

		self.starting_time = datetime.datetime.now()
		
		self.thread_process_count = thread_count

		self.load_frames(self.filepath)
		self.step_size = len(self.frames)/self.thread_process_count

		for i in range(self.thread_process_count):

			first = i*self.step_size
			last = (i+1)*self.step_size

			last = len(self.frames) if last >= len(self.frames) else last

			segment = self.frames[first:last]

			thread.start_new_thread(self.optical_flow_video, (segment, first))

		while self.all_done == False:
			pass

	def start_multiprocessing(self, process_count=1):

		self.starting_time = datetime.datetime.now()
		
		self.thread_process_count = process_count

		self.load_frames(self.filepath)
		self.step_size = len(self.frames)/self.thread_process_count

		print("Step size: %s" % self.step_size)

		self.pool = Pool(processes=self.thread_process_count)

		self.pool.map(star_func,
			[(self.frames[i*self.step_size:(i+1)*self.step_size], i*self.step_size) for i in range(process_count)])

		# while self.all_done == False:
		# 	pass


if __name__ == "__main__":

	def star_func(params):
		"""Converts f([1,2]) to f(1,2). A quirk needed to pass
		multiple params to optical_flow_video"""
		
		return of.optical_flow_video(*params)

	parser = argparse.ArgumentParser()
	parser.add_argument("--inpath")
	parser.add_argument("--outpath")
	parser.add_argument("--processthreadcount")
	  
	args = parser.parse_args()

	of = OpticalFlow(args.inpath, args.outpath)
	of.start_multiprocessing(int(args.processthreadcount))
