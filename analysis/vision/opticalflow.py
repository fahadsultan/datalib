import cv2 
import numpy as np
import argparse

import datetime

import thread
import json

class OpticalFlow:

	def __init__(self, filepath, outpath):

		self.flows = {}
		self.frames = []
		self.filepath = filepath
		self.outpath = outpath
		self.thread_count = 1

		self.threads_finished = []
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

		for i in range(1,len(self.frames)):

			print("Thread no. %s reporting\n" % thread_no)

			curr = cv2.cvtColor(self.frames[i],cv2.COLOR_BGR2GRAY)
			prev = cv2.cvtColor(self.frames[i-1],cv2.COLOR_BGR2GRAY)

			flow = cv2.calcOpticalFlowFarneback(prev, curr, None, 0.5, 3, 15, 3, 5, 1.2, 0)

			self.flows[first_frame_idx+i] = list(flow.flatten())

		self.threads_finished.append(True)

		print("Thread #%s finished" % thread_no)

		if len(self.threads_finished) == self.thread_count:
			
			print("ALL THREADS FINISHED EXECUTION")

			with open(self.outpath, 'w') as f:
				json.dump(self.flows, f)

			total_time = datetime.datetime.now() - self.starting_time
			
			print("TOTAL TIME ELAPSED: %s" % (total_time))

			self.all_done = True
			
	def start(self, thread_count=1):

		self.starting_time = datetime.datetime.now()
		
		self.thread_count = thread_count

		self.load_frames(self.filepath)
		self.step_size = len(self.frames)/self.thread_count

		for i in range(self.thread_count):

			first = i*self.step_size
			last = (i+1)*self.step_size

			last = len(self.frames) if last >= len(self.frames) else last

			segment = self.frames[first:last]

			thread.start_new_thread(self.optical_flow_video, (segment, first))

		while self.all_done == False:
			pass

if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("--inpath")
	parser.add_argument("--outpath")
	parser.add_argument("--threadcount")
	  
	args = parser.parse_args()

	OpticalFlow(args.inpath, args.outpath).start(args.threadcount)
