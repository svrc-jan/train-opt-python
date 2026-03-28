#!.venv/bin/python3

import sys

from array import array, ArrayType
from typing import List, Tuple

from instance import Instance, IDX_MAX, TIME_MAX
from preprocess import Preprocess


DEFAULT_DATA = 'data/nor1_critical_0.json'



class Solver:
	inst: Instance
	prepr: Preprocess

	choke_order: List[ArrayType[int]]

	def __init__(self, prepr: Preprocess):
		self.prepr = prepr
		self.inst = prepr.inst

		self.make_choke_order()

	def make_choke_order(self):
		self.choke_order = []

		for area in self.prepr.choke_areas:

			order = []

			print(area.idx, [abs(sec[0] - sec[1]) for sec in area.sections.values()])

			for t, sec in area.sections.items():

				mid = self.prepr.levels[sec[0]].time_lb + self.prepr.levels[sec[1]].time_lb
				order.append((mid, t))

			order.sort()

			self.choke_order.append(array('I', [x[1] for x in order]))

	def print_branch_order(self):
		for area in self.prepr.branch_areas:
			print(f'branch area {area.idx}')
			for _, ca in area.borders:
				print(ca, self.choke_order[ca])
	

if __name__ == '__main__':
	data = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATA
	print(data)
	inst = Instance(data)
	prepr = Preprocess(inst)
	
	slvr = Solver(prepr)
	# slvr.print_branch_order()
