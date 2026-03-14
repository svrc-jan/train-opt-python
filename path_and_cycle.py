#!.venv/bin/python3

import sys
import gurobipy as gp

from gurobipy import GRB
from instance import Instance
from preprocess import Preprocess


DEFAULT_DATA = 'data/nor1_critical_0.json'


class Path_and_cycle:
	inst: Instance
	prepr: Preprocess

	def __init__(self, prepr: Preprocess):
		self.inst = prepr.inst
		self.prepr = prepr

if __name__ == '__main__':
	data = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATA
	print(data)
	inst = Instance(data)
	prepr = Preprocess(inst)

	pnc = Path_and_cycle(prepr)
