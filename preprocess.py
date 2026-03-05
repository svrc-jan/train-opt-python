#!.venv/bin/python3

import sys
import numpy as np

from copy import copy
from typing import List, Dict
from dataclasses import dataclass, field
from disjoint_set import Disjoint_set
from instance import Instance

MAX_DUR = 100000
DEFAULT_DATA = 'data/nor1_critical_0.json'


@dataclass
class Level:
	idx: int = -1
	train: int = -1

	time_lb: int = 0
	time_ub: int|None = None

	ops_in: List[int] = field(default_factory=list)
	ops_out: List[int] = field(default_factory=list)

	@property
	def n_op_in(self):
		return len(self.ops_in)

	@property
	def n_op_out(self):
		return len(self.ops_out)


@dataclass
class Train:
	idx: int = -1
	level_start: int = -1
	level_end: int = -1

	@property
	def level_last(self):
		return self.level_end - 1


@dataclass
class Op:
	idx: int = -1
	level_start: int = -1
	level_end: int = -1


class Preprocess:
	inst: Instance
	trains: List[Train]
	levels: List[Level]

	def __init__(self, inst):
		self.inst = inst

		self.make_levels()
		self.make_level_bounds()


	def make_levels(self):
		self.trains = []
		self.levels = []

		disj_set = Disjoint_set(self.inst.n_ops)
		

		for op in self.inst.ops:
			for i, s1 in enumerate(op.succ):
				for s2 in op.succ[i+1:]:
					disj_set.union_set(s1, s2)

		sets = disj_set.get_sets()

		succ_group = [sets[disj_set.find_set(op.succ[0])] if op.n_succ > 0 else []
			for op in self.inst.ops]
		
		in_deg = [0]*self.inst.n_ops

		for o in range(self.inst.n_ops):
			for s in succ_group[o]:
				in_deg[s] += 1

		for inst_train in self.inst.trains:
			train = Train(idx=self.n_trains)
			

			train.level_start = self.n_levels

			zero_in = [inst_train.op_start]

			while zero_in:
				ops_out = copy(zero_in)
				zero_in = []

				for o in ops_out:
					for s in succ_group[o]:
						in_deg[s] -= 1
						if in_deg[s] == 0:
							zero_in.append(s)
				
				level = Level(idx=self.n_levels)
				level.ops_out = copy(ops_out)

				self.levels.append(level)
			
			#end level
			self.levels.append(Level(idx=self.n_levels))

			train.level_end = self.n_levels
			self.trains.append(train)

		self.ops = [Op(i) for i in range(self.inst.n_ops)]
				
		for level in self.levels:
			for o in level.ops_out:
				self.ops[o].level_start = level.idx

		for op in self.inst.ops:
			if op.n_succ > 0:
				self.ops[op.idx].level_end = self.ops[op.succ[0]].level_start

			else:
				self.ops[op.idx].level_end = self.trains[op.train].level_last

		for op in self.ops:
			self.levels[op.level_end].ops_in.append(op.idx)


	def make_level_bounds(self):
		for level in self.levels:
			if level.n_op_out > 0:
				lbs = [self.inst.ops[o].start_lb for o in level.ops_out]
				ubs = [self.inst.ops[o].start_ub for o in level.ops_out]
				
			else:
				ops_in = [self.inst.ops[o] for o in level.ops_in]
				lbs = [op.start_lb + op.dur for op in ops_in]
				ubs = [op.start_ub + op.dur if op.start_ub else None for op in ops_in]

			level.time_lb = min(lbs)
			if not any(x is None for x in ubs):
				level.time_ub = max(ubs)


	@property
	def n_trains(self):
		return len(self.trains)

	@property
	def n_levels(self):
		return len(self.levels)

def test_levels(prepr: Preprocess):
	for op in prepr.inst.ops:
		for s in op.succ:
			assert(prepr.ops[op.idx].level_end == prepr.ops[s].level_start)

if __name__ == '__main__':
	data = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATA
	print(data)
	inst = Instance(data)
	prepr = Preprocess(inst)

	test_levels(prepr)
