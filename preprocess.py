#!.venv/bin/python3

import sys

from copy import copy
from typing import List, Dict
from dataclasses import dataclass, field
from disjoint_set import Disjoint_set
from instance import Instance

MAX_DUR = 100000
DEFAULT_DATA = 'data/nor1_critical_0.json'


@dataclass
class Op:
	junct_start: int = -1
	junct_end: int = -1
	level_start: int = -1
	level_end: int = -1


@dataclass
class Junction:
	idx: int = -1
	level: int = -1

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
class Level:
	idx: int = -1
	juncts: List[int] = field(default_factory=list)


@dataclass
class Train:
	junct_start: int = -1
	junct_end: int = -1
	level_start: int = -1
	level_end: int = -1

	@property
	def junct_last(self):
		return self.junct_end - 1
	
	@property
	def level_last(self):
		return self.level_end - 1


class Preprocess:
	inst: Instance
	
	ops: List[Op]
	juncts: List[Junction]
	levels: List[Level]
	trains: List[Train]
	
	def __init__(self, inst):
		self.inst = inst

		self.make_junctions()
		self.make_levels()

		print(f'Preprocess - junctions: {self.n_juncts}, levels: {self.n_levels}')

	def make_junctions(self):
		self.ops = [Op() for _ in range(self.inst.n_ops)]
		self.trains = [Train() for _ in range(self.inst.n_trains)]

		n_juncts = 0

		for t, inst_train in enumerate(self.inst.trains):

			disj_set = Disjoint_set(inst_train.n_ops)

			for op in self.inst.ops[inst_train.op_start:inst_train.op_end]:
				for i in range(op.n_succ):
					for j in range(i+1, op.n_succ):
						a = op.succ[i] - inst_train.op_start
						b = op.succ[j] - inst_train.op_start
						disj_set.union_set(a, b)
			
			self.trains[t].junct_start = n_juncts
			n_juncts += disj_set.n_sets + 1
			self.trains[t].junct_end = n_juncts
			

			succ_set = disj_set.get_result()
			for i, j in enumerate(succ_set):
				self.ops[i + inst_train.op_start].junct_start = j + self.trains[t].junct_start

			self.ops[inst_train.op_last].junct_end = self.trains[t].junct_last

		for o, op in enumerate(self.ops):
			assert(op.junct_start >= 0 and op.junct_start < n_juncts)
			
			for p in self.inst.ops[o].pred:
				if self.ops[p].junct_end == -1:
					self.ops[p].junct_end = op.junct_start
				else:
					assert(self.ops[p].junct_end == op.junct_start)

		self.juncts = [Junction(idx=i) for i in range(n_juncts)]

		for o, op in enumerate(self.ops):
			assert(op.junct_end >= 0 and op.junct_end < n_juncts)

			self.juncts[op.junct_start].ops_out.append(o)
			self.juncts[op.junct_end].ops_in.append(o)


	def make_levels(self):
		in_deg = [junct.n_op_in for junct in self.juncts]

		self.levels = []

		for train in self.trains:
			train.level_start = self.n_levels

			zero_in_deg = [train.junct_start]

			while zero_in_deg:
				level = Level(idx=self.n_levels, juncts=copy(zero_in_deg))
				self.levels.append(level)

				zero_in_deg = []
				for j in level.juncts:
					for o in self.juncts[j].ops_out:
						s = self.ops[o].junct_end
						in_deg[s] -= 1
						if in_deg[s] == 0:
							zero_in_deg.append(s)
	
		for level in self.levels:
			for j in level.juncts:
				self.juncts[j].level = level.idx

		for junct in self.juncts:
			for o in junct.ops_in:
				self.ops[o].level_end = junct.level

			for o in junct.ops_out:
				self.ops[o].level_start = junct.level


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
	def n_juncts(self):
		return len(self.juncts)

	@property
	def n_levels(self):
		return len(self.levels)


	def test_juncts(self):
		for j, junct in enumerate(self.juncts):
			for o in junct.ops_in:
				assert(self.ops[o].junct_end == j)

			
			for o in junct.ops_out:
				assert(self.ops[o].junct_start == j)


if __name__ == '__main__':
	data = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATA
	print(data)
	inst = Instance(data)
	prepr = Preprocess(inst)

	prepr.test_juncts()
