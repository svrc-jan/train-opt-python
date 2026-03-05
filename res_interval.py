#!.venv/bin/python3

import sys
import gurobipy as gp
import numpy as np

from dataclasses import dataclass, field
from typing import List, NamedTuple
from instance import Instance
from preprocess import Preprocess
from path_select import Path_selection


MAX_DUR = 100000
DEFAULT_DATA = 'data/nor1_critical_0.json'


class Res_use(NamedTuple):
	level_lock: int = -1
	level_unlock: int = -1
	res_time: float = 0.0


class Res_int(NamedTuple):
	use: Res_use
	time_lock: float
	time_unlock: float

@dataclass
class Section:
	idx_start: int = -1
	idx_end: int = -1


@dataclass
class Path:
	ops: List[int] = field(default_factory=list)
	sections: List[Section] = field(default_factory=list)


class Res_interval:
	inst: Instance
	prepr: Preprocess
	
	paths: List[Path]
	res_uses: List[List[Res_use]]
	

	def __init__(self, prepr: Preprocess):
		self.inst = prepr.inst
		self.prepr = prepr


	def set_paths(self, paths):
		self.paths = [Path(ops=p) for p in paths]

		for t, path in enumerate(self.paths):
			sec = Section(
				level_start	=self.prepr.trains[t].level_start,
				level_end	=self.prepr.trains[t].level_last,
				idx_start	=0,
				idx_end		=len(path.ops),
				dur 		=sum(self.inst.ops[o].dur for o in path.ops)
			)
			path.sections.append(sec)


	def make_res_uses(self):
		self.res_uses = [[] for _ in range(self.inst.n_res)]

		for path in self.paths:
			ru: List[Res_use|None] = [None]*self.inst.n_res

			for o in path.ops:
				op = self.inst.ops[o]

				for res in op.res:
					if ru[res.idx] is None:
						ru[res.idx] = Res_use(
							level_lock	=self.prepr.ops[o].level_start,
							level_unlock=self.prepr.ops[o].level_end,
							res_time	=res.time
						)
					else:
						ru[res.idx] = Res_use(
							level_lock	=ru[res.idx].level_lock,
							level_unlock=self.prepr.ops[o].level_end,
							res_time	=res.time
						)
			
			for r in range(self.inst.n_res):
				if not ru[r] is None:
					self.res_uses[r].append(ru[r])

		for r, ru in enumerate(self.res_uses):
			print(r, len(ru))


	def init_model(self):
		self.model = gp.Model()

		self.var_time = {}
		self.cons_dur = {}

		for path in self.paths:
			self.init_path(path)


	def init_path(self, path: Path):
		sec = path.sections[0]

		lvl_start = self.prepr.ops[pat]

		self.add_level(sec.level_start)
		self.add_level(sec.level_end)

		self.cons_dur[sec.level_end] = self.model.addConstr(
			self.var_time[sec.level_start] + sec.dur <= self.var_time[sec.level_end])


	def add_level(self, l: int):
		level = self.prepr.levels[l]

		lb = level.time_lb
		ub = level.time_ub if level.time_ub else float('inf')
		obj = 1.0 if level.n_op_out == 0 else 0.0

		self.var_time[l] = self.model.addVar(lb, ub, obj, gp.GRB.CONTINUOUS, f'time{l}')


	def split_path(self, t: int, l_new: int):
		path = self.paths[t]

		i = 0
		while l_new < path.sections[i].level_end:
			i += 1

		new_sec = 



if __name__ == '__main__':
	data = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATA
	print(data)
	inst = Instance(data)
	prepr = Preprocess(inst)
	path_sel = Path_selection(prepr)
	res_int = Res_interval(prepr)

	# paths = path_sel.select_iqp_paths()
	paths = path_sel.select_non_overlap_paths()

	res_int.set_paths(paths)
	res_int.make_res_uses()

	res_int.init_model()
	res_int.model.optimize()

	# print(res_int.res_uses)
