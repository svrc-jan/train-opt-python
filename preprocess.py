#!.venv/bin/python3

import sys
import itertools as it
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from array import array, ArrayType
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Set

from disjoint_set import Disjoint_set
from instance import Instance, IDX_MAX, TIME_MAX


DEFAULT_DATA = 'data/nor1_full_0.json'


@dataclass(slots=True)
class Junction:
	idx: int = IDX_MAX
	level: int = IDX_MAX
	train: int = IDX_MAX

	time_lb: int = 0
	time_ub: int = TIME_MAX

	succ: List[Tuple[int, int]] = field(default_factory=list)
	pred: List[Tuple[int, int]] = field(default_factory=list)

	@property
	def n_succ(self):
		return len(self.succ)

	@property
	def n_pred(self):
		return len(self.pred)


@dataclass(slots=True)
class Level:
	idx: int = IDX_MAX
	train: int = IDX_MAX

	required: bool = True
	routing: bool = True

	time_lb: int = 0
	time_ub: int = TIME_MAX

	juncts: ArrayType[int] = field(default_factory=lambda: array('I'))

	succ: List[Tuple[int, int]] = field(default_factory=list)
	pred: List[Tuple[int, int]] = field(default_factory=list)

	@property
	def n_succ(self):
		return len(self.succ)

	@property
	def n_pred(self):
		return len(self.pred)


@dataclass(slots=True)
class Train:
	idx: int = IDX_MAX

	junct_first: int = IDX_MAX
	junct_after: int = IDX_MAX
	level_first: int = IDX_MAX
	level_after: int = IDX_MAX

	branch_sections: List[Tuple[Tuple[int, int], int]] = field(default_factory=list)
	choke_sections: List[Tuple[Tuple[int, int], int]] = field(default_factory=list)

	prepr_juncts: List[Junction] = None
	prepr_levels: List[Level] = None

	@property
	def junct_last(self):
		return self.junct_after - 1
	
	@property
	def level_last(self):
		return self.level_after - 1

	@property
	def juncts(self) -> List[Junction]:
		return self.prepr_juncts[self.junct_first:self.junct_after]
	
	@property
	def levels(self) -> List[Level]:
		return self.prepr_levels[self.level_first:self.level_after]
	
	@property
	def junct_range(self):
		return range(self.junct_first, self.junct_after)

	@property
	def level_range(self):
		return range(self.level_first, self.level_after)


@dataclass(slots=True)
class Branch_area:
	idx: int = IDX_MAX
	borders: List[Tuple[int, int]] = field(default_factory=list)
	sections: Dict[int, Tuple[int, int]] = field(default_factory=dict)
	res: Set[int] = field(default_factory=set)


@dataclass(slots=True)
class Choke_area:
	idx: int = IDX_MAX
	borders: Tuple[int, int] = (IDX_MAX, IDX_MAX)
	sections: Dict[int, Tuple[int, int]] = field(default_factory=dict)
	res: Set[int] = field(default_factory=set)


class Preprocess:
	inst: Instance

	juncts: List[Junction]
	levels: List[Level]
	trains: List[Train]

	branch_areas: List[Branch_area]
	choke_areas: List[Choke_area]

	op_junct_start: ArrayType[int]
	op_junct_end: ArrayType[int]

	op_level_start: ArrayType[int]
	op_level_end: ArrayType[int]
	
	op_route_start: ArrayType[int]
	op_route_end: ArrayType[int]

	op_required: ArrayType[int]
	op_choke: ArrayType[int]
	
	res_choke: ArrayType[int]
	res_choke_area: ArrayType[int]

	def __init__(self, inst):
		self.inst = inst

		self.make_junctions()
		self.make_levels()

		self.make_junction_bounds()
		self.make_level_bounds()

		self.find_required_levels()
		self.find_required_ops()

		self.find_choke_res()
		self.find_choke_ops()

		self.make_sections()
		self.make_areas()

		self.make_area_resources()
		self.add_boundary_sections()

		n_required = sum(level.required for level in self.levels)
		n_boundary_choke = sum(area.borders[1] == IDX_MAX for area in self.choke_areas)
		
		print(f'Preprocess - junctions: {self.n_juncts}, levels: {self.n_levels} (required: {n_required})\n' + 
			  f'             branch areas: {self.n_branch_areas}, choke areas: {self.n_choke_areas} (boundary: {n_boundary_choke})')

	def make_junctions(self):
		self.op_junct_start = array('I')
		self.op_junct_end = array('I')

		for _ in range(self.inst.n_ops):
			self.op_junct_start.append(IDX_MAX)
			self.op_junct_end.append(IDX_MAX)

		self.trains = [Train(idx=i) for i in range(self.inst.n_trains)]

		n_juncts = 0

		for t, inst_train in enumerate(self.inst.trains):

			disj_set = Disjoint_set(inst_train.n_ops)

			for op in self.inst.train_ops(t):
				for i in range(op.n_succ):
					for j in range(i+1, op.n_succ):
						a = op.succ[i] - inst_train.op_first
						b = op.succ[j] - inst_train.op_first
						disj_set.union_set(a, b)
			
			self.trains[t].junct_first = n_juncts
			n_juncts += disj_set.n_sets + 1
			self.trains[t].junct_after = n_juncts
			

			succ_set = disj_set.get_result()
			for i, j in enumerate(succ_set):
				self.op_junct_start[i + inst_train.op_first] = j + self.trains[t].junct_first

			self.op_junct_end[inst_train.op_last] = self.trains[t].junct_last

		for o in range(self.inst.n_ops):
			assert(self.op_junct_start[o] < n_juncts)
			
			for p in self.inst.ops[o].pred:
				if self.op_junct_end[p] == IDX_MAX:
					self.op_junct_end[p] = self.op_junct_start[o]
				else:
					assert(self.op_junct_end[p] == self.op_junct_start[o])

		self.juncts = [Junction(idx=i) for i in range(n_juncts)]

		for o in range(self.inst.n_ops):
			assert(self.op_junct_end[o] < n_juncts)

			j_start = self.op_junct_start[o]
			j_end = self.op_junct_end[o]

			self.juncts[j_start].succ.append((j_end, o))
			self.juncts[j_end].pred.append((j_start, o))

		for train in self.trains:
			train.prepr_juncts = self.juncts

			for junct in train.juncts:
				assert(junct.train == IDX_MAX)
				junct.train = train.idx


	def make_levels(self):
		in_deg = [junct.n_pred for junct in self.juncts]

		self.levels = []

		for train in self.trains:
			train.level_first = self.n_levels

			zero_in_deg = [train.junct_first]

			while zero_in_deg:
				level = Level(idx=self.n_levels)
				level.juncts = array('I', zero_in_deg)

				self.levels.append(level)

				zero_in_deg = []
				for j in level.juncts:
					for s, _ in self.juncts[j].succ:
						in_deg[s] -= 1
						if in_deg[s] == 0:
							zero_in_deg.append(s)
			
			train.level_after = self.n_levels
	
		for level in self.levels:
			for j in level.juncts:
				self.juncts[j].level = level.idx
		
		for train in self.trains:
			train.prepr_levels = self.levels

			for level in train.levels:
				assert(level.train == IDX_MAX)
				level.train = train.idx

		self.op_level_start = array('I')
		self.op_level_end = array('I')

		for o in range(self.inst.n_ops):

			l_start = self.juncts[self.op_junct_start[o]].level
			l_end = self.juncts[self.op_junct_end[o]].level

			self.op_level_start.append(l_start)
			self.op_level_end.append(l_end)

			self.levels[l_start].succ.append((l_end, o))
			self.levels[l_end].pred.append((l_start, o))

	
	def find_required_levels(self):
		for o in range(self.inst.n_ops):
			for l in range(self.op_level_start[o] + 1, self.op_level_end[o]):
				self.levels[l].required = False

	
	def find_routing_levels(self):
		for level in self.levels:
			if level.required:
				for _, o in level.pred:
					if self.inst.ops[o].n_succ < level.n_succ:
						level.routing = False
						break
			else:
				level.routing = False
			
			if level.routing:
				for _, o in level.succ:
					if self.inst.ops[o].n_pred < level.n_pred:
						level.routing = False
						break
		
		self.op_route_start = array('I')
		self.op_route_end = array('I')

		for o, op in enumerate(self.inst.ops):
			train = self.trains[op.train]

			r_start = self.op_level_start[o]
			while not self.levels[r_start].routing:
				r_start -= 1

			assert(r_start >= train.level_first)

			r_end = self.op_level_end[o]
			while not self.levels[r_end].routing:
				r_end += 1
				
			assert(r_end <= train.level_last)

			self.op_route_start.append(r_start)
			self.op_route_end.append(r_end)


	def make_junction_bounds(self):
		for junct in self.juncts:
			if junct.n_succ > 0:
				junct.time_lb = min(self.inst.ops[o].start_lb for _, o in junct.succ)
				junct.time_ub = max(self.inst.ops[o].start_ub for _, o in junct.succ)
			else:
				ops_in = [self.inst.ops[o] for _, o in junct.pred]
				junct.time_lb = min(op.start_lb + op.dur for op in ops_in)
				junct.time_ub = max(op.start_ub + op.dur for op in ops_in)


	def make_level_bounds(self):
		for level in self.levels:
			level.time_lb = min(self.juncts[j].time_lb for j in level.juncts)
			level.time_ub = max(self.juncts[j].time_ub for j in level.juncts)

	
	def train_juncts(self, t: int) -> List[Junction]:
		train = self.trains[t]
		return self.juncts[train.junct_first:train.junct_after]
		


	def count_routes(self):
		n_routes = 0

		def dfs(route: List[int]):
			o = route[-1]

			if self.op_level_end[o] == self.op_route_end[o]:
				return 1
			
			count = 0
			for s in self.inst.ops[o].succ:
				route.append(s)
				count += dfs(route)
				route.pop()
			
			return count


		for level in self.levels:
			if level.routing and level.n_succ > 1:
				# print(f'level {level.idx}:')
				for _, o in level.succ:
					route = [o]
					n_routes += dfs(route)


			if (level.idx + 1) % 100 == 0:
				print(f'level {level.idx + 1}/{self.n_levels}, count: {n_routes}', end='\r')
		
		print(f'\ntotal {n_routes} routes')


	def draw_train_graph(self, t: int):
		juncts = self.train_juncts(t)

		max_succ = max(junct.n_succ for junct in juncts)
		G = nx.MultiDiGraph()

		connectionstyle = [f"arc3,rad={r}" for r in it.accumulate([0.15] * max_succ)]
		
		for junct in juncts:
			G.add_node(junct.idx, level=junct.level)
		
		for junct in juncts:
			for s, o in junct.succ:
				G.add_edge(junct.idx, s, op=o)
		
		pos = nx.multipartite_layout(G, subset_key='level')

		plt.figure()
		nx.draw_networkx_nodes(G, pos)
		nx.draw_networkx_edges(G, pos, connectionstyle=connectionstyle)
		
		plt.title(f'train {t}')

	
	def print_section_lengths(self):
		max_len = 0
		for train in self.trains:
			rout_levels = [level.idx for level in train.levels if level.routing]
			sec_len = [b - a for a, b, in zip(rout_levels[:-1], rout_levels[1:])]

			max_len = max(max_len, *sec_len)
		
		print(f'max section length {max_len}')


	def find_required_ops(self):
		self.op_required = array('B')

		for o in range(self.inst.n_ops):
			level = self.levels[self.op_level_start[o]]
			op_req = (level.required) and (level.n_succ == 1)
			self.op_required.append(op_req)

	
	def find_choke_res(self):
		# 0 = unused
		# 1 = used
		# 2 = used by choke op
		res_class = np.zeros((self.inst.n_trains, self.inst.n_res))

		for op in self.inst.ops:
			for res in op.res:
				res_class[op.train][res] = 1

		for op in self.inst.ops:
			if self.op_required[op.idx]:
				for res in op.res:
					res_class[op.train][res] = 2

		self.res_choke = array('B')
		self.res_choke.fromlist(np.all(res_class != 1, axis=0).tolist())


	def find_choke_ops(self):
		self.op_choke = array('B')

		for op in self.inst.ops:
			if self.op_required[op.idx]:
				is_choke = True
				for res in op.res:
					if self.res_choke[res] == 0:
						is_choke = False
						break
			else:
				is_choke = False
			
			self.op_choke.append(is_choke)

		for train in self.inst.trains:
			border = []
			for o in train.op_range:
				if self.op_choke[o]:
					border.append(o)

		
	def make_sections(self):
		sections: List[Tuple[int, int, int]] = []
		
		for train in self.inst.trains:
			prepr_train = self.trains[train.idx]
			border_start = [prepr_train.level_first]
			border_end = []
			
			for o in train.op_range:
				if self.op_choke[o]:
					border_end.append(self.op_level_start[o])
					border_start.append(self.op_level_end[o])

			border_end.append(prepr_train.level_last)


			for l_start, l_end in zip(border_start, border_end):
				assert(l_start <= l_end)
				if (l_start < l_end):
					sections.append((l_start, l_end, train.idx))

		res_sections = [set() for _ in range(self.inst.n_res)]

		for s, sec in enumerate(sections):
			l_start, l_end, _ = sec
			
			for l in range(l_start, l_end):
				for _, o in self.levels[l].succ:
					for r in self.inst.ops[o].res:
						res_sections[r].add(s)
		
		disj_set = Disjoint_set(len(sections))
		for r, res_sec in enumerate(res_sections):
			if self.res_choke[r]:
				continue

			res_sec = list(res_sec)
			for i, j in zip(res_sec[:-1], res_sec[1:]):
				disj_set.union_set(i, j)

		sec_area = disj_set.get_result(True)
		self.branch_areas = [Branch_area(idx=i) for i in range(disj_set.n_sets)]

		branch_sections = [{} for _ in range(self.inst.n_trains)]

		for s, sec in enumerate(sections):
			l_start, l_end, t = sec

			bs = branch_sections[t]
			a = sec_area[s]

			if a in bs:
				bs = (min(l_start, bs[a][0]), max(l_end, bs[a][1]))
			else:
				bs[a] = (l_start, l_end)


		for train in self.trains:
			bs = [((v[0], v[1]), k) for k, v in branch_sections[train.idx].items()]
			bs.sort()
			
			cs = []
				
			for ((_, l_start), a_start), ((l_end, _), a_end) in zip(bs[:-1], bs[1:]):
				cs.append(((l_start, l_end), (a_start, a_end)))

			train.branch_sections = bs
			train.choke_sections = cs


	def make_areas(self):
		ca = set()

		for train in self.trains:
			for _, a in train.choke_sections:
				if a[0] > a[1]:
					a = a[1], a[0]

				ca.add(a)

		ca = list(ca)
		ca.sort()

		ca_mp = { v: i for i, v in enumerate(ca) }
		self.choke_areas = [Choke_area(idx=i, borders=b) for i, b in enumerate(ca)]

		for train in self.trains:
			for sec, a in train.branch_sections:
				self.branch_areas[a].sections[train.idx] = sec
			
			new_cs = []
			for sec, a in train.choke_sections:
				if a[0] > a[1]:
					a = a[1], a[0]
					sec = sec[1], sec[0]
				i = ca_mp[a]

				new_cs.append((sec, i))
				self.choke_areas[i].sections[train.idx] = sec

			train.choke_sections = new_cs

	def make_area_resources(self):
		self.res_choke_area = array('I')
		for _ in range(self.inst.n_res):
			self.res_choke_area.append(IDX_MAX)
		
		for area in self.branch_areas:
			for l_start, l_end in area.sections.values():
				for l in range(l_start, l_end):
					for _, o in self.levels[l].succ:
						area.res.update(r for r in self.inst.ops[o].res)
		
		for area in self.choke_areas:
			for sec in area.sections.values():
				for l in range(min(sec), max(sec)):
					for _, o in self.levels[l].succ:
						area.res.update(r for r in self.inst.ops[o].res)

		for area in self.choke_areas:
			for r in area.res:
				assert(self.res_choke_area[r] == IDX_MAX)
				self.res_choke_area[r] = area.idx
		

	def add_boundary_sections(self):
		for train in self.trains:
			self.add_start_boundary(train)
			self.add_end_boundary(train)


	def add_start_boundary(self, train: Train):
		l_start = train.level_first
		(l_end, _), a_end = train.branch_sections[0]

		while l_start < l_end:
			if any(self.inst.ops[o].n_res > 0 for _, o in self.levels[l_start].succ):
				break
			l_start += 1

		start_ops = array('I')
		for l in range(l_start, l_end):
			start_ops.extend(o for _, o in self.levels[l].succ)
		
		assert(all(self.op_choke[o] for o in start_ops))

		if len(start_ops) > 0:
			res = set()
			for o in start_ops:
				res.update(r for r in self.inst.ops[o].res)

			a = IDX_MAX
			for r in res:
				if a < IDX_MAX:
					assert(a == self.res_choke_area[r])
				else:
					a = self.res_choke_area[r]

			if a < IDX_MAX:
				ca = self.choke_areas[a]
				if ca.borders[1] == a_end:
					ca.sections[train.idx] = (l_start, l_end)
				else:
					assert(ca.borders[0] == a_end)
					ca.sections[train.idx] = (l_end, l_start)
			else:
				ca = Choke_area(idx=self.n_choke_areas)
				ca.borders = (a_end, IDX_MAX)
				ca.sections[train.idx] = (l_end, l_start)

				for r in res:
					self.res_choke_area[r] = ca.idx

				self.choke_areas.append(ca)
			

	def add_end_boundary(self, train: Train):
		l_end = train.level_last
		(_, l_start), a_start = train.branch_sections[-1]

		while l_start < l_end:
			if any(self.inst.ops[o].n_res > 0 for _, o in self.levels[l_end].pred):
				break
			l_end -= 1

		end_ops = array('I')
		for l in range(l_start, l_end):
			end_ops.extend(o for _, o in self.levels[l].succ)
		
		assert(all(self.op_choke[o] for o in end_ops))

		if len(end_ops) > 0:
			res = set()
			for o in end_ops:
				res.update(r for r in self.inst.ops[o].res)

			a = IDX_MAX
			for r in res:
				if a < IDX_MAX:
					assert(a == self.res_choke_area[r])
				else:
					a = self.res_choke_area[r]

			if a < IDX_MAX:
				ca = self.choke_areas[a]
				if ca.borders[0] == a_start:
					ca.sections[train.idx] = (l_start, l_end)
				else:
					assert(ca.borders[1] == a_start)
					ca.sections[train.idx] = (l_end, l_start)
			else:
				ca = Choke_area(idx=self.n_choke_areas)
				ca.borders = (a_start, IDX_MAX)
				ca.sections[train.idx] = (l_start, l_end)

				for r in res:
					self.res_choke_area[r] = ca.idx

				self.choke_areas.append(ca)


	@property
	def n_juncts(self):
		return len(self.juncts)

	@property
	def n_levels(self):
		return len(self.levels)
	
	@property
	def n_branch_areas(self):
		return len(self.branch_areas)
	
	@property
	def n_choke_areas(self):
		return len(self.choke_areas)


if __name__ == '__main__':
	data = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATA
	print(data)
	inst = Instance(data)
	prepr = Preprocess(inst)
	# prepr.print_section_lengths()
	# prepr.print_choke_op()
	
	# for t in range(inst.n_trains):
	# 	prepr.draw_train_graph(t)

	# plt.show()
