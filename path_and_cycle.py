#!.venv/bin/python3

import sys
import numpy as np
import gurobipy as gp

from typing import List, Tuple
from copy import copy
from gurobipy import GRB

from instance import Instance
from preprocess import Preprocess


DEFAULT_DATA = 'data/nor1_critical_0.json'


class Graph:
	# res_backward: List[List[int, int, int, bool]]
	res_forward: List[List[Tuple[int, int, int, bool]]]
	
	def __init__(self, time_lbs: List[int], path_durs: List[int]):
		self.set_paths(time_lbs, path_durs)


	def set_paths(self, time_lbs: List[int], path_durs: List[int]):
		self.n_alts = 0
		self.n_nodes = len(time_lbs)
		self.time_lbs = time_lbs

		has_in = [False]*self.n_nodes
		self.path_durs = path_durs

		for u, d, in enumerate(path_durs):
			if d >= 0:
				has_in[u + 1] = True
		
		self.start_nodes = [u for u in range(self.n_nodes) if not has_in[u]]
		self.last_nodes = [u for u in range(self.n_nodes) if self.path_durs[u] == -1]
		
		self.res_forward = [[] for _ in range(self.n_nodes)]


	def add_res_alt(self, u: int, v: int, d_u: int, d_v: int):
		# if u > v:
		# 	u, v = v, u

		alt_idx = self.n_alts
		self.n_alts += 1

		# self.res_backward[op2.level_start] = (op1.level_end, d1, alt_idx, False)
		self.res_forward[u + 1].append((v, d_u, alt_idx, False))

		# self.res_backward[op1.level_start] = (op2.level_end, d1, alt_idx, True)
		self.res_forward[v + 1].append((u, d_v, alt_idx, True))


	def get_cycles(self, alts: List[bool]):
		
		visited = [False]*self.n_nodes
		on_stack = [False]*self.n_nodes

		def find_cycle(u: int, cycle: List[int]):
			if on_stack[u]:
				on_stack[u] = False
				return u
			
			if visited[u]:
				on_stack[u] = False
				return -1

			visited[u] = True
			on_stack[u] = True

			for v, _, ai, av in self.res_forward[u]:
				if alts[ai] == av:
					r = find_cycle(v, cycle)
					if r != -1:
						cycle.append((ai, av))
						on_stack[u] = False
						return r if r != u else -2
					

			d = self.path_durs[u]
			if d >= 0:
				r = find_cycle(u + 1, cycle)
				if r != -1:
					on_stack[u] = False
					return r if r != u else -2


			on_stack[u] = False
			return -1


		cycles = []
		for u in self.start_nodes:
			c = []
			find_cycle(u, c)

			if len(c) > 0:
				cycles.append(c)

		return cycles
	

	def get_paths(self, alts: List[bool]):
		in_deg = [0]*self.n_nodes
		
		for u in range(self.n_nodes):
			d = self.path_durs[u]
			if d >= 0:
				in_deg[u + 1] += 1

			for v, _, ai, av in self.res_forward[u]:
				if alts[ai] == av:
					in_deg[v] += 1

		t = copy(self.time_lbs)
		pred_node = [-1]*self.n_nodes
		pred_var = [None]*self.n_nodes
		
		q = [u for u in range(self.n_nodes) if in_deg[u] == 0]
		while q:
			u = q.pop()

			d = self.path_durs[u]
			if d >= 0:
				v = u + 1
				if t[v] < t[u] + d:
					t[v] = t[u] + d
					pred_node[v] = u
					pred_var[v] = None

				in_deg[v] -= 1
				if in_deg[v] == 0:
					q.append(v)
			
			for v, d, ai, av in self.res_forward[u]:
				if alts[ai] == av:
					if t[v] < t[u] + d:
						t[v] = t[u] + d
						pred_node[v] = u
						pred_var[v] = (ai, av)

					in_deg[v] -= 1
					if in_deg[v] == 0:
						q.insert(0, v)

		return t, pred_node, pred_var

	
	def extract_path_vars(self, last, pred_node, pred_var):
		path_vars = [pred_var[last]]
		node = pred_node[last]

		while node >= 0:
			path_vars.append(pred_var[node])
			node = pred_node[node]

		path_vars = [x for x in path_vars if x is not None]

		return path_vars


class Path_and_cycle:
	inst: Instance
	graph: Alt_graph

	res_uses: List[List[Tuple[int, int, int]]]

	def __init__(self, inst: Preprocess):
		self.inst = inst

	def set_paths(self, paths=None):
		if paths is None:
			paths = [self.get_random_path(t) for t in range(self.inst.n_trains)]

		self.paths = paths

		self.prep_graph()
		self.prep_res_uses()
		self.prep_model()


	def get_random_path(self, t: int):
		train = self.inst.trains[t]

		path = [train.op_start]

		while path[-1] != train.op_last:
			op = self.inst.ops[path[-1]]
			s = int(np.random.choice(op.succ))
			path.append(s)
		
		return path
	

	def solve(self):
		while True:
			self.m.update()
			self.m.optimize()

			assert(self.m.Status == GRB.OPTIMAL)

			alt_vals = self.get_alt_values()
			if self.check_and_add_cycles(alt_vals):
				continue

			time_vals = self.get_time_values()
			graph_paths = self.graph.get_paths(alt_vals)

			if self.check_and_add_paths(time_vals, graph_paths):
				continue

			node_times = graph_paths[0]
			if not self.check_and_add_res_col(node_times):
				print('optimal sol')
				break


	def prep_graph(self):
		time_lbs = []
		path_durs = []

		self.node_map = {}

		for path in self.paths:
			for o in path:
				u = len(time_lbs)
				
				self.node_map[o] = u

				op = self.inst.ops[o]
				time_lbs.append(op.start_lb)
				path_durs.append(op.dur)
				

			last_dur = self.inst.ops[path[-1]].dur
			time_lbs.append(time_lbs[-1] + last_dur)
			path_durs.append(-1)

		self.graph = Alt_graph(time_lbs, path_durs)

		assert(len(self.graph.start_nodes) == self.inst.n_trains)
		assert(len(self.graph.last_nodes) == self.inst.n_trains)

	
	def prep_model(self):
		self.m = gp.Model()

		self.var_alt = []
		self.var_time = []

		path_durs = [sum(self.inst.ops[o].dur for o in path) for path in self.paths]
		total_dur = sum(path_durs)

		graph_times, _, _ = self.graph.get_paths([])
		path_times = [graph_times[u] for u in self.graph.last_nodes]

		for i in range(self.inst.n_trains):
			lb = path_times[i]
			ub = lb + total_dur - path_durs[i]
			
			self.var_time.append(self.m.addVar(lb, ub, 1.0, GRB.CONTINUOUS, f't_{i}'))
		
		self.m.ModelSense = 1
		self.m.Params.OutputFlag = 0

	
	def prep_res_uses(self):
		self.res_uses = [[] for _ in range(self.inst.n_res)]

		for t, path in enumerate(self.paths):
			for o in path:
				u = self.node_map[o]

				for res in self.inst.ops[o].res:
					self.res_uses[res.idx].append((t, u, res.time))


	def get_alt_values(self):
		return [v.X > 0.5 for v in self.var_alt]
	

	def get_time_values(self):
		return [int (round(v.X)) for v in self.var_time]


	def check_and_add_cycles(self, alts):
		cycles = self.graph.get_cycles(alts)

		if len(cycles) == 0:
			return False

		self.add_cycle_cons(min(cycles, key=lambda c: len(c)))

		return True

	
	def add_cycle_cons(self, cycle_vars):
		var_expr = self.get_var_expr(cycle_vars)
		cons = gp.quicksum(var_expr) <= len(var_expr) - 1
		self.m.addConstr(cons)
		print(f'cycle cons - vars: {cycle_vars}')


	def check_and_add_paths(self, time_vals, graph_paths):
		node_times, pred_nodes, pred_vars = graph_paths

		cons = None
		diff = 0

		for i, u_last in enumerate(self.graph.last_nodes):
			path_len = node_times[u_last]
			if path_len - time_vals[i] > diff:
				diff = path_len - time_vals[i]
				path_vars = self.graph.extract_path_vars(u_last, pred_nodes, pred_vars)
				cons = (i, path_len, path_vars)
				
		if diff > 0:
			self.add_path_cons(*cons)
			return True
		
		return False

	def add_path_cons(self, i, path_len, path_vars):
		var_expr = self.get_var_expr(path_vars)
		cons = self.var_time[i] >= path_len*(1 + gp.quicksum(var_expr) - len(var_expr))
		self.m.addConstr(cons)
		print(f'path cons - train: {i}, len: {path_len}, vars: {path_vars}')

	
	def check_and_add_res_col(self, node_times: List[int]):
		col = None
		col_time = float('inf')

		for ru in self.res_uses:
			intervals = []
			for t, u, rt in ru:
				s = node_times[u]
				e = node_times[u + 1] + rt

				intervals.append((s, e, t, u, rt))

			intervals.sort()

			for i, a in enumerate(intervals):
				s_a, e_a, t_a, u_a, rt_a = a
				
				if s_a > col_time:
					break	

				for b in intervals[i + 1:]:
					s_b, _, t_b, u_b, rt_b = b
					if s_b >= e_a or s_b >= col_time:
						break

					# same train
					if t_a == t_b:
						continue

					col = (u_a, u_b, rt_a, rt_b)
					col_time = s_b
		
		if col is None:
			return False
		

		self.graph.add_res_alt(*col)
		self.var_alt.append(self.m.addVar(vtype=GRB.BINARY, name=f'a_{len(self.var_alt)}'))

		print(f'res alt - i: {len(self.var_alt) - 1}, nodes: {col[:2]}')

		return True


	def get_var_expr(self, vars):
		return [self.var_alt[i] if v else 1 - self.var_alt[i] for i, v in vars]

if __name__ == '__main__':
	data = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATA
	print(data)
	inst = Instance(data)
	# prepr = Preprocess(inst)

	pnc = Path_and_cycle(inst)
	pnc.set_paths()
	pnc.solve()
