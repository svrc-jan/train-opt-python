#!.venv/bin/python3

import sys

import glpk
import numpy as np
import gurobipy as gp

from heapq import heappush, heappop, heapreplace
from typing import List, Dict
from instance import Instance
from preprocess import Preprocess


MAX_DUR = 100000
DEFAULT_DATA = 'data/nor1_critical_0.json'

class Path_selection:
	inst: Instance
	prepr: Preprocess

	def __init__(self, prepr: Preprocess):
		self.inst = prepr.inst
		self.prepr = prepr


	def get_op_importance(self):
		forw_imp = [0.0]*self.inst.n_ops
		backw_imp = [0.0]*self.inst.n_ops

		n_pred = [op.n_pred for op in self.inst.ops]
		n_succ = [op.n_succ for op in self.inst.ops]

		for train in self.inst.trains:
			forw_imp[train.op_start] = 1.0
			q = [train.op_start]

			while q:
				o = q.pop(0)
				op = self.inst.ops[o]
				imp = forw_imp[o]
				for s in op.succ:
					forw_imp[s] += imp/op.n_succ

					n_pred[s] -= 1
					if n_pred[s] == 0:
						q.append(s)


			backw_imp[train.op_last] = 1.0
			q = [train.op_last]

			while q:
				o = q.pop(0)
				op = self.inst.ops[o]
				imp = backw_imp[o]
				for p in op.pred:
					backw_imp[p] += imp/op.n_pred

					n_succ[p] -= 1
					if n_succ[p] == 0:
						q.append(p)

		op_imp = [(f + b)/2 for f, b in zip(forw_imp, backw_imp)]		
		
		return op_imp

	def get_op_in_path(self, paths):
		op_imp = [0.0]*self.inst.n_ops

		for path in paths:
			for o in path:
				op_imp[o] += 1

		return op_imp

	def get_res_importance(self, op_imp):
		res_imp = [0.0]*self.inst.n_res

		for op in self.inst.ops:
			for res in op.res:
				res_imp[res.idx] += op_imp[op.idx]

		return res_imp
	
	
	def get_op_res_cost(self, res_imp, op_imp):
		cost = [sum((res_imp[res.idx] - op_imp[op.idx]) for res in op.res) for op in self.inst.ops]

		return cost
	
	
	def select_path(self, train, cost: List[float]):
		if isinstance(train, int):
			train = self.inst.trains[train]
		
		pred = {train.op_start : None}
		dist = {train.op_start : 0.0}

		q = [(0.0, train.op_start)]

		while q:
			d, o = q[0]

			if d > dist[o]:
				heappop(q)
				continue

			op = self.inst.ops[o]
			
			d_succ = d + cost[o]

			pushed = False
			for s in op.succ:
				if s not in dist or d_succ < dist[s]:
					dist[s] = d_succ
					pred[s] = o

					if not pushed:
						heapreplace(q, (d_succ, s))
						pushed = True
					else:
						heappush(q, (d_succ, s))

			if not pushed:
				heappop(q)

		path = [train.op_last]

		while path[0] > train.op_start:
			path.insert(0, pred[path[0]])

		return path


	def select_all_paths(self, cost):
		return [self.select_path(train, cost) for train in self.inst.trains]


	def select_non_overlap_paths(self):
		op_imp = self.get_op_importance()
		res_imp = self.get_res_importance(op_imp)

		cost = self.get_op_res_cost(res_imp, op_imp)
		paths = self.select_all_paths(cost)
		
		return paths
	

	def select_iqp_paths(self):
		self.iqp_model = gp.Model()
		self.iqp_model.setParam('OutputFlag', 0)

		self.make_iqp_vars()
		self.make_iqp_flow_cons()
		self.make_iqp_res_use_cons()
		self.make_iqp_obj_res()

		self.iqp_model.optimize()

		self.make_iqp_res_use_cons()
		self.make_iqp_obj_dur()

		self.iqp_model.optimize()

		paths = self.get_iqp_paths()
		return paths

	def make_iqp_vars(self):
		m = self.iqp_model
		self.var_op = m.addVars(list(range(self.inst.n_ops)), vtype=gp.GRB.BINARY)

		self.var_res_uses = m.addVars(list(range(self.inst.n_res)))
	
	
	def make_iqp_flow_cons(self):
		m = self.iqp_model
		
		for level in self.prepr.levels:
			in_sum = gp.quicksum(self.var_op[o] for o in level.ops_in)
			out_sum = gp.quicksum(self.var_op[o] for o in level.ops_out)
			
			if level.n_op_in == 0:
				m.addConstr(out_sum == 1)
			elif level.n_op_out == 0:
				m.addConstr(in_sum == 1)
			else:
				m.addConstr(in_sum == out_sum)


	def make_iqp_res_use_cons(self):
		res_ops = [[] for _ in range(self.inst.n_res)]

		for op in self.inst.ops:
			for res in op.res:
				res_ops[res.idx].append(op.idx)

		m = self.iqp_model
		for r in range(self.inst.n_res):
			ops_sum = gp.quicksum(self.var_op[o] for o in res_ops[r])
			m.addConstr(self.var_res_uses[r] == ops_sum)


	def make_iqp_obj_res(self):
		m = self.iqp_model
		ru = self.var_res_uses
		m.setObjective(sum(ru[r]*ru[r] for r in range(self.inst.n_res)))
		

	def make_res_use_cons(self):
		m = self.iqp_model
		ru = self.var_res_uses
		for r in range(self.inst.n_res):
			m.addConstr(ru[r] <= ru[r].X)


	def make_iqp_obj_dur(self):
		m = self.iqp_model
		m.setObjective(gp.quicksum(self.var_op[op.idx]*op.dur for op in self.inst.ops))


	def get_iqp_paths(self):
		paths = []
		for train in self.inst.trains:
			path = [train.op_start]

			while path[-1] < train.op_last:
				o = path[-1]
				selected = 0
				for s in self.inst.ops[o].succ:
					if self.var_op[s].X > 0.5:
						path.append(s)
						selected += 1

				assert(selected == 1)
			
			paths.append(path)

		return paths


if __name__ == '__main__':
	data = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATA
	print(data)
	inst = Instance(data)
	prepr = Preprocess(inst)
	path_sel = Path_selection(prepr)
	# paths = path_sel.select_non_overlap_paths()

	paths = path_sel.select_iqp_paths()

	for p in paths:
		print(p)
