#!.venv/bin/python3

import sys
import numpy as np
import gurobipy as gp

from array import array, ArrayType
from typing import List, Tuple
from numpy.typing import NDArray
from gurobipy import GRB

from instance import Instance, IDX_MAX, TIME_MAX
from preprocess import Preprocess


DEFAULT_DATA = 'data/nor1_critical_0.json'


STATE_WAIT = 0
STATE_ON_STACK = 1
STATE_DONE = 2

EDGE_MAX = 0xffffffff

class Graph:
	
	n_vertices: int
	
	# (idx, vertex, dur)
	edge_in: List[List[Tuple[int, int, int]]]
	edge_out: List[List[Tuple[int, int, int]]]

	state: NDArray[np.uint8]
	ord_idx: NDArray[np.uint8]
	path_vertex: NDArray[np.uint16]
	path_edge: NDArray[np.uint32]
	time: NDArray[np.uint32]
	order: ArrayType[int]

	next_edge_idx: int
	free_edge_idx: List[int]
	
	def __init__(self):
		self.order = array('I')
		self.next_edge_idx = 0
		self.free_edge_idx = array('L')


	def set_vertices(self, n):
		self.n_vertices = n
		self.edge_in = [[] for _ in range(n)]
		self.edge_out = [[] for _ in range(n)]

		self.state = np.empty(shape=(n, ), dtype=np.uint8)
		self.ord_idx = np.empty(shape=(n, ), dtype=np.uint8)
		self.path_vertex = np.empty(shape=(n, ), dtype=np.uint16)
		self.path_edge = np.empty(shape=(n, ), dtype=np.uint32)
		self.time = np.empty(shape=(n, ), dtype=np.uint32)


	def add_edge(self, v_from: int, v_to: int, dur: int):
		assert(v_from != v_to)
		edge_idx = self.get_free_edge_idx()
		self.edge_in[v_to].append((edge_idx, v_from, dur))
		return edge_idx


	def make_order(self, targets, valid):
		self.reset_search()
		for v in targets:
			ret = self.order_rec(v, valid)

			if ret < IDX_MAX:
				v = self.path_vertex[ret]
				cycle = [self.path_edge[ret]]

				while v != ret:
					cycle.append(self.path_edge[v])
					v = self.path_vertex[v]
				
				return cycle

		return None

	def make_paths(self, targets, lbs, valid):
		self.reset_search()
		self.ord_idx.fill(0)
		self.time[:] = lbs

		for v in targets:
			self.path_rec(v, valid)


	def order_rec(self, v: int, valid):
		if self.state[v] == STATE_DONE:
			return IDX_MAX
		
		if self.state[v] == STATE_ON_STACK:
			return v
		
		ret = IDX_MAX
		self.state[v] = STATE_ON_STACK

		for e, p, _ in reversed(self.edge_in[v]):
			if valid[e]:
				self.path_edge[p] = e
				self.path_vertex[p] = v
				
				ret = self.order_rec(p, valid)
				if ret < IDX_MAX:
					break

		self.state[v] = STATE_DONE
		self.order.append(v)
		return ret


	def path_rec(self, v: int, valid):
		if self.state[v] == STATE_DONE:
			return

		for e, p, d in self.edge_in[v]:
			if valid[e]:
				self.path_rec(p, valid)
				pred_time = self.time[p] + d

				if self.time[v] < pred_time:
					self.time[v] = pred_time
					self.path_edge[v] = e
					self.path_vertex[v] = p

					if d == 0:
						self.ord_idx[v] = self.ord_idx[p] + 1


		self.state[v] = STATE_DONE
		return


	def reset_search(self):
		self.state.fill(STATE_WAIT)
		self.path_vertex.fill(IDX_MAX)
		self.path_edge.fill(EDGE_MAX)


	def get_path_edges(self, v):
		edges = []
		while True:
			v_pred = self.path_vertex[v]
			if v_pred == IDX_MAX:
				break

			edges.append(self.path_edge[v])
			v = v_pred 

		return edges


	def get_free_edge_idx(self):
		if self.free_edge_idx:
			idx = self.free_edge_idx.pop()
		else:
			idx = self.next_edge_idx
			self.next_edge_idx += 1

		return idx


class Path_and_cycle:
	inst: Instance
	graph: Graph

	op_vtx: NDArray[np.uint16]
	vtx_in: NDArray[np.uint16]
	vtx_out: NDArray[np.uint16]
	vtx_lb: NDArray[np.uint32]

	v_end: ArrayType[int]
	edge_valid: NDArray[np.int8]
	n_path_edges: int
	edge_var: List[Tuple[int, bool]]

	objs: List[Tuple[gp.Var, int, int, bool]]
	alts: List[Tuple[gp.Var, List[int], List[int]]]

	def __init__(self, inst: Instance):
		self.inst = inst
		self.v_end = array('I')
		self.edge_valid = array('B')
		self.edge_var = []

		self.op_vtx = np.empty(shape=(inst.n_ops), dtype=np.uint16)

	def set_paths(self, paths=None):
		if paths is None:
			paths = [self.get_random_path(t) for t in range(self.inst.n_trains)]

		self.paths = paths


	def get_random_path(self, t: int):
		train = self.inst.trains[t]

		path = array('I')
		path.append(train.op_first)

		while path[-1] != train.op_last:
			op = self.inst.ops[path[-1]]
			s = int(np.random.choice(op.succ))
			path.append(s)
		
		return path


	def prep_graph(self):
		n_vtx = 0
		for path in self.paths:
			n_vtx += len(path) + 1
		
		self.graph = Graph()
		self.graph.set_vertices(n_vtx)

		self.vtx_in = np.full(shape=(n_vtx), fill_value=IDX_MAX, dtype=np.uint16)
		self.vtx_out = np.full(shape=(n_vtx), fill_value=IDX_MAX, dtype=np.uint16)
		self.vtx_lb = np.full(shape=(n_vtx), fill_value=0, dtype=np.uint32)

		while len(self.v_end) > 0:
			self.v_end.pop()

		v = 0
		for path in self.paths:
			self.vtx_in[v] = IDX_MAX

			for o in path:
				op = self.inst.ops[o]

				w = v + 1

				self.op_vtx[o] = v
				self.vtx_out[v] = o
				self.vtx_in[w] = o
				
				self.vtx_lb[v] = op.start_lb
				self.graph.add_edge(v, w, op.dur)

				v = w	

			last_op = self.inst.ops[path[-1]]
			self.vtx_lb[v] = last_op.start_lb + last_op.dur
			self.vtx_out[v] = IDX_MAX
			self.v_end.append(v)

			v += 1
		
		self.n_path_edges = self.graph.next_edge_idx
		self.edge_valid = np.zeros(shape=(self.n_path_edges,), dtype=np.uint8)
		self.edge_var.clear()
		for _ in range(self.n_path_edges):
			self.edge_var.append(None)
		
	
	def prep_model(self):
		self.m = gp.Model()

		self.alts = []
		self.objs = []

		path_durs = [sum(self.inst.ops[o].dur for o in path) for path in self.paths]
		total_dur = sum(path_durs)

		self.make_edge_valid()
		self.graph.make_order(self.v_end, self.edge_valid)
		self.graph.make_paths(self.v_end, self.vtx_lb, self.edge_valid)

		for i, path in enumerate(self.paths):
			for o in path:
				obj = self.inst.get_op_obj(o)
				if obj:
					is_binary = obj.increment > 0
					vtx = self.op_vtx[o]
					if is_binary:
						var = self.m.addVar(obj=obj.increment, vtype=GRB.BINARY, name=f'obj_{o}_bin')
						
					else:
						lb = max(self.graph.time[vtx] - obj.threshold, 0)
						ub = lb + total_dur - path_durs[i]
			
						var = self.m.addVar(lb=lb, ub=ub, obj=obj.coeff, vtype=GRB.CONTINUOUS, name=f'obj_{o}')
					
					self.objs.append((var, vtx, obj.threshold, is_binary))
		
		self.m.ModelSense = 1
		self.m.Params.OutputFlag = 0


	def solve(self):

		while True:
			self.m.update()
			self.m.optimize()
			assert(self.m.Status == GRB.OPTIMAL)

			self.make_edge_valid()
			cycle_edges = self.graph.make_order(self.v_end, self.edge_valid)
			if cycle_edges:
				self.add_cycle_cons(cycle_edges)
				continue

			self.graph.make_paths(self.v_end, self.vtx_lb, self.edge_valid)

			obj_delay = self.get_obj_delay()
			if obj_delay is not None:
				self.add_obj_delay_cons(obj_delay)
				continue

			res_col = self.get_res_col()
			if res_col:
				self.add_res_col(res_col)
				continue
				
			print('optimal sol')
			break

	
	def make_edge_valid(self):
		valid_idx = [i for i in range(self.n_path_edges)]

		for var, neg_edges, pos_edges in self.alts:
			if var.X > 0.5:
				valid_idx.extend(pos_edges)
			else:
				valid_idx.extend(neg_edges)

		self.edge_valid.fill(0)
		self.edge_valid[valid_idx] = 1
		pass
			

	def add_cycle_cons(self, cycle_edges):
		expr = self.get_expr_from_edge(cycle_edges)
		self.m.addConstr(sum(expr) <= len(expr) - 1)
		print('cycle const', expr)
		
	
	def get_obj_delay(self):
		diffs = []
		for i, (var, vtx, threshold, is_binary) in enumerate(self.objs):
			diff = self.graph.time[vtx] - threshold
			if is_binary:
				if var.X < 0.5 and diff > 0:
					diffs.append((diff, i))
			else:
				if int(round(var.X)) < diff:
					diffs.append((diff - var.X, i))

		if diffs:
			i_max = max(diffs)[1]
			return i_max
		
		return None


	def add_obj_delay_cons(self, obj_delay):
		var, vtx, threshold, is_binary = self.objs[obj_delay]
		
		path_edges = self.graph.get_path_edges(vtx)
		expr = self.get_expr_from_edge(path_edges)

		diff = self.graph.time[vtx] - threshold

		if is_binary:
			assert(var.X < 0.5 and diff > 0)
			cons = self.m.addConstr(sum(expr) - len(expr) + 1 <= var)
		else:
			assert(int(round(var.X)) < diff)
			cons = self.m.addConstr((sum(expr) - len(expr) + 1)*diff <= var)

		cons.Lazy = True
		
		print('delay cons', obj_delay)


	def get_expr_from_edge(self, edges):
		edge_vars = [self.edge_var[e] for e in edges if self.edge_var[e] is not None]
		return [self.alts[i][0] if v else 1 - self.alts[i][0] for i, v in edge_vars]			


	def get_res_col(self):
		vtx_time = [(self.graph.time[v], self.graph.ord_idx[v], v) for v in self.graph.order]
		vtx_time.sort()

		res_locks = [(IDX_MAX, IDX_MAX, TIME_MAX) for _ in range(self.inst.n_res)]

		for time, _, v in vtx_time:
			o_unlock = int(self.vtx_in[v])
			if o_unlock < IDX_MAX:
				op = self.inst.ops[o_unlock]
				for r, res_time in zip(op.res, op.res_time):
					assert(res_locks[r][0] == o_unlock)
					res_locks[r] = (o_unlock, op.train, time + res_time)

			o_lock = int(self.vtx_out[v])
			if o_lock < IDX_MAX:
				op = self.inst.ops[o_lock]
				for r in op.res:
					o_res, t_res, time_res = res_locks[r]
					if (o_res == IDX_MAX) or (t_res == op.train) or (time_res <= time):
						res_locks[r] = (o_lock, op.train, TIME_MAX)
					else:
						return (r, o_res, o_lock)
		
		return None


	def add_res_col(self, res_col):
		r, o1, o2 = res_col


		var = self.m.addVar(vtype=GRB.BINARY, name=f'alt_{o1}_{o2}')

		v1 = int(self.op_vtx[o1])
		v2 = int(self.op_vtx[o2])
		
		neg_edges = []
		for u, v in self.get_infered_edges((v1+1, v2)):
			neg_edges.append(self.graph.add_edge(u, v, 0))

		pos_edges = []
		for u, v in self.get_infered_edges((v2+1, v1)):
			pos_edges.append(self.graph.add_edge(u, v, 0))

		self.edge_valid.resize((self.graph.next_edge_idx,))

		var_idx = len(self.alts)
		self.alts.append((var, neg_edges, pos_edges))

		for e in neg_edges:
			assert(e <= len(self.edge_var))
			if e < len(self.edge_var):
				self.edge_var[e] = (var_idx, False)
			else:
				self.edge_var.append((var_idx, False))
		
		for e in pos_edges:
			assert(e <= len(self.edge_var))
			if e < len(self.edge_var):
				self.edge_var[e] = (var_idx, False)
			else:
				self.edge_var.append((var_idx, True))


		print('res col', (o1, o2))


	def get_infered_edges(self, edge):
		u, v = edge

		added = set()

		edges = []
		q = [edge]

		while q:
			u, v = q.pop()

			# opposite dir
			#    r1     r2
			# a ---> b ---> c
			# z <--- y <--- x
			#
			# edge b -> y implies c -> x

			found = True
			while found:
				added.add((u, v))

				o1 = self.vtx_out[u]
				o2 = self.vtx_in[v]

				found = False

				if o1 < IDX_MAX and o2 < IDX_MAX:
					op1 = self.inst.ops[o1]
					op2 = self.inst.ops[o2]

					for res1 in op1.res:
						for res2 in op2.res:
							if res1 == res2:
								found = True
								u += 1
								v -= 1
								break
						
						if found:
							break
			
			edges.append((u, v))

			# same dir
			#    r1     r2
			# a ---> b ---> c
			# x ---> y ---> z
			#
			# edge b -> x implies c -> y
			# edge c -> y implies b -> x

			# front case

			o1 = self.vtx_out[u]
			o2 = self.vtx_out[v+1]

			if o1 < IDX_MAX and o2 < IDX_MAX:
				op1 = self.inst.ops[o1]
				op2 = self.inst.ops[o2]

				found = False

				for res1 in op1.res:
					for res2 in op2.res:
						if res1 == res2:
							found = True
							eg = (u+1, v+1)
							if eg not in added:
								added.add(eg)
								q.append(eg)
							break
					
					if found:
						break
			
			# back case
			
			o1 = self.vtx_in[u-1]
			o2 = self.vtx_in[v]

			if o1 < IDX_MAX and o2 < IDX_MAX:
				op1 = self.inst.ops[o1]
				op2 = self.inst.ops[o2]

				found = False

				for res1 in op1.res:
					for res2 in op2.res:
						if res1 == res2:
							found = True
							eg = (u-1, v-1)
							if eg not in added:
								added.add(eg)
								q.append(eg)
							break
					
					if found:
						break
		
		return list(edges)


if __name__ == '__main__':
	data = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATA
	print(data)
	inst = Instance(data)
	# prepr = Preprocess(inst)

	np.random.seed(43)

	pnc = Path_and_cycle(inst)
	pnc.set_paths()
	pnc.solve()
