#!.venv/bin/python3

import sys

from hexaly.optimizer import HexalyOptimizer, HxModel


from instance import Instance
from preprocess import Preprocess


DEFAULT_DATA = 'data/nor1_critical_0.json'


class Hexaly_solver:
	inst: Instance
	prepr: Preprocess

	def __init__(self, inst, prepr):
		self.inst = inst
		self.prepr = prepr

	def solve(self):
		with HexalyOptimizer() as optimizer:
			optimizer.param.verbosity = 0

			m = optimizer.model

			self.make_op_vars(m)
			self.make_time_vars(m)

			self.make_flow_cons(m)
			self.make_time_cons(m)

			self.make_time_sum_obj(m)

			it = 0
			while True:
				m.close()
				optimizer.solve()

				res_col = self.get_res_col()
				if res_col is None:
					print('solution found')
					break

				print(f'it: {it}, res_col: {res_col}')
				
				m.open()
				self.add_res_col_cons(res_col, m)

				it += 1
				

	def make_time_vars(self, m: HxModel):
		self.var_time = [m.int(level.time_lb, level.time_ub) for level in self.prepr.levels]

	
	def make_op_vars(self, m: HxModel):
		self.var_op = [m.bool() for _ in range(self.inst.n_ops)]


	def make_flow_cons(self, m: HxModel):
		for junct in self.prepr.juncts:
			lhs = 1 if junct.n_op_in == 0 else m.sum(*[self.var_op[o] for o in junct.ops_in])
			rhs = 1 if junct.n_op_out == 0 else m.sum(*[self.var_op[o] for o in junct.ops_out])
			m.constraint(lhs == rhs)

		
	def make_time_cons(self, m: HxModel):
		for o, op in enumerate(self.inst.ops):
			m.constraint(self.op_time_start(o) + op.dur*self.var_op[o] <= self.op_time_end(o))


	def make_time_sum_obj(self, m: HxModel):
		obj = m.sum(self.var_time[train.level_last] for train in self.prepr.trains)
		m.minimize(obj)

	
	def get_res_col(self):
		res_uses = [list() for _ in range(self.inst.n_res)]

		for o, op in enumerate(self.inst.ops):
			if self.var_op[o].value == 1:

				t_start = self.op_time_start(o).value
				t_end = self.op_time_end(o).value
				
				for res in op.res:
					rt = max(res.time, 1)
					res_uses[res.idx].append((t_start, t_end + rt, o, rt))

		res_col_start = float('inf') 
		res_col = None

		for ru in res_uses:
			ru.sort()
			
			for i in range(len(ru) - 1):
				start_a, end_a, o_a, rt_a = ru[i]
				start_b, end_b, o_b, rt_b = ru[i+1]

				if start_b > res_col_start:
					break

				if start_b < end_a:
					res_col_start = start_b
					res_col = (o_a, o_b, rt_a, rt_b)
		
		return res_col
	

	def add_res_col_cons(self, res_col, m: HxModel):
		o_a, o_b, rt_a, rt_b = res_col

		cons1 = self.op_time_end(o_a) + rt_a <= self.op_time_start(o_b)
		cons2 = self.op_time_end(o_b) + rt_b <= self.op_time_start(o_a)

		m.constraint(m.or_(cons1, cons2, m.not_(self.var_op[o_a]), m.not_(self.var_op[o_b])))


	def op_time_start(self, o: int):
		return self.var_time[self.prepr.ops[o].level_start]
	

	def op_time_end(self, o: int):
		return self.var_time[self.prepr.ops[o].level_end]


if __name__ == '__main__':
	data = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATA
	print(data)
	inst = Instance(data)
	prepr = Preprocess(inst)

	hxl = Hexaly_solver(inst, prepr)
	hxl.solve()
