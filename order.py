#!.venv/bin/python3

import sys
import os

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, NamedTuple
from instance import Instance
from path_select import Path_selection

DEFAULT_DATA = 'data/nor1_critical_0.json'

@dataclass
class Event:
	train: int
	op_in: int 
	op_out: int


@dataclass
class Item:
	train: int = -1
	idx: int = -1


@dataclass
class Res_lock:
	item: Item = field(default_factory=Item)
	count: int = 0



class Order:
	inst: Instance
	paths: List[List[int]]

	def __init__(self, inst: Instance):
		self.inst = inst
		self.paths = [[]]*self.inst.n_trains
		
		self.pred = [-1]*self.inst.n_ops
		self.succ = [-1]*self.inst.n_ops
		self.lbs = [op.start_lb for op in self.inst.ops]


	def set_path(self, t, path):
		self.paths[t] = path
		
		path_len = len(path)
		for i in range(path_len):
			o = path[i]
			p = path[i-1] if i > 0 else -1
			s = path[i+1] if i < path_len - 1 else -1

			self.pred[o] = p
			self.succ[o] = s


	def set_all_paths(self, paths):
		for t, path in enumerate(paths):
			self.set_path(t, path)


	def get_init_order(self):
		order = []

		for t, path in enumerate(self.paths):
			for o in path:
				order.append(Event(t, self.pred[o], o))

		order.sort(key=lambda e: self.lbs[e.op_out])

		return order
	

	def process(self, order: List[Event]) -> Tuple[Item, Item]:
		res_locks = [Res_lock()]*self.inst.n_res

		first_col = None
		n_col = 0
		for i, e in enumerate(order):
			if e.op_in >= 0:
				for res in self.inst.ops[e.op_in].res:
					res_locks[res.idx].count -= 1

			for res in self.inst.ops[e.op_out].res:
				if res_locks[res.idx].count > 0:
					n_col += res_locks[res.idx].count

					if first_col is None:
						first_col = (res_locks[res.idx].item, Item(e.train, i))
				
				res_locks[res.idx].count += 1
				res_locks[res.idx].item = Item(e.train, i)

		return first_col, n_col
	

	def process_times(self, order: List[Event]):
		
		time = [0.0]*len(order)
		train_release = [0.0]*self.inst.n_trains
		res_release = [0.0]*self.inst.n_res

		for i, e in enumerate(order):
			op = self.inst.ops[e.op_out]
			t = max(op.start_lb, train_release[e.train])

			for res in op.res:
				t = max(t, res_release[res.idx])

			time[i] = t
			train_release[e.train] = t + op.dur
			
			if e.op_in >= 0:
				for res in self.inst.ops[e.op_in].res:
					res_release[res.idx] = t + res.time

		
		time_order = list(zip(time, order))
		time_order.sort(key=lambda x: x[0])

		return time_order


	def push(self, order: List[Event], items: Tuple[Item, Item]) -> List[Event]:
		i1, i2 = items

		reord = [x for x in order[i1.idx:i2.idx+1] if x.train == i2.train] \
			+ [x for x in order[i1.idx:i2.idx+1] if x.train != i2.train]
		
		return order[:i1.idx] + reord + order[i2.idx+1:]


	def find_unlock(self, order: List[Event], item: Item) -> Item:
		idx_unlock = next(i for i in range(item.idx, len(order)) if order[item.idx].op_out == order[i].op_in)
		return Item(item.train, idx_unlock)
	
	def find_lock(self, order: List[Event], item: Item) -> Item:
		idx_unlock = next(i for i in range(item.idx, 0, -1) if order[item.idx].op_in == order[i].op_out)
		return Item(item.train, idx_unlock)
	

	def solve(self, order):
		col, n_col = self.process(order)

		it = 0
		while n_col > 0:
			it += 1

			print(f'iter {it}, idx {col[1].idx}/{len(order)}')


			order = self.push(order, (col[1], self.find_unlock(order, col[0])))
			col, n_col = self.process(order)

		return order


if __name__ == '__main__':
	data = os.environ.get('TRAIN_OPT_INST', DEFAULT_DATA)	
	data = sys.argv[1] if len(sys.argv) > 1 else data

	print(data)
	inst = Instance(data)
	path_sel = Path_selection(inst)
	ord = Order(inst)

	cost = path_sel.get_op_res_cost()
	paths = path_sel.select_all_paths(cost)

	ord.set_all_paths(paths)
	order = ord.get_init_order()

	
	order = ord.solve(order)
	print(order)

	to = ord.process_times(order)
	
	print([x[1].train for x in  to])
