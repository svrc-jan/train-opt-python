#!.venv/bin/python3

import sys
import json

from typing import List, Dict
from collections import defaultdict
from dataclasses import dataclass, field
from array import array, ArrayType


DEFAULT_DATA = 'data/nor1_critical_2.json'

IDX_MAX = 0xffff
TIME_MAX = 0xffffffff

@dataclass(slots=True)
class Res:
	idx: int = IDX_MAX
	time: int = 0

@dataclass(slots=True)
class Op:
	idx: int = IDX_MAX
	train: int = IDX_MAX

	dur: int = 0
	start_lb: int = 0
	start_ub: int = TIME_MAX

	succ: ArrayType[int] = field(default_factory=lambda: array('I'))
	pred: ArrayType[int] = field(default_factory=lambda: array('I'))
	res: ArrayType[int] = field(default_factory=lambda: array('I'))
	res_time: ArrayType[int] = field(default_factory=lambda: array('I'))

	obj: int = IDX_MAX

	@property
	def n_succ(self):
		return len(self.succ)
	
	@property
	def n_pred(self):
		return len(self.pred)

	@property
	def n_res(self):
		return len(self.res)


@dataclass(slots=True)
class Obj:
	op_idx: int = IDX_MAX
	threshold: int = 0
	coeff: int = 0
	increment: int = 0


@dataclass(slots=True)
class Train:
	idx: int = IDX_MAX
	op_first: int = IDX_MAX
	op_after: int = IDX_MAX

	inst_ops: List[Op] = None

	@property
	def n_ops(self):
		return self.op_after - self.op_first
	
	@property
	def op_last(self):
		return self.op_after - 1
	
	@property
	def ops(self) -> List[Op]:
		return self.inst_ops[self.op_first:self.op_after]

	@property
	def op_range(self):
		return range(self.op_first, self.op_after)


class Instance:
	trains: List[Train]
	ops: List[Op]
	objs: List[Obj]
	res_name_idx: Dict[str, int]

	def __init__(self, jsn_file: str):
		self.parse_json_file(jsn_file)
		self.make_pred_ops()

		self.propagate_lb()
		self.set_max_ub()
		self.propagete_ub()
		self.test_bounds()

		print(f'Instance - trains: {self.n_trains}, ops: {self.n_ops}, res: {self.n_res}')


	def parse_json_file(self, jsn_file: str):
		self.trains = []
		self.ops = []
		self.objs = []

		self.res_name_idx = defaultdict(lambda: self.n_res)

		with open(jsn_file, 'r') as fd:
			jsn = json.load(fd)
		
		for jsn_train in jsn['trains']:
			self.parse_json_train(jsn_train)

		for jsn_obj in jsn['objective']:
			self.parse_json_obj(jsn_obj)


	def parse_json_train(self, jsn_train: dict):
		train = Train(idx=self.n_trains)
		train.inst_ops = self.ops

		train.op_first = self.n_ops
		
		for jsn_op in jsn_train:
			self.parse_json_op(jsn_op, train)

		train.op_after = self.n_ops
		self.trains.append(train)

		# check if only last op is ending op (n_succ == 0), required for solver
		for op in self.ops[train.op_first:train.op_last]:
			assert(op.n_succ > 0)

		assert(self.ops[train.op_last].n_succ == 0)


	def parse_json_op(self, jsn_op: dict, train: Train):
		op = Op(
			idx 	=self.n_ops,
			train 	=train.idx,
			dur		=jsn_op['min_duration'],
			start_lb=jsn_op.get('start_lb', 0),
			start_ub=jsn_op.get('start_ub', TIME_MAX)
		)

		for s in jsn_op['successors']:
			op.succ.append(s + train.op_first)

		for jsn_res in jsn_op.get('resources', []):
			res_name = jsn_res['resource']
			res_time = jsn_res.get('release_time', 0)

			res_idx = self.res_name_idx[res_name]
			
			try:
				i = op.res.index(res_idx)
				op.res_time[i] = max(op.res_time[i], res_time)
			except ValueError:			
				op.res.append(res_idx)
				op.res_time.append(res_time)

		self.ops.append(op)


	def parse_json_obj(self, jsn_obj):
		if jsn_obj['type'] != 'op_delay':
			return
		
		op_idx = self.trains[jsn_obj['train']].op_first + jsn_obj['operation']
		
		obj = Obj(
			op_idx		=op_idx,
			threshold	=jsn_obj.get('threshold', 0),
			coeff		=jsn_obj.get('coeff', 0),
			increment	=jsn_obj.get('increment', 0)
		)

		if obj.coeff == 0 and obj.increment == 0:
			return
		
		assert(obj.coeff == 0 or obj.increment == 0)
		self.ops[op_idx].obj = self.n_objs
		self.objs.append(obj)


	def make_pred_ops(self):
		for op in self.ops:
			for s in op.succ:
				succ = self.ops[s]
				succ.pred.append(op.idx)


	def train_ops(self, t: int) -> List[Op]:
		train = self.trains[t]
		return self.ops[train.op_first:train.op_after]


	def set_max_ub(self):
		n_succ = [op.n_succ for op in self.ops]
		dist = [0]*self.n_ops

		for train in self.trains:
			q = [train.op_last]

			while q:
				o = q.pop(0)
				op = self.ops[o]

				for p in op.pred:
					dist[p] = max(dist[p], dist[o] + self.ops[p].dur)
					n_succ[p] -= 1
					if n_succ[p] == 0:
						q.append(p)

		train_dur = [dist[train.op_first] for train in self.trains]
		total_dur = sum(train_dur)

		max_ub = 0
		for op in self.ops:
			max_ub = max(max_ub, op.start_lb + dist[o] + total_dur - train_dur[op.train])

		for op in self.ops:
			if op.start_ub == TIME_MAX:
				op.start_ub = max_ub


	def propagate_lb(self):
		n_pred = [op.n_pred for op in self.ops]

		for train in self.trains:
			q = [train.op_first]

			while q:
				o = q.pop(0)
				op = self.ops[o]
				
				pred_ops = [self.ops[p] for p in op.pred]
				pred_bounds = [pred.start_lb + pred.dur for pred in pred_ops]

				if pred_bounds:
					op.start_lb = max(op.start_lb, min(pred_bounds))

				for s in op.succ:
					n_pred[s] -= 1
					if n_pred[s] == 0:
						q.append(s)


	def propagete_ub(self):
		n_succ = [op.n_succ for op in self.ops]

		for train in self.trains:
			q = [train.op_last]

			while q:
				o = q.pop(0)
				op = self.ops[o]
				
				succ_ops = [self.ops[p] for p in op.succ]
				succ_bounds = [succ.start_ub - op.dur for succ in succ_ops]

				if succ_bounds:
					op.start_ub = min(op.start_ub, max(succ_bounds))

				for p in op.pred:
					n_succ[p] -= 1
					if n_succ[p] == 0:
						q.append(p)


	def test_bounds(self):
		for op in self.ops:
			assert(op.start_lb <= op.start_ub)


	def get_op_obj(self, o: int):
		op = self.ops[o]
		if op.obj < IDX_MAX:
			return self.objs[op.obj]
		return None


	@property
	def n_trains(self):
		return len(self.trains)
	
	@property
	def n_ops(self):
		return len(self.ops)

	@property
	def n_res(self):
		return len(self.res_name_idx)
	
	@property
	def n_objs(self):
		return len(self.objs)


def test_op_succ(inst: Instance):
	for op in inst.ops:
		for s in op.succ:
			succ = inst.ops[s]
			assert(op.idx in succ.pred)

		for p in op.pred:
			pred = inst.ops[p]
			assert(op.idx in pred.succ)


if __name__ == '__main__':
	data = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATA
	print(data)
	inst = Instance(data)
	
	res_uses = [[] for _ in range(inst.n_res)]

	for op in inst.ops:
		for res in op.res:
			res_uses[res.idx].append((op.train, op.idx))


	edges = [[[] for _ in range(inst.n_trains)] for _ in range(inst.n_trains)]
	
	count = 0
	for r, ru in enumerate(res_uses):
		for t1, o1 in ru:
			for t2, o2 in ru:
				if t1 != t2:
					edges[t1][t2].append((o1, o2))
					count += 1

	

	print(f'edge count: {count}')


