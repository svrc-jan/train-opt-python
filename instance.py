#!.venv/bin/python3

import sys
import json

from typing import List, Dict
from collections import defaultdict
from dataclasses import dataclass, field


DEFAULT_DATA = 'data/nor1_critical_2.json'

@dataclass
class Res:
	idx: int = -1
	time: int = 0


@dataclass
class Op:
	idx: int = -1
	train: int = -1

	dur: int = 0
	start_lb: int = 0
	start_ub: int|None = None

	succ: List[int] = field(default_factory=list)
	pred: List[int] = field(default_factory=list)
	res: List[Res] = field(default_factory=list)

	@property
	def n_succ(self):
		return len(self.succ)
	
	@property
	def n_pred(self):
		return len(self.pred)

	@property
	def n_res(self):
		return len(self.res)


@dataclass
class Obj:
	op_idx: int = -1
	threshold: int = 0
	coeff: int = 0
	increment: int = 0


@dataclass
class Train:
	idx: int = -1
	op_start: int = -1
	op_end: int = -1

	@property
	def n_ops(self):
		return self.op_end - self.op_start
	
	@property
	def op_last(self):
		return self.op_end - 1


class Instance:
	trains: List[Train]
	ops: List[Op]
	objs: List[Obj]
	res_name_idx: Dict[str, int]

	def __init__(self, jsn_file: str):
		self.parse_json_file(jsn_file)
		self.make_pred_ops()

		self.set_max_ub()
		self.propagate_lb()
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
		train.op_start = self.n_ops
		
		for jsn_op in jsn_train:
			self.parse_json_op(jsn_op, train)

		train.op_end = self.n_ops
		self.trains.append(train)

		# check if only last op is ending op (n_succ == 0), required for solver
		for op in self.ops[train.op_start:train.op_last]:
			assert(op.n_succ > 0)

		assert(self.ops[train.op_last].n_succ == 0)


	def parse_json_op(self, jsn_op: dict, train: Train):
		op = Op(
			idx 	=self.n_ops,
			train 	=train.idx,
			dur		=jsn_op['min_duration'],
			start_lb=jsn_op.get('start_lb', 0),
			start_ub=jsn_op.get('start_ub', None)
		)

		for s in jsn_op['successors']:
			op.succ.append(s + train.op_start)

		for jsn_res in jsn_op.get('resources', []):
			res_name = jsn_res['resource']
			res_time = jsn_res.get('release_time', 0)
			
			op.res.append(Res(idx=self.res_name_idx[res_name], time=res_time))

		self.ops.append(op)


	def parse_json_obj(self, jsn_obj):
		if jsn_obj['type'] != 'op_delay':
			return
		
		op_idx = self.trains[jsn_obj['train']].op_start + jsn_obj['operation']
		
		obj = Obj(
			op_idx		=op_idx,
			threshold	=jsn_obj.get('threshold', 0),
			coeff		=jsn_obj.get('coeff', 0),
			increment	=jsn_obj.get('increment', 0)
		)

		if obj.coeff == 0 and obj.increment == 0:
			return
		
		assert(obj.coeff == 0 or obj.increment == 0)
		self.objs.append(obj)


	def make_pred_ops(self):
		for op in self.ops:
			for s in op.succ:
				self.ops[s].pred.append(op.idx)


	def train_ops(self, t) -> List[Op]:
		train = self.trains[t]
		return self.ops[train.op_start:train.op_last]


	def set_max_ub(self):
		max_ub = sum(op.dur for op in self.ops) + \
			max(self.ops[train.op_start].start_lb for train in self.trains)
		
		for op in self.ops:
			if op.start_ub is None:
				op.start_ub = max_ub


	def propagate_lb(self):
		n_pred = [op.n_pred for op in self.ops]

		for train in self.trains:
			q = [train.op_start]

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

	@property
	def n_trains(self):
		return len(self.trains)
	
	@property
	def n_ops(self):
		return len(self.ops)

	@property
	def n_res(self):
		return len(self.res_name_idx)


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
	test_op_succ(inst)
	