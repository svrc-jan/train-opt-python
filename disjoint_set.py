
from typing import List
from array import array

class Disjoint_set:
	def __init__(self, n_items: int):
		self.n_items = n_items
		self.n_sets = n_items

		self.parent = array('L')
		self.size = array('L')
		
		for i in range(n_items):
			self.parent.append(i)
			self.size.append(1)


	def find_set(self, v: int) -> int:
		while (v != self.parent[v]):
			self.parent[v] = self.parent[self.parent[v]]
			v = self.parent[v]
		
		return v


	def union_set(self, a: int, b: int) -> int:
		a = self.find_set(a)
		b = self.find_set(b)

		if (a != b):
			if self.size[a] < self.size[b]:
				a, b = b, a
			
			self.parent[b] = a
			self.size[a] += self.size[b]

			self.n_sets -= 1


	def get_result(self, sort_lowest=False) -> List[int]:
		idx_map = {}

		for v in range(self.n_items):
			if v == self.find_set(v):
				idx_map[v] = len(idx_map)
		
		set_idx = array('L')
		for v in range(self.n_items):
			set_idx.append(idx_map[self.find_set(v)])

		if sort_lowest:
			lowest = [float('inf') for _ in range(self.n_sets)]
			for i, s in enumerate(set_idx):
				lowest[s] = min(lowest[s], i)

			order = list(range(self.n_sets))
			order.sort(key=lambda x: lowest[x])

			mp = [0]*self.n_sets

			for i, x in enumerate(order):
				mp[x] = i

			for i in range(self.n_items):
				set_idx[i] = mp[set_idx[i]]

		return set_idx
