from collections import defaultdict

class Disjoint_set:
	def __init__(self, n_items):
		self.n_items = n_items
		self.n_sets = n_items

		self.parent = list(range(n_items))
		self.size = [1] * n_items

	def find_set(self, v):
		while (v != self.parent[v]):
			self.parent[v] = self.parent[self.parent[v]]
			v = self.parent[v]
		
		return v
	
	def union_set(self, a, b):
		a = self.find_set(a)
		b = self.find_set(b)

		if (a != b):
			if self.size[a] < self.size[b]:
				a, b = b, a
			
			self.parent[b] = a
			self.size[a] += self.size[b]

			self.n_sets -= 1
	
	def get_result(self):
		idx_map = {}

		for v in range(self.n_items):
			if v == self.find_set(v):
				idx_map[v] = len(idx_map)
		
		set_idx = [idx_map[self.find_set(v)] for v in range(self.n_items)]

		return set_idx
