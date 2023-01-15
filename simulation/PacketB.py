from simulation.PacketA import PacketA
from network.Pair import Pair


class PacketB(PacketA):
    def __init__(self, id_, start_, target_, birth_time):
        super().__init__(id_, start_, target_, birth_time)
        self.id = 'B' + str(id_) + "#" + str(start_.id) + "#" + str(target_)
        self.start = start_
        self.target = target_
        self.type = 'B'
        self.adj_matrix = []
        self.route = [[self.start.id, 0]]
        self.pref_nodes = []
        self.birth_time = birth_time

    def add_new_node(self, actual_node):
        tmp = Pair(self.route[-1][0], actual_node.id)
        if tmp not in self.adj_matrix:
            self.adj_matrix.append(tmp)
        self.route.append([actual_node.id, 0])
        for i in actual_node.neighbours:
            tmp = Pair(actual_node.id, i.id)
            if tmp not in self.adj_matrix:
                self.adj_matrix.append(tmp)

    def info(self):
        return len(self.adj_matrix)*2*8 + 8 + len(self.pref_nodes)*8 + len(self.route)*9