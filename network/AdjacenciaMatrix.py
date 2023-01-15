from network.Pair import Pair


class AdjacenciaMatrix:
    def __init__(self, int_node):
        self.int_node = int_node
        self.compact_matrix = []                                            # [pair, pair, ...]
        for i in int_node.neighbours:
            tmp = Pair(int_node.id, i.id)
            if tmp not in self.compact_matrix:
                self.compact_matrix.append(tmp)

    def add_packet_information(self, packet):
        for pair in packet.adj_matrix:
            if pair not in self.compact_matrix:
                self.compact_matrix.append(pair)

    def get_neighbours(self, node_id):
        neighbours_list = []
        for pair in self.compact_matrix:
            if pair.start == node_id:
                neighbours_list.append(pair.destination)
            if pair.destination == node_id:
                neighbours_list.append(pair.start)
        return neighbours_list

    def diff(self):
        return len(self.compact_matrix)

    def clear(self):
        self.compact_matrix = []
        for i in self.int_node.neighbours:
            tmp = Pair(self.int_node.id, i.id)
            if tmp not in self.compact_matrix:
                self.compact_matrix.append(tmp)
