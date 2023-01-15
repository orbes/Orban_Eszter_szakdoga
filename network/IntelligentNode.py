from network.Node import Node
from network.AdjacenciaMatrix import AdjacenciaMatrix
from simulation.Logs import Logs
from random import random


class IntelligentNode(Node):
    def __init__(self, id_, n, packet_type, neighbours):
        super().__init__(id_, n, packet_type)
        self.id = id_
        self.neighbours = neighbours
        self.packet_type = packet_type
        self.log_packet = Logs("packets.txt")
        self.make_log_file()
        self.adj_matrix = AdjacenciaMatrix(self)

    def make_log_file(self):
        self.log_node.log("En egy intelligens node vagyok " + str(self.id) + "\n")
        self.log_node.log("szomszedok: " + ' '.join(str(x.id) for x in self.neighbours) + "\n")
        # self.log_node.log("szomszedsagi matrix: " + ''.join(str(x) for x in self.neighbours_matrix))

    def accept_packet(self, packet):
        packet.add_new_node(self)
        if packet.target != self.id:
            self.packet_list.add(packet)
        else:
            self.log_node.log("Csomag celba ert: " + packet.id + ", utvonala:  " +
                              " ".join(str(x[0]) for x in packet.route) + " preferalt utvonala: " +
                              " ".join(str(x) for x in packet.pref_nodes) + "\n")

    # Tovabbitas lepese: Meg keresi a kovetkezo csomopontott ahova tovabbitani fog.
    # - return -1 ha az input és output list is ures
    # - return -2 ha izolalt pont
    # - return -4 ha az output list ures
    # - ellenorizzuk, hogy a szomszadok kozt van-e a csomag celja
    # - ellenorizzuk, hogy a szomszedok kozt van-e a csomag preferalt nodjai kozt egy vagy tobb
    # - ellenorizzuk, hogy vezet-e cel a csomag celjaba
    #    - ha igen: BFS
    #    - ha nem: hagyomanyos uton keressunk kovetkezo pontot
    def get_next_node_id(self, alpha):
        if len(self.packet_list.input) == 0 and len(self.packet_list.output) == 0:
            return -1
        if len(self.neighbours) == 0:
            return -2
        if len(self.packet_list.output) == 0:
            return -4
        packet = self.packet_list.get_first()
        target = packet.target
        self.adj_matrix.add_packet_information(packet)
        # ha a cél a szomszédok közt van oda továbbítjuk
        for node in self.neighbours:
            if node.id == target:
                return node.id
        # megnézni van-e a szomszédok közt a csomagnak preferált nodja, ha igen akkor azzok irányába továbbítjuk
        alt_node = []
        for pref_node in packet.iterate_pref():
            for node in self.iterate_neighbours():
                if pref_node == node.id:
                    alt_node.append(node)
        if len(alt_node) == 1:
            return alt_node[0].id
        if len(alt_node) != 0:
            for i in range(len(alt_node) - 1):
                send_probability = random()
                probability = pow(alt_node[i].degree(), alpha) / self.get_neighbours_degree(alpha)
                if send_probability < probability:
                    return alt_node[i].id
            return alt_node[-1].id
        # BFS algoritmus segítségével a legrövidebb út megkeresése
        path = self.BFS(packet)
        # ha nincs ilyen akkor a "hagyományos" úton generálunk szomszédot...
        if len(path) == 0:
            for i in range(len(self.neighbours)-1):
                send_probability = random()
                probability = pow(self.neighbours[i].degree(), alpha) / self.get_neighbours_degree(alpha)
                if send_probability < probability:
                    return self.neighbours[i].id
            return self.neighbours[-1].id
        else:
            # BFS algoritmus eredményét betöltjük a preferalt nodok mezobe
            packet.pref_nodes = path
            return path[1]

    #  https://www.geeksforgeeks.org/building-an-undirected-graph-and-finding-shortest-path-using-dictionaries-in-python/
    def BFS(self, packet):
        target = packet.target
        explored = []
        queue = [[self.id]]
        while queue:
            path = queue.pop(0)
            node = path[-1]
            if node not in explored:
                neighbours = self.adj_matrix.get_neighbours(node)
                for neighbour in neighbours:
                    new_path = list(path)
                    new_path.append(neighbour)
                    queue.append(new_path)
                    if neighbour == target:
                        return new_path
                explored.append(node)
        return []

    def clear(self):
        self.adj_matrix.clear()

    def diff(self):
        return self.adj_matrix.diff()