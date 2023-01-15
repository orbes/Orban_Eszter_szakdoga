from network.Pair import Pair


class PacketA:
    def __init__(self, id_, start_, target_, birth_time):
        self.id = 'A' + str(id_) + "#" + str(start_.id) + "#" + str(target_)
        self.start = start_
        self.target = target_
        self.type = 'A'
        self.route = [[start_.id, 0]]
        self.adj_matrix = []
        self.pref_nodes = []
        self.birth_time = birth_time
        self.life_time = 0
        self.distance = 0                   #milyen messze van a céljától

    def __add__(self, other):
        if isinstance(other, PacketA):
            return len(self.route) + len(other.route)
        else:
            return 0

    def __radd__(self, other):
        return self + other

    def iterate_pref(self):
        for i in self.pref_nodes:
            yield i

    def iterate_route(self):
        for i in self.route:
            yield i[0]

    def add_new_node(self, actual_node):
        tmp = Pair(self.route[-1][0], actual_node.id)
        if tmp not in self.adj_matrix:
            self.adj_matrix.append(tmp)
        self.route.append([actual_node.id, 0])

    def add_pref_nodes(self, node):
        self.pref_nodes.append(node)

    def arrived(self, die_time):
        self.life_time = die_time-self.birth_time
        return self.life_time

    def info(self):
        return 8 + len(self.route)*8*2 + len(self.pref_nodes)*8   # szomszédsági mátrix hossza *2 (1 pairban 2 node)
                                                                                    # +1 a cél node idja
                                                                                    # pereferált nodok


