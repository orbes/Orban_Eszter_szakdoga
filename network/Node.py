from network.Queue import Queue
from simulation.PacketA import PacketA
from simulation.PacketB import PacketB
from random import randrange, random
from simulation.Logs import Logs


class Node:
    def __init__(self, id_, n, packet_type):
        self.n = n
        self.id = id_
        self.packet_type = packet_type
        self.neighbours = []
        self.packet_list = Queue()
        self.counter = 0
        self.log_node = Logs("node" + str(self.id) + ".txt", "./logs/nodes")
        self.centrality = 0

    def __repr__(self):
        return str(self.id) + " " + str(self.centrality) + " " + str(self.degree()) + "\n"

    def close(self):
        self.log_node.close()

    def make_log_file(self):
        self.log_node.log("En egy " + self.packet_type + " tipusu node vagyok " + str(self.id) + "\n")
        self.log_node.log("szomszedaim: " + ' '.join(str(x.id) for x in self.neighbours) + "\n")

    # szomszédot ad hozzó
    def add_new_neighbour(self, node):
        self.neighbours.append(node)

    def add_new_neighbours(self, neighbours_list):
        self.neighbours = neighbours_list

    # szomszédos node-ok id-ja str-ben
    def get_neighbours(self):
        return ' '.join(str(x.id) for x in self.neighbours)

    # adott csúcs szomszédainak fokszámosszege
    def get_neighbours_degree(self, alpha):
        return sum(pow(i.degree(), alpha) for i in self.neighbours)

    # adott csúcs fokszáma
    def degree(self):
        return len(self.neighbours)

    def neighbours_id_list(self):
        tmp = []
        for i in self.neighbours:
            tmp.append(i.id)
        return tmp

    # új csomag generálása véletlen célla, ha a cél megegyezik az aktuális célal, új cél
    def generate_new_packet(self, time):
        target = randrange(self.n)
        while target == self.id:
            target = randrange(self.n)
        if self.packet_type == 'A':
            packet = PacketA(self.counter, self, target, time)
        else:
            packet = PacketB(self.counter, self, target, time)
        self.packet_list.add(packet)
        self.counter += 1
        return packet

    # csomag fogadása, ha nem a cél node sorhoz adjuk, ha igen akkor logoljuk...
    def accept_packet(self, packet):
        packet.add_new_node(self)
        if packet.target != self.id:
            self.packet_list.add(packet)
        else:
            self.log_node.log("Csomag celba ert: " + packet.id + ", utvonala:  " +
                              " ".join(str(x) for x in packet.route) + " preferalt utvonala: " +
                              " ".join(str(x) for x in packet.pref_nodes) + "\n")

    # input list átírása az output list végéhez
    def copy(self):
        self.packet_list.copy_input_to_output()

    # sor törlése
    def clear_queue(self):
        self.packet_list.clear()

    # visszaadja az output list első elemét, és kitörli onnan
    def get_first_packet(self):
        return self.packet_list.remove_first()

    # vissza adja a következő node id-ját
    # -1 ha nincs a sorban következő elem
    # -2 ha nincs szomszéd
    # -4 ha az output list üres
    def get_next_node_id(self, alpha):
        if len(self.packet_list.input) == 0 and len(self.packet_list.output) == 0:
            return -1
        if len(self.neighbours) == 0:
            return -2
        if len(self.packet_list.output) == 0:
            return -4
        packet = self.packet_list.get_first()
        target = packet.target
        for node in self.iterate_neighbours():
            if node == target:
                return node
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
        for i in range(len(self.neighbours) - 1):
            send_probability = random()
            probability = pow(self.neighbours[i].degree(), alpha) / self.get_neighbours_degree(alpha)
            if send_probability < probability:
                return self.neighbours[i].id
        return self.neighbours[-1].id

    def iterate_neighbours(self):
        for i in self.neighbours:
            yield i

    def get_data(self):
        tmp = 0
        for i in self.packet_list.iterate_queue():
            tmp += i.info()
        return tmp

    def get_packets_num(self):
        return self.packet_list.size()