from time import time
import matplotlib.pyplot as plt
import os.path
from simulation.Logs import Logs
from network.IntelligentNode import IntelligentNode
from operator import add
from collections import OrderedDict
import numpy as np
import random


class Simulation:
    def __init__(self, G, c_n=10):
        self.G = G
        self.graph = G.graph
        self.n = G.n
        self.c_n = c_n
        self.alpha = -1
        self.log_simulation = Logs("simulation.txt")
        self.log_packet = Logs("packets.txt")
        self.arrived_packets = []
        self.figure1 = [[], []]  # l, átlagos sorhossz
        self.figure2 = [[], []]  # l, csomagok átlag úthossza
        self.figure3 = [[], []]  # l, távolság
        self.figure4 = [[], [], []]  # fix N_L [fokszámok][l - x] [n_l - y] csomópontok átlagos sorhossza
        self.figure5 = [[], [], []]  # fix P_L [fokszámok][l - x] [p_l - y] csomagok átlagos élettartama
        self.fig_R_L_t = [[],
                          []]  # átlagos úthoszza a célba ért csomagoknak [[l], [[sum_úthoszak/|D|],[sum_úthoszak/|D|], ..]]
        self.fig_n_o_t = [[], []]  # túlterhelt csomópontok száma [[l], [[túlterhelt csomópontok száma],[],[]...]]
        self.fig_n_pio_t = [[], []]  # túlterhelt csomópontokban a csomagok száma [[l],[[npio],[],...]]
        self.fig_pa_t = [[], []]  # beérkezett csomagok aránya
        self.fig_data = [[], [[], [], [], []]]  # counter, K=0, K=1, K=2, K=5
        self.times_list = [[], []]  # num int node
        self.fig_data_in_sim = [[], [[], [], [], []]]
        self.fig_cent_nl = [[], []]
        self.fig_cent_pl = [[], []]
        self.fig_L_max = [[], [], [], []]
        self.packet_distance = {}
        self.data_in_t = 0    #mennyi adat továbbítódott egy időpillanatban
        self.sim_time = 500
        self.sim_counter = 10  # hányszor fusson a szimuláció

    def close_files(self):
        self.log_simulation.close()
        self.log_packet.close()

    # counter = l időpillanatokban generálódó csomagok
    def start(self):
        self.log_simulation.log("Start\n")
        print("Start simulation - diff K")
        self.G.remove_all_int_node()
        self.G.clear_queue()
        self.log_simulation.log("Csomag generálás kezdés \n")
        counter = [10, 15, 20, 25, 30, 35]
        percent_int_node = [0, 0.01, 0.02, 0.05]
        num_int_node = [item * self.n for item in percent_int_node]
        self.figure1[0] = counter
        self.figure2[0] = counter
        self.fig_R_L_t[0] = counter
        self.fig_n_o_t[0] = counter
        self.fig_n_pio_t[0] = counter
        self.fig_pa_t[0] = counter
        self.fig_data[0] = counter
        self.times_list[0] = percent_int_node
        self.fig_data_in_sim[0] = counter
        self.figure1[1] = [[0 for _ in range(len(counter))] for _ in range(len(num_int_node))]
        self.figure2[1] = [[0 for _ in range(len(counter))] for _ in range(len(num_int_node))]
        self.fig_R_L_t[1] = [[0 for _ in range(len(counter))] for _ in range(len(num_int_node))]
        self.fig_n_o_t[1] = [[0 for _ in range(len(counter))] for _ in range(len(num_int_node))]
        self.fig_n_pio_t[1] = [[0 for _ in range(len(counter))] for _ in range(len(num_int_node))]
        self.fig_pa_t[1] = [[0 for _ in range(len(counter))] for _ in range(len(num_int_node))]
        self.fig_data[1] = [[[0 for _ in range(self.sim_time)]
                             for _ in range(len(counter))]
                            for _ in range(len(num_int_node))]
        self.fig_data_in_sim[1] = [[[0 for _ in range(self.sim_time)]
                                    for _ in range(6)]
                                   for _ in range(4)]
        self.times_list[1] = [0 for _ in num_int_node]
        for j in np.arange(self.sim_counter):
            print(str(j + 1) + '. futás')
            self.G.remove_all_int_node()
            for i in np.arange(len(num_int_node)):
                print("intnode = " + str(num_int_node[i]))
                self.clear_int_nodes()
                data = []
                pr_data = []
                data_R_L_t = []
                data_n_o_t = []
                data_n_pio_t = []
                data_pa_t = []
                data_adat0 = []
                data_sim0 = []
                self.data = 0
                while self.G.num_of_intelligent_node != num_int_node[i]:
                    self.G.add_new_intelligent_node()
                self.log_simulation.log("intelligens nodok szama: " + str(self.G.num_of_intelligent_node) + '\n')
                start_time = time()
                for l in counter:
                    self.clear_int_nodes()
                    print("l = " + str(l))
                    self.log_simulation.log("l = " + str(l) + '\n')
                    time_ = 1
                    data_adat1 = []
                    data_sim = []
                    while time_ <= self.sim_time:
                        self.data_in_t = 0
                        self.start_generate_packets(l, time_)
                        self.start_packet_forwarding(self.alpha, time_, num_int_node[i])
                        data_adat1.append(self.calculate_data(num_int_node[i]))
                        data_sim.append(self.data_in_t)
                        time_ += 1
                    data.append(self.node_queue_length_avg())
                    pr_data.append(self.arrived_packets_life_time())
                    data_n_o_t.append(self.n_o_and_n_pio()[0])
                    data_n_pio_t.append(self.n_o_and_n_pio()[1])
                    data_R_L_t.append(self.r_l_t())
                    data_pa_t.append(len(self.arrived_packets))
                    data_adat0.append(data_adat1)
                    data_sim0.append(data_sim)
                    self.G.clear_queue()
                    self.arrived_packets.clear()
                run_time = time() - start_time
                self.times_list[1][i] += run_time
                self.figure1[1][i] = list(map(add, self.figure1[1][i], data))
                self.figure2[1][i] = list(map(add, self.figure2[1][i], pr_data))
                self.fig_n_o_t[1][i] = list(map(add, self.fig_n_o_t[1][i], data_n_o_t))
                self.fig_n_pio_t[1][i] = list(map(add, self.fig_n_pio_t[1][i], data_n_pio_t))
                self.fig_R_L_t[1][i] = list(map(add, self.fig_R_L_t[1][i], data_R_L_t))
                self.fig_pa_t[1][i] = list(map(add, self.fig_pa_t[1][i], data_pa_t))
                self.fig_data[1][i] = np.add(self.fig_data[1][i], data_adat0)
                self.fig_data_in_sim[1][i] = np.add(self.fig_data_in_sim[1][i], data_sim0)
                self.log_simulation.log("Futási idő: " + str(run_time) + " seconds\n")
        self.figure()
        print("End Simulation")

    def start_fix(self):
        print("Start simulation - 5% int node - diff edge num")
        self.G.remove_all_int_node()
        self.G.clear_queue()
        tmp_list = self.find_degree_nodes()
        int_nodes_max = tmp_list[0]
        int_nodes_mid = tmp_list[1]  # ebbe kell egy alacsony, magas, és közepes fokszámú
        int_nodes_min = tmp_list[2]
        int_nodes = [[], int_nodes_max, int_nodes_mid, int_nodes_min]
        counter = [10, 15, 20, 25, 30, 35]  # node amiből intelligenset fogunk csinálni
        self.figure4[1] = counter
        self.figure5[1] = counter
        labels = ['nincs', 'magas', 'kozepes', 'alacsony']
        for i in range(4):
            print(labels[i])
            self.G.remove_all_int_node()
            if len(int_nodes[i]) != 0:
                for int_node in int_nodes[i]:
                    self.G.add_new_intelligent_node_in_id(int_node.id)
            data_n_l = []
            data_p_l = []
            for l in counter:
                self.clear_int_nodes()
                print("l = " + str(l))
                self.log_simulation.log("l = " + str(l) + '\n')
                time_ = 1
                while time_ <= self.sim_time:
                    self.start_generate_packets(l, time_)
                    self.start_packet_forwarding(self.alpha, time_, 10)
                    time_ += 1
                data_n_l.append(self.node_queue_length_avg())
                data_p_l.append(self.arrived_packets_life_time())
                self.G.clear_queue()
                self.arrived_packets.clear()
            self.figure4[2].append(data_n_l)
            self.figure5[2].append(data_p_l)
        self.figure_fix_n_l()
        self.figure_fix_p_l()
        print('End simulation')

    def start_cent(self):
        print("Start simulation - 5% int node - diff centrality")
        self.G.remove_all_int_node()
        self.G.clear_queue()
        tmp_list = self.find_centrality_nodes()

        int_nodes_max = tmp_list[0]
        int_nodes_mid = tmp_list[1]  # ebbe kell egy alacsony, magas, és közepes fokszámú
        int_nodes_min = tmp_list[2]
        int_nodes = [[], int_nodes_max, int_nodes_mid, int_nodes_min]
        counter = [10, 15, 20, 25, 30, 35]
        self.fig_cent_nl[0] = counter
        self.fig_cent_pl[0] = counter
        labels = ['nincs', 'magas', 'kozepes', 'alacsony']
        for i in range(4):
            print(labels[i])
            self.G.remove_all_int_node()
            if len(int_nodes[i]) != 0:
                for int_node in int_nodes[i]:
                    self.G.add_new_intelligent_node_in_id(int_node.id)
            data_n_l = []
            data_p_l = []
            for l in counter:
                self.clear_int_nodes()
                print("l = " + str(l))
                self.log_simulation.log("l = " + str(l) + '\n')
                time_ = 1
                while time_ <= self.sim_time:
                    self.start_generate_packets(l, time_)
                    self.start_packet_forwarding(self.alpha, time_, 10)
                    time_ += 1
                data_n_l.append(self.node_queue_length_avg())
                data_p_l.append(self.arrived_packets_life_time())
                self.G.clear_queue()
                self.arrived_packets.clear()
            self.fig_cent_pl[1].append(data_p_l)
            self.fig_cent_nl[1].append(data_n_l)
        self.figure_cent()
        print('End simulation')

    def start_simple(self):
        self.log_simulation.log("Start\n")
        print("Start simulation - one int node")
        self.G.remove_all_int_node()
        self.G.clear_queue()
        self.log_simulation.log("Csomag generálás kezdés \n")
        counter = [10, 15, 20, 25, 30, 35]
        self.figure3[1] = [[0 for _ in range(self.sim_time)] for _ in range(len(counter))]
        for i in range(self.sim_counter):
            print(str(i + 1) + '. futás')
            for j in range(len(counter)):
                self.clear_int_nodes()
                print("l = " + str(counter[j]))
                self.log_simulation.log("l = " + str(counter[j]) + '\n')
                time_ = 1
                data_diff = []
                while time_ <= self.sim_time:
                    self.start_generate_packets(counter[j], time_)
                    self.start_packet_forwarding(self.alpha, time_, 1)
                    data_diff.append(self.diff())
                    time_ += 1
                self.G.clear_queue()
                self.arrived_packets.clear()
                self.figure3[1][j] = list(map(add, self.figure3[1][j], data_diff))
        self.figure_diff()
        print('End Simulation - one int node')

    def start_k0(self):
        print("Start without intelligent node")
        self.G.remove_all_int_node()
        self.G.clear_queue()
        counter = [15, 25, 35]
        map_arrived = [{}, {}, {}]
        map_all = [{}, {}, {}]
        for l in range(len(counter)):
            print("l=" + str(counter[l]))
            while self.G.num_of_intelligent_node != 10:
                self.G.add_new_intelligent_node()
            time_ = 1
            self.packet_distance = {}
            while time_ <= self.sim_time:
                self.start_generate_packets(counter[l], time_)
                self.start_packet_forwarding(self.alpha, time_, 10)
                time_ += 1
            map_all[l] = self.packet_distance
            for packet in self.arrived_packets:
                if packet.distance not in map_arrived[l].keys():
                    map_arrived[l][packet.distance] = 1
                else:
                    map_arrived[l][packet.distance] += 1
            self.G.clear_queue()
            self.arrived_packets.clear()
        self.figure_bar(map_arrived, map_all)

    def start_L_max(self):
        print("Start L_max")
        self.G.remove_all_int_node()
        self.G.clear_queue()
        intnode_list = [0, 2, 4, 10]
        alpha_list = [-2, -1.5, -1, 0, 0.5]
        self.fig_L_max = [[0 for _ in range(5)] for _ in range(4)]  #[[k0][k1][k2][k5])
        for i in range(self.sim_counter):
            print(str(i+1) + ". futás")
            c_intnode = 0
            for n_in in intnode_list:
                self.G.remove_all_int_node()
                print("int node=" + str(n_in))
                while self.G.num_of_intelligent_node != n_in:
                    self.G.add_new_intelligent_node()
                data_list = []
                for alpha in alpha_list:
                    c_alpha = 0
                    traffic_jam = False
                    print("alpha=" + str(alpha))
                    for l in range(0, 35):
                        time_ = 1
                        print(l)
                        while time_ <= 500:
                            self.start_generate_packets(l, time_)
                            self.start_packet_forwarding(alpha, time_, n_in)
                            time_ += 1
                        for node in self.graph:
                            if node.packet_list.size() >= 10:
                                data_list.append(l)
                                traffic_jam = True
                                break
                        self.G.clear_queue()
                        self.arrived_packets.clear()
                        if traffic_jam:
                            break
                    c_alpha += 1
                self.fig_L_max[c_intnode] = list(map(add, self.fig_L_max[c_intnode], data_list))
                c_intnode += 1
        print(self.fig_L_max)
        self.figure_L_max()

    def start_generate_packets(self, l, birth_time):
        generated = []
        while len(generated) < l:
            tmp = np.random.randint(self.n)
            while tmp in generated:
                tmp = np.random.randint(self.n)
            generated.append(tmp)
        for i in generated:
            packet = self.graph[i].generate_new_packet(birth_time)
            packet.distance = self.BFS(packet)
            if packet.distance not in self.packet_distance.keys():
                self.packet_distance[packet.distance] = 1
            else:
                self.packet_distance[packet.distance] += 1

    def BFS(self, packet):
        target = packet.target
        start = packet.start
        explored = []
        queue = [[start]]
        while queue:
            path = queue.pop(0)
            node = path[-1]
            if node not in explored:
                neighbours = node.neighbours
                for neighbour in neighbours:
                    new_path = list(path)
                    new_path.append(neighbour)
                    queue.append(new_path)
                    if neighbour.id == target:
                        return len(new_path)-1
                explored.append(node)
        return 99999

    def start_packet_forwarding(self, alpha, die_time, num_int_node):
        for node in self.graph:
            i = 0
            is_next: bool = False
            while i < self.c_n and not is_next:
                next_node_id = node.get_next_node_id(alpha)
                if next_node_id == -1:
                    is_next = True
                elif next_node_id == -2:
                    node.clear_queue()
                    is_next = True
                elif next_node_id == -4:
                    is_next = True
                elif next_node_id == -3:
                    print("nincs ervenyes ut!")
                    is_next = True
                elif next_node_id is None:
                    is_next = True
                else:
                    packet = node.get_first_packet()
                    if next_node_id == packet.target:
                        self.arrived_packets.append(packet)
                        packet.arrived(die_time)
                    self.graph[next_node_id].accept_packet(packet)
                    if num_int_node == 0:
                        self.data_in_t += 1
                    else:
                        self.data_in_t += packet.info()
                i += 1
        self.G.copy()

    def find_degree_nodes(self):
        list_d_min = []
        list_d_max = []
        list_d_mid = []
        self.graph.sort(key=lambda x: x.degree())
        for i in range(10):
            list_d_max.append(self.graph[-i])  # max
            list_d_min.append(self.graph[i])  # min
        tmp_list = self.graph[20:-20]
        id_list = random.sample(range(len(tmp_list)), 10)
        for id in id_list:
            list_d_mid.append(self.graph[id])
        self.graph.sort(key=lambda x: x.id)
        print(list_d_max, list_d_mid, list_d_min)
        return list_d_max, list_d_mid, list_d_min

    def find_centrality_nodes(self):
        list_c_min = []
        list_c_max = []
        list_c_mid = []
        self.graph.sort(key=lambda x: x.centrality)
        for i in range(10):
            list_c_max.append(self.graph[-i])
            list_c_min.append(self.graph[i])
        tmp_list = self.graph[20:-20]
        id_list = random.sample(range(len(tmp_list)), 10)
        for id in id_list:
            while self.graph[id].centrality == 0:
                id = random.randint(20, 180)
            list_c_mid.append(self.graph[id])
        self.graph.sort(key=lambda x: x.id)
        print(list_c_max, list_c_mid, list_c_min)
        return list_c_max, list_c_mid, list_c_min

    def n_o_and_n_pio(self):
        counter_n_o = sum(map(lambda node: node.packet_list.size() > self.c_n, self.graph))
        counter_n_pio = sum(node.packet_list.size() for node in self.graph
                            if node.packet_list.size() > self.c_n)
        return counter_n_o, counter_n_pio

    def r_l_t(self):
        tmp = sum(len(a.route) for a in self.arrived_packets)
        return tmp / len(self.arrived_packets)

    def node_queue_length_avg(self):
        queue_length = sum(node.packet_list.size() for node in self.graph)
        return queue_length / self.n

    def arrived_packets_life_time(self):
        life_time_sum = sum(packet.life_time for packet in self.arrived_packets)
        return life_time_sum / len(self.arrived_packets)

    def calculate_data(self, num_int_node):
        if num_int_node != 0:
            return sum(node.get_data() for node in self.G.iterate_graph())
        else:  # megszámoljuk hány csomag van a hálózatban perpill
            return sum(node.get_packets_num() for node in self.G.iterate_graph())

    def clear_int_nodes(self):
        for node in self.graph:
            if isinstance(node, IntelligentNode):
                node.clear()

    def diff(self):
        tmp = 0
        for node in self.graph:
            if isinstance(node, IntelligentNode):
                tmp = self.G.all_link() - node.diff()
        return tmp

    # alcsony magas és közepes fokszám
    def figure_fix_n_l(self):
        x = self.figure4[1]
        y0 = [a for a in self.figure4[2][0]]
        y1 = [a for a in self.figure4[2][1]]
        y2 = [a for a in self.figure4[2][2]]
        y3 = [a for a in self.figure4[2][3]]
        plt.grid(True)
        plt.plot(x, y0, 'yo', label='K=0')
        plt.plot(x, y0, '--y')
        plt.plot(x, y1, 'ro', label='magas')
        plt.plot(x, y1, 'r')
        plt.plot(x, y2, 'bo', label='közepes')
        plt.plot(x, y2, 'b')
        plt.plot(x, y3, 'go', label='alacsony')
        plt.plot(x, y3, 'g')
        plt.legend()
        plt.xlabel("L")
        plt.ylabel('$N_L$(' + str(self.sim_time) + ')')
        complete_name = os.path.join('./logs', 'N_L_kul_fokszamu_int_node_t'
                                     + str(self.sim_time) + '_N' + str(self.n)
                                     + '_' + self.G.network_type + '_' + self.G.packet_type
                                     + '.png')
        plt.savefig(complete_name)
        plt.close()

    # alcsony magas és közepes fokszám
    def figure_fix_p_l(self):
        x = self.figure5[1]
        y0 = self.figure5[2][0]
        y1 = self.figure5[2][1]
        y2 = self.figure5[2][2]
        y3 = self.figure5[2][3]
        plt.grid(True)
        plt.plot(x, y0, 'yo', label='K=0')
        plt.plot(x, y0, '--y')
        plt.plot(x, y1, 'ro', label='magas')
        plt.plot(x, y1, 'r')
        plt.plot(x, y2, 'bo', label='közepes')
        plt.plot(x, y2, 'b')
        plt.plot(x, y3, 'go', label='alacsony')
        plt.plot(x, y3, 'g')
        plt.legend()
        plt.xlabel("L")
        plt.ylabel('$P_L$(' + str(self.sim_time) + ')')
        complete_name = os.path.join('./logs', "P_L_kul_fokszamu_int_node_t"
                                     + str(self.sim_time) + "_N" + str(self.n)
                                     + "_" + self.G.network_type + "_" + self.G.packet_type
                                     + ".png")
        plt.savefig(complete_name)
        plt.close()

    def figure_n_l_t(self):
        x = self.figure1[0]
        y0 = [a / self.sim_counter for a in self.figure1[1][0]]
        y1 = [a / self.sim_counter for a in self.figure1[1][1]]
        y2 = [a / self.sim_counter for a in self.figure1[1][2]]
        y3 = [a / self.sim_counter for a in self.figure1[1][3]]
        plt.grid(True)
        plt.plot(x, y0, 'yo', label='K=0%')
        plt.plot(x, y0, '--y')
        plt.plot(x, y1, 'go', label='K=1%')
        plt.plot(x, y1, '--g')
        plt.plot(x, y2, 'bo', label='K=2%')
        plt.plot(x, y2, '--b')
        plt.plot(x, y3, 'ro', label='K=5%')
        plt.plot(x, y3, '--r')
        plt.legend()
        plt.xlabel("L")
        plt.ylabel('$N_L$(' + str(self.sim_time) + ')')
        complete_name = os.path.join('./logs', "Atlagos_sor_hossz_t"
                                     + str(self.sim_time) + "_N" + str(self.n)
                                     + "_" + self.G.network_type + "_" + self.G.packet_type
                                     + ".png")
        plt.savefig(complete_name)
        plt.close()

    def figure_p_l_t(self):
        plt.grid(True)
        x = self.figure2[0]
        y0 = [a / self.sim_counter for a in self.figure2[1][0]]
        y1 = [a / self.sim_counter for a in self.figure2[1][1]]
        y2 = [a / self.sim_counter for a in self.figure2[1][2]]
        y3 = [a / self.sim_counter for a in self.figure2[1][3]]
        plt.plot(x, y0, 'yo', label='K=0%')
        plt.plot(x, y0, '--y')
        plt.plot(x, y1, 'go', label='K=1%')
        plt.plot(x, y1, '--g')
        plt.plot(x, y2, 'bo', label='K=2%')
        plt.plot(x, y2, '--b')
        plt.plot(x, y3, 'ro', label='K=5%')
        plt.plot(x, y3, '--r')
        plt.legend()
        plt.xlabel("L")
        plt.ylabel("$P_L$(" + str(self.sim_time) + ")")
        complete_name = os.path.join('./logs', "Atlagos_elet_tartam_t"
                                     + str(self.sim_time) + "_N" + str(self.n)
                                     + "_" + self.G.network_type + "_" + self.G.packet_type
                                     + ".png")
        plt.savefig(complete_name)
        plt.close()

    # Távolság
    def figure_diff(self):
        plt.grid(True)
        x = range(500)
        y0 = [a / self.sim_counter for a in self.figure3[1][0]]
        y2 = [a / self.sim_counter for a in self.figure3[1][2]]
        y5 = [a / self.sim_counter for a in self.figure3[1][5]]
        plt.plot(x, y0, 'g', label='L=10')
        plt.plot(x, y2, 'b', label='L=20')
        plt.plot(x, y5, 'r', label='L=35')
        plt.legend()
        plt.xlabel('t')
        plt.ylabel(r'd($\mathbf{A}_r, \mathbf{A}_{n_i}$)(t)')
        complete_name = os.path.join('./logs', "Diff_t"
                                     + str(self.sim_time) + "_N" + str(self.n)
                                     + "_" + self.G.network_type + "_" + self.G.packet_type
                                     + ".png")
        plt.savefig(complete_name)
        plt.close() #

    def figure_time(self):
        plt.grid(True)
        x = [a * 100 for a in self.times_list[0]]
        y = [a / self.sim_counter for a in self.times_list[1]]
        plt.plot(x, y, 'ro', x, y, '--r')
        plt.ylabel("t(s)")
        plt.xlabel("K")
        complete_name = os.path.join('./logs', "Futasi_ido"
                                     + str(self.sim_time) + "_N" + str(self.n)
                                     + "_" + self.G.network_type + "_" + self.G.packet_type
                                     + ".png")
        plt.savefig(complete_name)
        plt.close()

    def figure_R_L_t(self):
        plt.grid(True)
        x = self.fig_R_L_t[0]
        y0 = [a / self.sim_counter for a in self.fig_R_L_t[1][0]]
        y1 = [a / self.sim_counter for a in self.fig_R_L_t[1][1]]
        y2 = [a / self.sim_counter for a in self.fig_R_L_t[1][2]]
        y3 = [a / self.sim_counter for a in self.fig_R_L_t[1][3]]
        plt.plot(x, y0, 'yo', label='K=0%')
        plt.plot(x, y0, '--y')
        plt.plot(x, y1, 'go', label='K=1%')
        plt.plot(x, y1, '--g')
        plt.plot(x, y2, 'bo', label='K=2%')
        plt.plot(x, y2, '--b')
        plt.plot(x, y3, 'ro', label='K=5%')
        plt.plot(x, y3, '--r')
        plt.legend()
        plt.xlabel("L")
        plt.ylabel("$R_L$(" + str(self.sim_time) + ")")
        complete_name = os.path.join('./logs', "Atlagos_ut_hossz_t"
                                     + str(self.sim_time) + "_N" + str(self.n)
                                     + "_" + self.G.network_type + "_" + self.G.packet_type
                                     + ".png")
        plt.savefig(complete_name)
        plt.close()

    def figure_n_o_t(self):
        plt.grid(True)
        x = self.fig_n_o_t[0]
        y0 = [a / self.sim_counter for a in self.fig_n_o_t[1][0]]
        y1 = [a / self.sim_counter for a in self.fig_n_o_t[1][1]]
        y2 = [a / self.sim_counter for a in self.fig_n_o_t[1][2]]
        y3 = [a / self.sim_counter for a in self.fig_n_o_t[1][3]]
        plt.plot(x, y0, 'yo', label='K=0%')
        plt.plot(x, y0, '--y')
        plt.plot(x, y1, 'go', label='K=1%')
        plt.plot(x, y1, '--g')
        plt.plot(x, y2, 'bo', label='K=2%')
        plt.plot(x, y2, '--b')
        plt.plot(x, y3, 'ro', label='K=5%')
        plt.plot(x, y3, '--r')
        plt.legend()
        plt.xlabel("L")
        plt.ylabel("$n_o$(" + str(self.sim_time) + ")")
        complete_name = os.path.join('./logs', "n_o_t"
                                     + str(self.sim_time) + "_N" + str(self.n)
                                     + "_" + self.G.network_type + "_" + self.G.packet_type
                                     + ".png")
        plt.savefig(complete_name)
        plt.close()

    def figure_n_pio_t(self):
        plt.grid(True)
        x = self.fig_n_pio_t[0]
        y0 = [a / self.sim_counter for a in self.fig_n_pio_t[1][0]]
        y1 = [a / self.sim_counter for a in self.fig_n_pio_t[1][1]]
        y2 = [a / self.sim_counter for a in self.fig_n_pio_t[1][2]]
        y3 = [a / self.sim_counter for a in self.fig_n_pio_t[1][3]]
        plt.plot(x, y0, 'yo', label='K=0%')
        plt.plot(x, y0, '--y')
        plt.plot(x, y1, 'go', label='K=1%')
        plt.plot(x, y1, '--g')
        plt.plot(x, y2, 'bo', label='K=2%')
        plt.plot(x, y2, '--b')
        plt.plot(x, y3, 'ro', label='K=5%')
        plt.plot(x, y3, '--r')
        plt.legend()
        plt.xlabel("L")
        plt.ylabel("$n_pio$(" + str(self.sim_time) + ")")
        complete_name = os.path.join('./logs', "n_pio_t"
                                     + str(self.sim_time) + "_N" + str(self.n)
                                     + "_" + self.G.network_type + "_" + self.G.packet_type
                                     + ".png")
        plt.savefig(complete_name)
        plt.close()

    def figure_pa_t(self):
        plt.grid(True)
        x = self.fig_pa_t[0]
        y = [self.sim_time * i for i in x]
        print(self.fig_pa_t)
        y0 = [(a / self.sim_counter) / i for a, i in zip(self.fig_pa_t[1][0], y)]
        y1 = [(a / self.sim_counter) / i for a, i in zip(self.fig_pa_t[1][1], y)]
        y2 = [(a / self.sim_counter) / i for a, i in zip(self.fig_pa_t[1][2], y)]
        y3 = [(a / self.sim_counter) / i for a, i in zip(self.fig_pa_t[1][3], y)]
        plt.plot(x, y0, 'yo', label='K=0%')
        plt.plot(x, y0, '--y')
        plt.plot(x, y1, 'go', label='K=1%')
        plt.plot(x, y1, '--g')
        plt.plot(x, y2, 'bo', label='K=2%')
        plt.plot(x, y2, '--b')
        plt.plot(x, y3, 'ro', label='K=5%')
        plt.plot(x, y3, '--r')
        plt.legend()
        plt.xlabel("L")
        plt.ylabel("$p_a($" + str(self.sim_time) + ")")
        complete_name = os.path.join('./logs', "pa_t"
                                     + str(self.sim_time) + "_N" + str(self.n)
                                     + "_" + self.G.network_type + "_" + self.G.packet_type
                                     + ".png")
        plt.savefig(complete_name)
        plt.close()

    def figure_D_t(self):
        plt.grid(True)
        x = self.fig_pa_t[0]
        y0 = [a / self.sim_counter for a in self.fig_pa_t[1][0]]
        y1 = [a / self.sim_counter for a in self.fig_pa_t[1][1]]
        y2 = [a / self.sim_counter for a in self.fig_pa_t[1][2]]
        y3 = [a / self.sim_counter for a in self.fig_pa_t[1][3]]
        y4 = [self.sim_time * i for i in x]
        plt.plot(x, y0, 'yo', label='K=0%')
        plt.plot(x, y0, '--y')
        plt.plot(x, y1, 'go', label='K=1%')
        plt.plot(x, y1, '--g')
        plt.plot(x, y2, 'bo', label='K=2%')
        plt.plot(x, y2, '--b')
        plt.plot(x, y3, 'ro', label='K=5%')
        plt.plot(x, y3, '--r')
        plt.plot(x, y4, 'mo', label='|$D_L(500)$|')
        plt.plot(x, y4, '--m')
        plt.legend()
        plt.xlabel("L")
        plt.ylabel("$|D($" + str(self.sim_time) + ")|")
        complete_name = os.path.join('./logs', "D_t"
                                     + str(self.sim_time) + "_N" + str(self.n)
                                     + "_" + self.G.network_type + "_" + self.G.packet_type
                                     + ".png")
        plt.savefig(complete_name)
        plt.close()

    def figure_data(self):
        plt.grid(True)
        x = range(self.sim_time)
        y0 = [a / self.sim_counter for a in self.fig_data[1][0][0]]  # l=10
        y2 = [a / self.sim_counter for a in self.fig_data[1][0][2]]  # l=20
        y5 = [a / self.sim_counter for a in self.fig_data[1][0][5]]  # l=35
        plt.plot(x, y0, 'y', label='L=10')
        plt.plot(x, y2, 'b', label='L=20')
        plt.plot(x, y5, 'r', label='L=35')
        plt.legend()
        plt.yscale("log")
        plt.xlabel("t(s)")
        plt.ylabel("<N(" + str(self.sim_time) + ")>(bit)")
        complete_name = os.path.join('./logs', "data_K0_"
                                     + str(self.sim_time) + "_N" + str(self.n)
                                     + "_" + self.G.network_type + "_" + self.G.packet_type
                                     + ".png")
        plt.savefig(complete_name)
        plt.close()

        plt.grid(True)
        x = range(self.sim_time)
        y0 = [a / self.sim_counter for a in self.fig_data[1][1][0]]  # l=10
        y2 = [a / self.sim_counter for a in self.fig_data[1][1][2]]  # l=20
        y5 = [a / self.sim_counter for a in self.fig_data[1][1][5]]  # l=35
        plt.plot(x, y0, 'y', label='L=10')
        plt.plot(x, y2, 'b', label='L=20')
        plt.plot(x, y5, 'r', label='L=35')
        plt.legend()
        plt.yscale("log")
        plt.xlabel("t(s)")
        plt.ylabel("<N(" + str(self.sim_time) + ")>(bit)")
        complete_name = os.path.join('./logs', "data_K1_"
                                     + str(self.sim_time) + "_N" + str(self.n)
                                     + "_" + self.G.network_type + "_" + self.G.packet_type
                                     + ".png")
        plt.savefig(complete_name)
        plt.close()

        plt.grid(True)
        x = range(self.sim_time)
        y0 = [a / self.sim_counter for a in self.fig_data[1][2][0]]  # l=10
        y2 = [a / self.sim_counter for a in self.fig_data[1][2][2]]  # l=20
        y5 = [a / self.sim_counter for a in self.fig_data[1][2][5]]  # l=35
        plt.plot(x, y0, 'y', label='L=10')
        plt.plot(x, y2, 'b', label='L=20')
        plt.plot(x, y5, 'r', label='L=35')
        plt.legend()
        plt.yscale("log")
        plt.xlabel("t(s)")
        plt.ylabel("<N(" + str(self.sim_time) + ")>(bit)")
        complete_name = os.path.join('./logs', "data_K2_"
                                     + str(self.sim_time) + "_N" + str(self.n)
                                     + "_" + self.G.network_type + "_" + self.G.packet_type
                                     + ".png")
        plt.savefig(complete_name)
        plt.close()

        plt.grid(True)
        x = range(self.sim_time)
        y0 = [a / self.sim_counter for a in self.fig_data[1][3][0]]  # l=10
        y2 = [a / self.sim_counter for a in self.fig_data[1][3][2]]  # l=20
        y5 = [a / self.sim_counter for a in self.fig_data[1][3][5]]  # l=35
        plt.plot(x, y0, 'y', label='L=10')
        plt.plot(x, y2, 'b', label='L=20')
        plt.plot(x, y5, 'r', label='L=35')
        plt.legend()
        plt.yscale("log")
        plt.xlabel("t")
        plt.ylabel("<N(" + str(self.sim_time) + ")>(bit)")
        complete_name = os.path.join('./logs', "data_K5_"
                                     + str(self.sim_time) + "_N" + str(self.n)
                                     + "_" + self.G.network_type + "_" + self.G.packet_type
                                     + ".png")
        plt.savefig(complete_name)
        plt.close()

    def figure_data_in_sim(self):
        plt.grid(True)
        x = range(self.sim_time)
        y0 = [(a / self.sim_counter) / 8000 for a in self.fig_data_in_sim[1][0][0]]  # l=10
        y2 = [(a / self.sim_counter) / 8000 for a in self.fig_data_in_sim[1][0][2]]  # l=20
        y5 = [(a / self.sim_counter) / 8000 for a in self.fig_data_in_sim[1][0][5]]  # l=35
        plt.plot(x, y0, 'y', label='L=10')
        plt.plot(x, y2, 'b', label='L=20')
        plt.plot(x, y5, 'r', label='L=35')
        plt.legend()
        plt.xlabel("t(s)")
        plt.ylabel("<S(" + str(self.sim_time) + ")>(kB)")
        complete_name = os.path.join('./logs', "data_in_sim_K0_"
                                     + str(self.sim_time) + "_N" + str(self.n)
                                     + "_" + self.G.network_type + "_" + self.G.packet_type
                                     + ".png")
        plt.savefig(complete_name)
        plt.close()

        plt.grid(True)
        x = range(self.sim_time)
        y0 = [(a / self.sim_counter) / 8000 for a in self.fig_data_in_sim[1][1][0]]  # l=10
        y2 = [(a / self.sim_counter) / 8000 for a in self.fig_data_in_sim[1][1][2]]  # l=20
        y5 = [(a / self.sim_counter) / 8000 for a in self.fig_data_in_sim[1][1][5]]  # l=35
        plt.plot(x, y0, 'y', label='L=10')
        plt.plot(x, y2, 'b', label='L=20')
        plt.plot(x, y5, 'r', label='L=35')
        plt.legend()
        plt.xlabel("t(s)")
        plt.ylabel("<S(" + str(self.sim_time) + ")>(kB)")
        complete_name = os.path.join('./logs', "data_in_sim_K1_"
                                     + str(self.sim_time) + "_N" + str(self.n)
                                     + "_" + self.G.network_type + "_" + self.G.packet_type
                                     + ".png")
        plt.savefig(complete_name)
        plt.close()

        plt.grid(True)
        x = range(self.sim_time)
        y0 = [(a / self.sim_counter) / 8000 for a in self.fig_data_in_sim[1][2][0]]  # l=10
        y2 = [(a / self.sim_counter) / 8000 for a in self.fig_data_in_sim[1][2][2]]  # l=20
        y5 = [(a / self.sim_counter) / 8000 for a in self.fig_data_in_sim[1][2][5]]  # l=35
        plt.plot(x, y0, 'y', label='L=10')
        plt.plot(x, y2, 'b', label='L=20')
        plt.plot(x, y5, 'r', label='L=35')
        plt.legend()
        plt.xlabel("t(s)")
        plt.ylabel("<S(" + str(self.sim_time) + ")>(kB)")
        complete_name = os.path.join('./logs', "data_in_sim_K2_"
                                     + str(self.sim_time) + "_N" + str(self.n)
                                     + "_" + self.G.network_type + "_" + self.G.packet_type
                                     + ".png")
        plt.savefig(complete_name)
        plt.close()

        plt.grid(True)
        x = range(self.sim_time)
        y0 = [(a / self.sim_counter) / 8000 for a in self.fig_data_in_sim[1][3][0]]  # l=10
        y2 = [(a / self.sim_counter) / 8000 for a in self.fig_data_in_sim[1][3][2]]  # l=20
        y5 = [(a / self.sim_counter) / 8000 for a in self.fig_data_in_sim[1][3][5]]  # l=35
        plt.plot(x, y0, 'y', label='L=10')
        plt.plot(x, y2, 'b', label='L=20')
        plt.plot(x, y5, 'r', label='L=35')
        plt.legend()
        plt.xlabel("t")
        plt.ylabel("<S(" + str(self.sim_time) + ")>(kB)")
        complete_name = os.path.join('./logs', "data_in_sim_K5_"
                                     + str(self.sim_time) + "_N" + str(self.n)
                                     + "_" + self.G.network_type + "_" + self.G.packet_type
                                     + ".png")
        plt.savefig(complete_name)
        plt.close()

    def figure_cent(self):
        x = self.fig_cent_pl[0]
        y0 = self.fig_cent_pl[1][0]
        y1 = self.fig_cent_pl[1][1]
        y2 = self.fig_cent_pl[1][2]
        y3 = self.fig_cent_pl[1][3]
        plt.grid(True)
        plt.plot(x, y0, 'yo', label='K=0')
        plt.plot(x, y0, '--y')
        plt.plot(x, y1, 'ro', label='magas')
        plt.plot(x, y1, 'r')
        plt.plot(x, y2, 'bo', label='közepes')
        plt.plot(x, y2, 'b')
        plt.plot(x, y3, 'go', label='alacsony')
        plt.plot(x, y3, 'g')
        plt.legend()
        plt.xlabel("L")
        plt.ylabel('$P_L$(' + str(self.sim_time) + ')')
        complete_name = os.path.join('./logs', "P_L_kul_cent_int_node_t"
                                     + str(self.sim_time) + "_N" + str(self.n)
                                     + "_" + self.G.network_type + "_" + self.G.packet_type
                                     + ".png")
        plt.savefig(complete_name)
        plt.close()

        x = self.fig_cent_nl[0]
        y0 = self.fig_cent_nl[1][0]
        y1 = self.fig_cent_nl[1][1]
        y2 = self.fig_cent_nl[1][2]
        y3 = self.fig_cent_nl[1][3]
        plt.grid(True)
        plt.plot(x, y0, 'yo', label='K=0')
        plt.plot(x, y0, '--y')
        plt.plot(x, y1, 'ro', label='magas')
        plt.plot(x, y1, 'r')
        plt.plot(x, y2, 'bo', label='közepes')
        plt.plot(x, y2, 'b')
        plt.plot(x, y3, 'go', label='alacsony')
        plt.plot(x, y3, 'g')
        plt.legend()
        plt.xlabel("L")
        plt.ylabel('$P_L$(' + str(self.sim_time) + ')')
        complete_name = os.path.join('./logs', "N_L_kul_cent_int_node_t"
                                     + str(self.sim_time) + "_N" + str(self.n)
                                     + "_" + self.G.network_type + "_" + self.G.packet_type
                                     + ".png")
        plt.savefig(complete_name)
        plt.close()

    def figure(self):
        self.figure_p_l_t()
        self.figure_n_l_t()
        self.figure_n_pio_t()
        self.figure_n_o_t()
        self.figure_R_L_t()
        self.figure_pa_t()
        self.figure_D_t()
        self.figure_time()
        self.figure_data_in_sim()
        self.figure_data()

    def figure_bar(self, map_arrived, map_all):
        max_len = 0
        for i in range(len(map_all)):
            if len(map_all[i]) > max_len:
                max_len = len(map_all[i])
        for my_dict in map_arrived:
            for i in range(1, max_len+1):
                if i not in my_dict.keys():
                    my_dict[i] = 0
        for my_dict in map_all:
            for i in range(1, max_len+1):
                if i not in my_dict.keys():
                    my_dict[i] = 0

        for i in range(len(map_all)):
            map_arrived[i] = OrderedDict(sorted(map_arrived[i].items()))
            map_all[i] = OrderedDict(sorted(map_all[i].items()))

        percent_list0 = [(x / y) for x, y in zip(list(map_arrived[0].values()),
                                                 list(map_all[0].values())) if y != 0]
        percent_list1 = [(x / y) for x, y in zip(list(map_arrived[1].values()),
                                                 list(map_all[1].values())) if y != 0]
        percent_list2 = [(x / y) for x, y in zip(list(map_arrived[2].values()),
                                                 list(map_all[2].values())) if y != 0]
        print(percent_list0, percent_list1, percent_list2)

        barWidth = 0.25
        plt.subplots(figsize=(12, 8))
        l11 = list(map_arrived[0].values())    #arrived packets l=15
        l12 = list(map_arrived[1].values())    #arrived packets l=25
        l13 = list(map_arrived[2].values())    #arrived packets l=35

        l21 = list(map_all[0].values())   #all packet plot l=15
        l22 = list(map_all[1].values())   #all packet plot l=25
        l23 = list(map_all[2].values())   #all packet plot l=35

        l1_plot = [x2 - x1 for x1, x2 in zip(l11, l21)]
        l2_plot = [x2 - x1 for x1, x2 in zip(l12, l22)]
        l3_plot = [x2 - x1 for x1, x2 in zip(l13, l23)]

        br11 = np.arange(len(l11))
        br12 = [x + barWidth for x in br11]
        br13 = [x + barWidth for x in br12]

        plt.bar(br11, l11, color='y', width=barWidth, edgecolor='grey', label='L=15')
        plt.bar(br11, l1_plot, color='gold', bottom=l11, width=barWidth, edgecolor='grey')
        plt.bar(br12, l12, color='b', width=barWidth, edgecolor='grey', label='L=25')
        plt.bar(br12, l2_plot, color='cornflowerblue', bottom=l12, width=barWidth, edgecolor='grey')
        plt.bar(br13, l13, color='r', width=barWidth, edgecolor='grey', label='L=35')
        plt.bar(br13, l3_plot, color='lightsalmon', bottom=l13, width=barWidth, edgecolor='grey')

        plt.xlabel('d')
        plt.ylabel('$|D_L(500)|$')
        plt.xticks([r + barWidth for r in range(len(l11))], map_arrived[0].keys())
        plt.legend()

        complete_name = os.path.join('./logs', "bar_t"
                                     + str(self.sim_time) + "_N" + str(self.n)
                                     + "_" + self.G.network_type + "_" + self.G.packet_type
                                     + ".png")
        plt.savefig(complete_name)
        plt.close()

    def figure_L_max(self):
        x = [-2, -1.5, -1, 0, 0.5]

        y0 = [a / self.sim_counter for a in self.fig_L_max[0]]
        y1 = [a / self.sim_counter for a in self.fig_L_max[1]]
        y2 = [a / self.sim_counter for a in self.fig_L_max[2]]
        y3 = [a / self.sim_counter for a in self.fig_L_max[3]]

        plt.grid(True)
        plt.plot(x, y0, 'yo', label='K=0')
        plt.plot(x, y0, '--y')
        plt.plot(x, y1, 'go', label='K=1')
        plt.plot(x, y1, '--g')
        plt.plot(x, y2, 'bo', label='K=2')
        plt.plot(x, y2, '--b')
        plt.plot(x, y3, 'ro', label='K=5')
        plt.plot(x, y3, '--r')
        plt.legend()
        plt.xlabel(r"$\alpha$")
        plt.ylabel('$L^{max}$(' + str(self.sim_time) + ')')
        complete_name = os.path.join('./logs', "L_max"
                                     + str(self.sim_time) + "_N" + str(self.n)
                                     + "_" + self.G.network_type + "_" + self.G.packet_type
                                     + ".png")
        plt.savefig(complete_name)
        plt.close()
