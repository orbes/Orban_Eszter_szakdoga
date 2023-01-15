# from random import randrange, random, randint, sample
import matplotlib.pyplot as plt
import numpy as np
import collections
import os.path
from network.Node import Node
from network.IntelligentNode import IntelligentNode
from simulation.Logs import Logs
import os, shutil
import random


class Graph:
    def __init__(self, packet_type, network_type, n):
        self.log_file()
        self.network_type = ""
        self.n = n
        self.graph = []
        self.num_of_intelligent_node = 0
        self.packet_type = packet_type
        self.generate()
        self.longest_short_path = 0
        if network_type == 0:
            self.network_type = 'Scale_free'
            print("Start generate the graph: " + self.network_type)
            self.generate_scale_free()
        elif network_type == 1:
            self.network_type = 'Random'
            print("Start generate the graph: " + self.network_type)
            self.generate_random()
        elif network_type == 2:
            self.read('./network/graph_scale_' + str(self.n) + '.txt')
            self.network_type = 'Scale_free'
            print("Start generate the graph: " + self.network_type)
        else:
            self.network_type = 'Random'
            self.read('./network/graph_random_' + str(self.n) + '.txt')
            print("Start generate the graph: " + self.network_type)
        self.log_graph = Logs('graph.txt')
        self.print_graph()
        self.edge_probability()
        self.calculate_centrality()
        self.log_graph.close()

    def __repr__(self):
        repr = ''
        for node in self.iterate_graph():
            repr += str(node)
        return repr

    @staticmethod
    def log_file():
        folder = "./logs/nodes"
        if not os.path.exists(folder):
            os.makedirs(folder)

    def generate(self):
        for i in range(self.n):
            self.graph.append(Node(i, self.n, self.packet_type))
        int_nodes = random.sample(range(self.n), self.num_of_intelligent_node)
        for i in int_nodes:
            self.graph[i] = IntelligentNode(i, self.n, self.packet_type)

    def add_new_intelligent_node(self):
        node = np.random.randint(self.n)
        while isinstance(self.graph[node], IntelligentNode):
            node = np.random.randint(self.n)
        tmp = self.graph[node].neighbours
        self.graph[node] = IntelligentNode(node, self.n, self.packet_type, tmp)
        self.num_of_intelligent_node += 1

    def add_new_intelligent_node_in_id(self, node_id):
        tmp = self.graph[node_id].neighbours
        tmp_type = self.graph[node_id].packet_type
        self.graph[node_id] = IntelligentNode(node_id, self.n, tmp_type, tmp)

    def remove_all_int_node(self):
        self.num_of_intelligent_node = 0
        for node in self.iterate_graph():
            if isinstance(node, IntelligentNode):
                tmp = node.neighbours
                self.graph[node.id] = Node(node.id, self.n, node.packet_type)
                self.graph[node.id].add_new_neighbours(tmp)

    def print_graph(self):
        for i in self.graph:
            if isinstance(i, IntelligentNode):
                self.log_graph.log("i" + str(i.id) + ": " + i.get_neighbours() + '\n')
            else:
                self.log_graph.log(str(i.id) + ": " + i.get_neighbours() + '\n')

    def all_link(self):
        tmp = 0
        for i in self.graph:
            tmp += i.degree()
        return tmp / 2

    def edge_probability(self):
        tmp = {}
        for i in range(len(self.graph)):
            tmp[self.graph[i].degree()] = 0
        for i in range(len(self.graph)):
            tmp[self.graph[i].degree()] += 1
        data = collections.OrderedDict(sorted(tmp.items()))
        k = list(data.keys())
        p_k = list(data.values())
        p_k = [a / self.n for a in p_k]
        plt.plot(k, p_k)
        plt.title("Fokszám eloszlás")
        plt.xlabel("k")
        plt.ylabel("p(k)")
        complete_name = os.path.join('./logs', "Fokszam_eloszlas.png")
        plt.savefig(complete_name)
        plt.close()

    def copy(self):
        for i in self.iterate_graph():
            i.copy()

    def clear_queue(self):
        for i in self.iterate_graph():
            i.clear_queue()

    def generate_scale_free(self):
        for i in range(6):
            if i < 5:
                self.graph[i].add_new_neighbour(self.graph[i + 1])
                self.graph[i + 1].add_new_neighbour(self.graph[i])
            else:
                self.graph[5].add_new_neighbour(self.graph[0])
                self.graph[0].add_new_neighbour(self.graph[5])
        for i in range(6, len(self.graph)):
            counter = 0
            while 1:
                for j in range(i):
                    pi = self.graph[j].degree() / self.all_link()
                    if random.random() < pi:
                        self.graph[i].add_new_neighbour(self.graph[j])
                        self.graph[j].add_new_neighbour(self.graph[i])
                        counter += 1
                if counter > 0:
                    break

    def generate_random(self):
        num_edges = 0
        for j in range(self.n - 1):
            for i in range(j + 1, self.n):
                rnd = random.random()
                if rnd > 0.98:
                    self.graph[i].add_new_neighbour(self.graph[j])
                    self.graph[j].add_new_neighbour(self.graph[i])
                    num_edges += 1
        for i in range(len(self.graph)):
            if self.graph[i].degree() == 0:
                tmp = random.randint(0, self.n)
                self.graph[i].add_new_neighbour(self.graph[tmp])
                self.graph[tmp].add_new_neighbour(self.graph[i])
        print(num_edges)

    def read(self, file_name):
        file = open(file_name, 'r')
        lines = file.readlines()
        count = 0
        num_edge = 0
        for line in lines:
            count += 1
            nodes = line.strip().split(': ')
            start_node = nodes[0]
            goal_nodes = line.split()
            for i in range(len(goal_nodes) - 1):
                self.graph[int(start_node)].add_new_neighbour(self.graph[int(goal_nodes[i + 1])])
                num_edge += 1
        print(num_edge / 2)
        self.n = count

    def close_nodes_log_files(self):
        for node in self.graph:
            node.close()

    def iterate_graph(self):
        for node in self.graph:
            yield node

    def BFS(self, node_from, node_to):
        target = node_to
        expolered = []
        queue = [[node_from]]
        all_path = []
        tmp = 99999999
        while queue:
            path = queue.pop(0)
            node = path[-1]
            if node not in expolered:
                neighbours = node.neighbours
                for neighbour in neighbours:
                    new_path = list(path)
                    new_path.append(neighbour)
                    queue.append(new_path)
                    if neighbour == target and len(new_path) <= tmp:
                        tmp = len(new_path)
                        all_path.append(new_path)
                expolered.append(node)
        return all_path

    # hány legrövidebb út megy át rajta
    def calculate_centrality(self):
        for i in range(len(self.graph) - 1):
            for j in range(i + 1, len(self.graph)):
                paths = self.BFS(self.graph[i], self.graph[j])
                counter = len(paths)
                for path in paths:
                    if len(path) > self.longest_short_path:
                        self.longest_short_path = len(path)
                    for k in range(1, len(path) - 1):
                        node = path[k].id
                        self.graph[node].centrality += 1 / counter

    def get_neighbours_by_id(self, node_id):
        for i in self.iterate_graph():
            if i.id == node_id:
                return i.neighbours_id_list()
