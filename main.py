from simulation.Simulation import Simulation
from simulation.Graph import Graph
import os, shutil


"""
0: new scale free
1: new random
2: old scale free
3: old random
"""
if __name__ == "__main__":
    folder = './logs'
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

    network1 = Graph('A', 2, 200)
    simulation1 = Simulation(network1)
    # simulation1.start_L_max()
    simulation1.start()
    # simulation1.start_k0()
    # simulation1.start_cent()
    # simulation1.start_fix()
    # simulation1.start_simple()
    simulation1.close_files()
    network1.close_nodes_log_files()

    network4 = Graph('B', 3, 200)
    simulation4 = Simulation(network4)
    #simulation4.start_L_max()
    #simulation4.start_k0()
    simulation4.start()
    #simulation4.start_fix()
    #simulation4.start_cent()
    #simulation4.start_simple()
    simulation4.close_files()
    network4.close_nodes_log_files()

    network2 = Graph('B', 2, 200)
    simulation2 = Simulation(network2)
    #simulation2.start_L_max()
    #simulation2.start_k0()
    simulation2.start()
    #simulation2.start_fix()
    #simulation2.start_cent()
    #simulation2.start_simple()
    simulation2.close_files()
    network2.close_nodes_log_files()

    network3 = Graph('A', 3, 200)
    simulation3 = Simulation(network3)
    #simulation3.start_L_max()
    simulation3.start()
    #simulation3.start_k0()
    #simulation3.start_fix()
    #simulation3.start_cent()
    #simulation3.start_simple()
    simulation3.close_files()
    network3.close_nodes_log_files()




