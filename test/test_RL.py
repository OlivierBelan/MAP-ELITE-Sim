import sys
sys.path.append('../')
sys.path.append('../src/')
sys.path.append('../src/snn_simulator/')
sys.path.append('../src/evo_simulator/')
import os
os.environ["RAY_DEDUP_LOGS"] = "0"

import numpy as np
from evo_simulator.GENERAL.Neuro_Evolution import Neuro_Evolution

from problem.RL.REINFORCEMENT import Reinforcement_Manager


from evo_simulator.ALGORITHMS.Algorithm import Algorithm

# Algorithms Mono-Objective
from evo_simulator.ALGORITHMS.MAP_ELITE.MAP_ELITE import MAP_ELITE
from evo_simulator.ALGORITHMS.NSLC.NSLC import NSLC


# QD Gym
from RL_problems_config.QDHalfCheetah import QDHalfCheetah
from RL_problems_config.QDAnt import QDAnt
from RL_problems_config.QDHopper import QDHopper
from RL_problems_config.QDWalker2D import QDWalker2D
from RL_problems_config.QDHumanoid import QDHumanoid


from typing import List, Dict, Tuple
np.set_printoptions(threshold=sys.maxsize)




# Algo Multi-Objective
def map_elite(config_path) -> Tuple[Neuro_Evolution, str]:
    # 1 - Config path file
    local_dir = os.path.dirname(__file__)
    neat_config_path = os.path.join(local_dir, start_config_path + config_path)
    
    return "MAPELITE", MAP_ELITE, neat_config_path
   
def nslc(config_path) -> Tuple[Neuro_Evolution, str]:
    # 1 - Config path file
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, start_config_path + config_path)
    
    return "NSLC", NSLC, config_path


start_config_path = "./config/config_snn/RL/"
# start_config_path = "./config/config_ann/RL/"

def neuro_evo_matrix_func(args:List[str]):
    if len(args) == 0:raise Exception("Error: No arguments")

    aglos_dict:Dict[str, Tuple[Neuro_Evolution, str]] = {
        # 1.2 - Algorithms Multi-Objective
        "MAP_ELITE":      map_elite("MAP_ELITE_CONFIG_RL.cfg"),
        "NSLC":           nslc("NSLC_CONFIG_RL.cfg"),
        
    }
    algos = [aglos_dict[arg] for arg in args if arg in aglos_dict]
    nb_runs:int = 3
    nb_episode:int = 1
    nb_generation:int = 100
    max_seeds:int = 100_000
    seeds = []
    for _ in range(nb_runs): seeds.append(np.random.choice(np.arange(max_seeds), size=nb_episode, replace=False))
    seeds = np.array(seeds)
    print("seeds: ", seeds)

    for name, algorithm, config_path in algos:
        # 2 - Environnement


        # 2.4 - QD Gym
        environnement:QDHalfCheetah = QDHalfCheetah("QDHalfCheetah", config_path, nb_input=26, nb_output=6, obs_max_init_value=5, obs_min_init_value=-5, auto_obersvation=True)
        # environnement:QDAnt = QDAnt("QDAnt", config_path, nb_input=28, nb_output=8, obs_max_init_value=5, obs_min_init_value=-5, auto_obersvation=True)
        # environnement:QDHopper = QDHopper("QDHopper", config_path, nb_input=15, nb_output=3, obs_max_init_value=5, obs_min_init_value=-5, auto_obersvation=True)
        # environnement:QDWalker2D = QDWalker2D("QDWalker", config_path, nb_input=22, nb_output=6, obs_max_init_value=5, obs_min_init_value=-5, auto_obersvation=True)
        # environnement:QDHumanoid = QDHumanoid("QDHumanoid", config_path, nb_input=44, nb_output=17, obs_max_init_value=5, obs_min_init_value=-5, auto_obersvation=True)

        # 3 - Reinforcement Manager -> Run
        neuro:Neuro_Evolution = Neuro_Evolution(nb_generations=nb_generation, nb_runs=nb_runs, is_record=True, config_path=config_path, cpu=15)
        neuro.init_algorithm(name, algorithm, config_path)

        # If you want to run QD Gym uncomment the following line and comment the following line (neuro.run_rastrigin)
        # neuro.init_problem_RL(Reinforcement_Manager, config_path, environnement, nb_episode=nb_episode, seeds=seeds, render=False)
        # neuro.run()

        # If you want to run rasstrigin uncomment the following line and comment the following line (neuro.init_problem_RL & neuro.run)
        neuro.run_rastrigin()

def main():
    neuro_evo_matrix_func(sys.argv[1:])

if __name__ == "__main__":
    main()
