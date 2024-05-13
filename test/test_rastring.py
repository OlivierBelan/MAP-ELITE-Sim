import sys
sys.path.append('../')
sys.path.append('../src/')
sys.path.append('../src/snn_simulator/')
sys.path.append('../src/evo_simulator/')
import os
os.environ["RAY_DEDUP_LOGS"] = "0"

import numpy as np
from evo_simulator.GENERAL.Neuro_Evolution import Neuro_Evolution
from evo_simulator.GENERAL.Genome import Genome_Classic
from evo_simulator.GENERAL.Population import Population
# Algorithms Mono-Objective
from evo_simulator.ALGORITHMS.MAP_ELITE.MAP_ELITE import MAP_ELITE_rastigin
from evo_simulator.ALGORITHMS.NSLC.NSLC import NSLC

import math
from typing import List, Dict, Tuple
np.set_printoptions(threshold=sys.maxsize)

def test_rastrigin(self, population_manager:Population) -> None:
    population_dict:Dict[int, Genome_Classic] = population_manager.population
    for genome in population_dict.values():
        genome.fitness.score, genome.info["description"] = rastrigin_map_elite(genome.parameter)

def rastrigin_map_elite(self, xx:np.ndarray) -> Tuple[float, np.ndarray]:
    x = xx * 10 - 5 # scaling to [-5, 5]
    f = 10 * x.shape[0] + (x * x - 10 * np.cos(2 * math.pi * x)).sum()
    return -f, np.array([xx[0], xx[1]])



# Algo Multi-Objective
def map_elite(config_path) -> Tuple[Neuro_Evolution, str]:
    # 1 - Config path file
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, start_config_path + config_path)
    
    return "MAPELITE", MAP_ELITE_rastigin, config_path
   
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

        # 3 - Reinforcement Manager -> Run
        neuro:Neuro_Evolution = Neuro_Evolution(nb_generations=nb_generation, nb_runs=nb_runs, is_record=True, config_path=config_path, cpu=1)
        neuro.init_algorithm(name, algorithm, config_path)
        # neuro.init_problem_RL(Reinforcement_Manager, config_path, environnement, nb_episode=nb_episode, seeds=seeds, render=False)
        neuro.run_rastigin()

def main():
    neuro_evo_matrix_func(sys.argv[1:])

if __name__ == "__main__":
    main()
