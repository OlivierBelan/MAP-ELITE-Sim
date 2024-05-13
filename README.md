# MAP-ELITE-SIM

Multi-dimensional Archive of Phenotypic Elites (MAP-Elites)

A more detailed README will be added soon.

The algorithm is base on [MAP-ELITE paper](https://arxiv.org/abs/1504.04909) by Jean-Baptiste Mouret and Jeff Clune.

To test the program, execute either `python test_RL.py MAP-ELITE` from within the test folder. Additional options are available in the test_RL.py files, as well as within the configuration files located in the config folder.

To facilitate comparisons, an ANN runner built with PyTorch is also available. To use it, uncomment the line start_config_path = "./config/config_ann/RL/" in test_RL.py, and comment out the corresponding SNN configuration line start_config_path = "./config/config_snn/RL/"

A more complete version with more algorithms and more examples is also available at: https://github.com/OlivierBelan/Evo-Sim (mainly NeuroEvolution algorithms)


During the installation/run of pybullet env (QD_env), if you encounter the following error:
```AttributeError: 'dict' object has no attribute 'env_specs'```
You can fix it by changing the file ```pybullet_envs/__init__.py``` the section:
```python
def register(id, *args, **kvargs):
  if id in registry.env_specs:
    return
  else:
    return gym.envs.registration.register(id, *args, **kvargs)
```
to:
```python
def register(id, *args, **kvargs):
  if id in registry:
    return
  else:
    return gym.register(id, *args, **kvargs)
```