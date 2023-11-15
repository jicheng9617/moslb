# Multi-objective Stochastic Contextual Bandits

Code for AAAI2024 Paper: Hierarchize Pareto Dominance in Multi-objective Stochastic Linear Bandits

The repository contains: 
- <code>oracle.py</code> contains simulators for multi-objective stochastic linear bandits; to apply to real-world dataset, rewrite methods *observe_context* and *expected_reward* for subclass of the base class *mo_contextual_bandit*.
- <code>moslb.py</code> provides bandit algorithms, including ParetoUCB, MOSLB-PC, and MOSLB-PL; one can follow the implementation in the **example.ipynb** for quick start.
- <code>utils.py</code> contains basic functions to judge the optimality under different preference. 

## Reference

If you find our work helpful, please consider citing our paper:
```
@inproceedings
```
