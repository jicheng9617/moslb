import numpy as np 
import copy


def fast_non_dominated_sorting(population, number_of_functions = 2):
    population = -population
    S     = [[] for i in range(0, population.shape[0])]
    front = [[]]
    n     = [0 for i in range(0, population.shape[0])]
    rank  = [0 for i in range(0, population.shape[0])]
    for p in range(0, population.shape[0]):
        S[p] = []
        n[p] = 0
        for q in range(0, population.shape[0]):
            if ((population[p,-number_of_functions:] <= population[q,-number_of_functions:]).all()):
                if (q not in S[p]):
                    S[p].append(q)
            elif ((population[q,-number_of_functions:] <= population[p,-number_of_functions:]).all()):
                n[p] = n[p] + 1
        if (n[p] == 0):
            rank[p] = 0
            if (p not in front[0]):
                front[0].append(p)
    i = 0
    while (front[i] != []):
        Q = []
        for p in front[i]:
            for q in S[p]:
                n[q] = n[q] - 1
                if(n[q] == 0):
                    rank[q] = i+1
                    if q not in Q:
                        Q.append(q)
        i = i+1
        front.append(Q)
    del front[len(front)-1]
    rank = np.zeros((population.shape[0], 1))
    for i in range(0, len(front)):
        for j in range(0, len(front[i])):
            rank[front[i][j], 0] = i + 1
    return np.where(rank == 1)[0]

def par_non_dominated_sorting(pop: np.ndarray): 

    pop = np.atleast_2d(pop)
    Np = len(pop)
    non_dominate_ind = [] 
    for i in range(Np): 
        p = pop[i] 
        num_d = 0

        for q in pop: 
            if par_dominance(q, p): 
                num_d += 1
                break

        if num_d == 0: non_dominate_ind.append(i)
    return np.array(non_dominate_ind)

def lex_dominance(u: np.ndarray, v: np.ndarray, epsilon=1e-6) -> bool: 
    """
    To judge if u dominates v lexicographically 
    in maximization

    Parameters
    ----------
    u : np.ndarray
        _description_
    v : np.ndarray
        _description_

    Returns
    -------
    bool
        _description_
    """
    for i in range(len(u)): 
        if u[i]-v[i]>epsilon:
            return True
        elif np.abs(u[i]-v[i]) <= epsilon: 
            continue
        else: 
            return False
        
def par_dominance(u: np.ndarray, v: np.ndarray) -> bool: 
    """
    To judge if u dominates v in Pareto form 
    in maximization

    Parameters
    ----------
    u : np.ndarray
        _description_
    v : np.ndarray
        _description_

    Returns
    -------
    bool
        _description_
    """
    for i in range(len(u)): 
        if v[i] > u[i]: return False
    
    if (u==v).all(): return False

    return True
    
def pc_dominance(u:list, v:list) -> bool: 
    """
    Optimality whether u is Pareto-lexicographic
    under priority chains dominated v

    Parameters
    ----------
    u : list
        _description_
    v : list
        _description_

    Returns
    -------
    bool
        _description_
    """
    c = len(u)
    for i in range(c): 
        if not lex_dominance(u[i], v[i]): 
            return False
    return True
    
def pc_non_dominated_sorting(pop: list): 
    c = len(pop)
    Np = pop[0].shape[0] 
    non_dominate_ind = []
    for i in range(Np): 
        p = [pop[j][i] for j in range(c)]
        num_d = 0

        for k in range(Np): 
            q = [pop[j][k] for j in range(c)]
            if pc_dominance(q, p): 
                num_d +=1
                break

        if num_d == 0: non_dominate_ind.append(i)

    return np.array(non_dominate_ind)

def prior_free_lexi_filter(ucb: np.ndarray, lcb: np.ndarray) -> np.ndarray: 
    """
    filter the optimal arms based on transitive closure relation of the linked relation  

    Parameters
    ----------
    ucb : np.ndarray
        upper confidence bound of the arms
    lcb : np.ndarray
        lower confidence bound of the arms

    Returns
    -------
    np.ndarray
        index of the optimal arms
    """
    K,mc = ucb.shape
    opt_ind = [np.arange(K)]

    for i in range(mc): 
        x_i = opt_ind[i][np.argmax(ucb[opt_ind[i], i])]
        opt_ind.append(
            chain_filter(x_i,opt_ind[i],ucb[:, i],lcb[:, i])
        )
    return opt_ind[-1]

def chain_filter(arm, D1, u_t, l_t):
    pre_results = []
    results = [arm]
    lowest = l_t[arm]

    while len(results) != len(pre_results):
        pre_results = copy.deepcopy(results)
        for i in D1:
            if u_t[i] >= lowest and i not in results:
                results.append(i)
                lowest = np.min([lowest, l_t[i]])
    return np.array(results)


if __name__ == "__main__": 

    # x1 = [np.array([1., -1.]), np.array([1., -1.])]

    # x2 = [np.array([1., 1.]), np.random.randn(2)]

    # print(par_lex_dominance(x1, x2))

    # print(x1)

    # print(x2)

    # K = 10

    # pop = [np.random.rand(K,2), np.random.rand(K,2)]

    # print(pop)

    # print(par_lex_non_dominated_sorting(pop))

    # [0.95272219, 0.4606369 ],[0.55505175, 0.65496876],[0.62047633, 0.01251895]
    # [0.40258481, 0.38275378],[0.68383117, 0.5550799 ],[0.61244796, 0.50949707]

    # x1 = np.array([1., 2.2])
    # x2 = np.array([1.1, 2.1])
    # print(par_dominance(x1, x2))

    print()