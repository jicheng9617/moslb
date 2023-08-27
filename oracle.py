import numpy as np 
import os 
import pickle

from utils import fast_non_dominated_sorting, pc_non_dominated_sorting, par_dominance, pc_dominance, par_non_dominated_sorting

from pprint import pprint

class env_moslb:
    def __init__(
        self, 
        num_obj:int, 
        num_dim:int, 
        noise:any=np.random.normal, 
        R:int=1
    ) -> None:
        """
        Environment for multi-objective stochastic linear bandits

        Parameters
        ----------
        num_obj : int
            number of objectives 
        num_dim : int
            number of dimension of the arms' context
        noise : any, optional
            R-sub-Gaussian noise for reward, by default np.random.normal
        R : int, optional
            parameter in noise, by default 1
        """
        self.m = num_obj
        self.d = num_dim
        self.noise = noise
        self.R = R

    @property
    def get_num_obj(self) -> int:
        return self.m
    
    @property
    def get_num_arm(self) -> int: 
        return self.A.shape[0]
    
    @property
    def get_num_dim(self) -> int: 
        return self.d
    
    @property
    def get_arms(self) -> np.ndarray: 
        return self.A

    def _sample_theta(self) -> None: 
        """
        Generate the unknown parameters from unit sphere
        """
        theta = np.zeros(shape=(self.m,self.d))
        for j in range(self.m): 
            temp = np.random.normal(size=self.d)
            temp /= np.linalg.norm(temp) 
            rad = np.random.uniform() ** (1 / self.d) 
            theta[j] = temp * rad
        self.th = theta
    
    def _sample_arms(self, num_arms:int) -> None:
        """
        Generate arms' context randomly within unit sphere.

        Parameters
        ----------
        num_arms : int
            number of arms
        """
        self.A = np.zeros(shape=(num_arms, self.d))
        for i in range(num_arms): 
            point = np.random.normal(size=self.d)
            point /= np.linalg.norm(point)
            rad = np.random.uniform() ** (1 / self.d)
            self.A[i] = rad * point

    def _expected_reward(self) -> None: 
        """
        Evaluate the expected loss for the arms.
        """
        self.y = np.matmul(self.A, self.th.T)

    def _optimal_arms(self) -> None: 
        """
        Assess the non-dominated arms (optimal arms). 
        """
        self.opt_ind = par_non_dominated_sorting(self.y)
        self.opt_arm = self.A[self.opt_ind] 
        self.opt_y = self.y[self.opt_ind]

    def _eval_regret_arm(self, arm: np.ndarray) -> float: 
        """
        Evaluate the Pareto suboptimal gap for an input arm. 

        Parameters
        ----------
        arm : np.ndarray
            contextual information for the arm

        Returns
        -------
        float
            Pareto suboptimal gap
        """
        arm_y = np.matmul(arm, self.th.T)
        psg = 0
        for j in range(len(self.opt_ind)): 
            opt_y = self.opt_y[j]
            if par_dominance(opt_y, arm_y): 
                tmp = np.min(opt_y-arm_y)
                if tmp > psg: psg = tmp
        return psg
    
    def save_env(self, filename:str=None) -> None: 
        """
        Save the environment to the "data" folder

        Parameters
        ----------
        filename : str, optional
            saving path, by default None
        """
        if filename is None: 
            if not os.path.exists('.\\data\\'): os.makedirs('.\\data\\')
            filename = '.\\data\\ParetoOrder_d_' + str(self.get_num_dim) + '.pkl'
        data = {
            'theta': self.th, 
            'arms': self.A
                }
        with open(filename, 'wb') as f: 
            pickle.dump(data, f)

    def load_env(self, filename: str=None, verbose:bool=False) -> None: 
        """
        Load the existing environment for MOSLB with Pareto order

        Parameters
        ----------
        filename : str, optional
            file path, by default None. If default, the file in subfolder 'data' named
            ParetoOrder_d_*.pkl will be loaded. 
        verbose : bool, optional
            whether to print the information, by default False
        """
        if filename is None: 
            filename = '.\\data\\ParetoOrder_d_' + str(self.get_num_dim) + '.pkl'
        with open(filename, 'rb') as f: 
            data = pickle.load(f)
        self.th = data['theta']
        self.A = data['arms']
        # initialize the parameters 
        self._expected_reward()
        self._optimal_arms()
        self._eval_regret()
        if verbose: self.print_info()

    def _eval_regret(self) -> None: 
        """
        Evaluate the regret for all the arms.
        """
        self.reg = np.vstack([self._eval_regret_arm(self.A[i]) for i in range(self.get_num_arm)])

    def regret(self, arm: np.ndarray) -> float: 
        """
        Return the regret for the chosen arm.

        Parameters
        ----------
        arm : np.ndarray
            arm context or index

        Returns
        -------
        float (or np.ndarray in pc and pl)
            the regret value(s)
        """
        if not isinstance(arm, int): 
            ind = np.where(np.isclose(self.A, arm).all(axis=1))
        else: 
            ind = arm
        return self.reg[ind]

    def reset(self, num_arms:int, verbose:bool=False) -> None: 
        """
        Initialize the environment and sample the unknown parameters randomly. 

        Parameters
        ----------
        num_arms : int
            number of the arms 
        verbose : bool, optional
            whether to print the information, by default False
        """
        self._sample_theta()
        self._sample_arms(num_arms)
        self._expected_reward()
        self._optimal_arms()
        self._eval_regret()
        if verbose: self.print_info()

    def print_info(self): 
        """
        Print the information of the environment. 
        """
        pprint(
            {'#objective': self.m, 
             '#dimension': self.d, 
             '#arms': self.A.shape[0], 
             '#optimal arms': len(self.opt_ind), 
             'Regret for each arm': self.reg
             }
        )



class env_moslb_pl(env_moslb):

    def __init__(self, num_dim, priority_level, noise=np.random.normal, R=1) -> None:

        self.pl = priority_level
        self.l = len(self.pl) 
        self.ml = [len(self.pl[i]) for i in range(self.l)]

        self.num_obj = sum(self.ml)
        super().__init__(self.num_obj, num_dim, noise, R)
        
        self.noise = noise
        self.R = R

    def _assert_input(self) -> None: 

        pass

    def _sample_theta(self) -> list: 

        self.th = [] 
        for i in range(self.l): 
            theta = np.zeros(shape=(self.ml[i],self.d))
            for j in range(self.ml[i]): 
                temp = np.random.normal(size=self.d)
                temp /= np.linalg.norm(temp) 
                rad = np.random.uniform() ** (1 / self.d) 
                theta[j] = temp * rad
            self.th.append(theta)
    
    def _expected_reward(self) -> list: 

        self.y = [np.dot(self.A, self.th[i].T) for i in range(self.l)]

    def _optimal_arms(self) -> list: 

        opt_arm = []
        opt_y = []
        opt_ind = []
        tmp_ind = fast_non_dominated_sorting(self.y[0], self.ml[0])
        opt_ind.append(tmp_ind)

        for i in range(self.l-1): 
            tmp_ind = fast_non_dominated_sorting(self.y[i+1][opt_ind[i]], self.ml[i+1])
            opt_ind.append(opt_ind[-1][tmp_ind])

        for i in range(self.l): 
            opt_arm.append(self.A[opt_ind[i]])
            opt_y.append(self.y[i][opt_ind[i]])

        self.opt_ind, self.opt_arm, self.opt_y = opt_ind, opt_arm, opt_y

    def get_reward(self, arm) -> np.ndarray: 
        """Get reward 

        Returns
        -------
        np.ndarray
            _description_
        """
        res = [] 
        for i in range(self.l): 
            res.append(self.y[i][arm] + self.noise(0,self.R,(self.ml[i],)))
        return res
    
    def print_info(self): 

        pprint(
            {'#objective': self.m, 
             '#dimension': self.d, 
             '#arms': self.A.shape[0], 
             'priority level': self.pl, 
             '#optimal arms': [len(self.opt_ind[i]) for i in range(self.l)], 
             'Regret for each arm': self.reg
             }
        )

    def reset(self, num_arms, verbose=False) -> None: 

        self._sample_theta()
        self._sample_arms(num_arms)
        self._expected_reward()
        self._optimal_arms()
        self._eval_regret()
        if verbose: self.print_info()

    def _eval_regret_arm(self, arm) -> np.ndarray:

        arm_y = [np.dot(arm, self.th[i].T) for i in range(self.l)]

        reg = np.zeros((self.l,))
        for i in range(self.l): 
            delta_x = 0
            for j in range(len(self.opt_ind[i])): 
                opt_y = self.opt_y[i][j]
                if par_dominance(opt_y, arm_y[i]): 
                    tmp = np.min(opt_y-arm_y[i])
                    if tmp > delta_x: delta_x = tmp
            if delta_x > 0: 
                reg[i] = delta_x
                break

        return reg
    
    def save_env(self, filename: str=None) -> None:

        if filename is None: 
            filename = '.\\data\\pl_' + '-'.join(str(e) for e in self.ml) + \
                '_d_' + str(self.get_num_dim) + '.pickle'
        data = {
            'theta': self.th, 
            'arms': self.A
                }
        with open(filename, 'wb') as f: 
            pickle.dump(data, f)

    def load_env(self, filename: str=None, verbose=False) -> None:

        if filename is None: 
            filename = '.\\data\\pl_' + '-'.join(str(e) for e in self.ml) + \
                '_d_' + str(self.get_num_dim) + '.pickle'
        with open(filename, 'rb') as f: 
            data = pickle.load(f)
        self.th = data['theta']
        self.A = data['arms']

        self._expected_reward()
        self._optimal_arms()
        self._eval_regret()
        if verbose: self.print_info()




class env_moslb_pc(env_moslb): 

    def __init__(self, num_dim, priority_chain, noise=np.random.normal, R=1) -> None:

        self.d = num_dim
        self.pc = priority_chain
        self.c = len(self.pc) 
        self.mc = [len(self.pc[i]) for i in range(self.c)]

        self.num_obj = sum(self.mc)
        super().__init__(self.num_obj, num_dim, noise, R)

        self.noise = noise
        self.R = R

    def _sample_theta(self) -> list: 

        self.th = [] 
        for i in range(self.c): 
            theta = np.zeros(shape=(self.mc[i],self.d))
            for j in range(self.mc[i]): 
                temp = np.random.normal(size=self.d)
                temp /= np.linalg.norm(temp) 
                rad = np.random.uniform() ** (1 / self.d) 
                theta[j] = temp * rad
            self.th.append(theta)

    def _sample_arms(self, num_arms) -> np.ndarray:

        n_subopt = self.d
        A_part1 = np.zeros(shape=(num_arms-n_subopt, self.d))
        A_part2 = []

        for i in range(num_arms-n_subopt): 
            point = np.random.normal(size=self.d)
            point /= np.linalg.norm(point)
            rad = np.random.uniform() ** (1 / self.d)
            A_part1[i] = rad * point

        y_part1 = [np.matmul(A_part1, self.th[i].T).round(decimals=2) for i in range(self.c)]
        non_dom_ind = pc_non_dominated_sorting(y_part1)
        y_opt = [y_part1[i][non_dom_ind] for i in range(self.c)]
        num_opt = len(non_dom_ind) 

        index = 0
        n_subopt1 = np.random.choice(n_subopt)
        while index < n_subopt1: 
            id = np.random.choice(num_opt)
            point = np.random.normal(size=self.d)
            point /= np.linalg.norm(point)
            rad = np.random.uniform() ** (1 / self.d)
            arm = rad * point
            arm_y = [np.dot(arm, self.th[j].T).round(decimals=2) for j in range(self.c)]
            if arm_y[0][0] == y_opt[0][id,0] and arm_y[1][0] == y_opt[1][id,0]: 
                A_part2.append(arm) 
                index += 1
                print("First form suboptimal arm {} generated.".format(index))

        while index < n_subopt: 
            id = np.random.choice(num_opt)
            point = np.random.normal(size=self.d)
            point /= np.linalg.norm(point)
            rad = np.random.uniform() ** (1 / self.d)
            arm = rad * point
            arm_y = [np.dot(arm, self.th[j].T).round(decimals=2) for j in range(self.c)]
            if arm_y[0][0] == y_opt[0][id,0] and arm_y[1][0] == y_opt[1][id,0] \
                and arm_y[1][1] == y_opt[1][id,1]: 
                A_part2.append(arm)  
                index += 1
                print("Second form suboptimal arm {} generated.".format(index))

        self.A = np.append(A_part1, np.vstack(A_part2), axis=0)

    def _expected_reward(self) -> list: 

        self.y = [np.dot(self.A, self.th[i].T).round(decimals=2) for i in range(self.c)]

    def _optimal_arms(self) -> list: 

        self.opt_ind = []
        self.opt_ind = pc_non_dominated_sorting(self.y)
        self.opt_arm = self.A[self.opt_ind]
        self.opt_y = [self.y[i][self.opt_ind] for i in range(self.c)]

    def get_reward(self, arm) -> np.ndarray: 
        """Get reward 

        Returns
        -------
        np.ndarray
            _description_
        """
        res = [] 
        for i in range(self.c): 
            # res.append(np.dot(arm, self.th[i].T) + self.noise(0,self.R,(self.mc[i],)))
            res.append(self.y[i][arm] + self.noise(0,self.R,(self.mc[i],)))
        return res
    
    def print_info(self): 

        pprint(
            {'#objective': self.m, 
             '#dimension': self.d, 
             '#arms': self.A.shape[0], 
             'priority chain': self.pc, 
             '#optimal arms': self.opt_ind, 
             'Regret for each arm': self.reg
             }
        )

    def reset(self, num_arms, verbose=False) -> None: 

        self._sample_theta()
        self._sample_arms(num_arms)
        self._expected_reward()
        self._optimal_arms()
        self._eval_regret()
        if verbose: self.print_info()

    def _eval_regret_arm(self, arm) -> np.ndarray:

        arm_y = [np.dot(arm, self.th[i].T).round(decimals=2) for i in range(self.c)]
        c_max = np.max(self.mc)
        reg = []

        for i in range(len(self.opt_ind)):
            opt_y = [self.opt_y[j][i] for j in range(self.c)]
            if pc_dominance(opt_y, arm_y):
                tmp_reg = np.zeros((self.c, c_max))
                for j in range(self.c): 
                    for k in range(self.mc[j]):
                        tmp_reg[j][k] = np.maximum(0, opt_y[j][k]-arm_y[j][k])
                ind = np.lexsort([tmp_reg[:, o] for o in reversed(range(c_max))])[0]
                reg.append(tmp_reg[ind])

        if len(reg) == 0: 
            return np.zeros((c_max))
        elif len(reg) == 1: 
            return reg[0]
        else: 
            reg = np.vstack(reg)
            ind = np.lexsort([reg[:, o] for o in reversed(range(c_max))])[0]
            return reg[ind]
        
    def save_env(self, filename: str=None) -> None:

        if filename is None: 
            filename = '.\\data\\pc_' + '-'.join(str(e) for e in self.mc) + \
                '_d_' + str(self.get_num_dim) + '.pkl'
        data = {
            'theta': self.th, 
            'arms': self.A
                }
        with open(filename, 'wb') as f: 
            pickle.dump(data, f)

    def load_env(self, filename: str=None, verbose=False) -> None:

        if filename is None: 
            filename = '.\\data\\pc_' + '-'.join(str(e) for e in self.mc) + \
                '_d_' + str(self.get_num_dim) + '.pkl'
        with open(filename, 'rb') as f: 
            data = pickle.load(f)
        self.th = data['theta']
        self.A = data['arms']

        self._expected_reward()
        self._optimal_arms()
        self._eval_regret()
        if verbose: self.print_info()



if __name__ == "__main__": 

    num_obj = 3 
    num_dim = 5

    # env = env_moslb_pl(num_obj, num_dim, priority_level=[[0,1], [2,3]])

    # env.reset(num_arms=5*num_dim)

    # env.save_env(".\\data\\test.pickle")

    # filename = ".\\data\\test.pickle" 

    # with open(filename, 'rb') as f: 
    #     env = pickle.load(f)

    # env.print_info()

    # for i in range(env.get_num_arm):
    #     print(env.eval_regret(env.A[i]))

    # env = env_moslb_pc(num_dim, priority_chain=[[0,1], [2,3,4]])

    # env.reset(5*num_dim, verbose=0)

    # print(env.opt_y)

    # env.save_env()

    # env.load_env(verbose=1)

    env = env_moslb(num_obj, num_dim)

    env.reset(5*num_dim, verbose=1)
