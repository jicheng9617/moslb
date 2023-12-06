import numpy as np 
import os 
import pickle

from utils import fast_non_dominated_sorting, pc_non_dominated_sorting, par_dominance, pc_dominance, par_non_dominated_sorting

from pprint import pprint

class mo_contextual_bandit: 
    def __init__(
        self, 
        num_obj:int, 
        num_dim:int, 
        num_arm:int, 
        noise:any= np.random.normal, 
        R:int= 1
    ) -> None:
        """
        Environment for multi-objective stochastic contextual bandits

        Parameters
        ----------
        num_obj : int
            number of objectives 
        num_dim : int
            number of dimension of the arms' context
        num_arm : int
            number of arms
        noise : any, optional
            R-sub-Gaussian noise for reward, by default normal distribution
        R : int, optional
            variance proxy in subGaussian, by default 1
        """
        self.m = num_obj
        self.d = num_dim
        self.K = num_arm
        self.noise = noise
        self.R = R

    @property
    def get_num_obj(self) -> int:
        return self.m
    
    @property
    def get_num_arm(self) -> int: 
        return self.K
    
    @property
    def get_num_dim(self) -> int: 
        return self.d
    
    @property
    def get_arms(self) -> np.ndarray: 
        return self.A
    
    def observe_context(self) -> np.ndarray: 
        """
        Output the arms' context at round $t$
        """
        raise NotImplementedError("Subclasses should implement this method.")
    
    def get_reward(self, arm:int) -> float: 
        """
        Get the reward for the arm

        Parameters
        ----------
        arm : int
            index of selected arm

        Returns
        -------
        float
            reward value
        """
        return self.expected_reward(arm) + self.noise(0,self.R,(self.m,))
    
    def expected_reward(self, arm:int) -> np.ndarray: 
        """
        Method to evaluate the unknown reward function
        """
        raise NotImplementedError("Subclasses should implement this method.")



class simulator_moslb(mo_contextual_bandit):
    def __init__(
        self, 
        num_obj:int, 
        num_dim:int, 
        num_arm:int, 
        noise:any= np.random.normal, 
        R:int= 1,
        vary_context:bool= False
    ) -> None:
        """
        Environment for multi-objective stochastic linear bandits

        Parameters
        ----------
        num_obj : int
            number of objectives 
        num_dim : int
            number of dimension of the arms' context
        num_arm : int
            number of arms
        noise : any, optional
            R-sub-Gaussian noise for reward, by default np.random.normal
        R : int, optional
            parameter in noise, by default 1
        vary_context : bool, optional
            whether the contexts at each round are fixed or not
        """
        super().__init__(num_obj,num_dim,num_arm,noise,R)
        self.vary_context = vary_context

    @property
    def get_optimal_arms(self) -> np.ndarray: 
        return self.opt_ind
    
    @property
    def get_expected_rewards(self) -> np.ndarray: 
        return self.y

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

    def expected_reward(self, arm:int) -> np.ndarray: 
        """
        Evaluate the expected reward by the linear model

        Parameters
        ----------
        arm : int
            index of the arm

        Returns
        -------
        np.ndarray
            true reward
        """
        assert isinstance(arm, int)
        return np.dot(self.A[arm], self.th.T)

    def _optimal_arms(self) -> None: 
        """
        Assess the non-dominated arms (optimal arms). 
        """
        self.y = np.vstack([self.expected_reward(i) for i in range(self.get_num_arm)])
        self.opt_ind = par_non_dominated_sorting(self.y)

    def _eval_regret_arm(self, arm:int) -> float: 
        """
        Evaluate the Pareto suboptimal gap for an input arm. 

        Parameters
        ----------
        arm : int
            index of the arm

        Returns
        -------
        float
            Pareto suboptimal gap
        """
        arm_y = self.expected_reward(arm)
        psg = 0
        for j in self.opt_ind: 
            opt_y = self.y[j]
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
            'theta': self.th
                }
        with open(filename, 'wb') as f: 
            pickle.dump(data, f)

    def load_env(self, filename: str=None) -> None: 
        """
        Load the existing environment for MOSLB with Pareto order

        Parameters
        ----------
        filename : str, optional
            file path, by default None. If default, the file in subfolder 'data' named
            ParetoOrder_d_*.pkl will be loaded.
        """
        if filename is None: 
            filename = '.\\data\\ParetoOrder_d_' + str(self.get_num_dim) + '.pkl'
        with open(filename, 'rb') as f: 
            data = pickle.load(f)
        self.th = data['theta']

    def _eval_all_regret(self) -> None: 
        """
        Evaluate the regret for all the arms.
        """
        return np.vstack([self._eval_regret_arm(i) for i in range(self.get_num_arm)])

    def regret(self, arm:int) -> float: 
        """
        Return the regret for the chosen arm.

        Parameters
        ----------
        arm : int
            arm's index

        Returns
        -------
        float (or np.ndarray in pc and pl)
            the regret value(s)
        """
        assert isinstance(arm, int)
        return self._eval_regret_arm(arm)

    def reset(self) -> None: 
        """
        Initialize the environment, i.e., sample the unknown parameters randomly. 
        """
        self._sample_theta()

    def observe_context(self, verbose:bool=False) -> np.ndarray:
        """
        Obtain the arms' context 

        Parameters
        ----------
        verbose : bool, optional
            whether to print the information, by default False

        Returns
        -------
        np.ndarray
            arms' context
        """
        if self.vary_context or not hasattr(self, "A"): 
            self._sample_arms(self.get_num_arm)
            self._optimal_arms()
            if verbose: self.print_info()
        return self.A

    def print_info(self): 
        """
        Print the information of the environment. 
        """
        pprint(
            {'#objective': self.m, 
             '#dimension': self.d, 
             '#arms': self.A.shape[0], 
             '#optimal arms': len(self.opt_ind), 
             'Regret for each arm': self._eval_all_regret()
             }
        )



class simulator_moslb_pl(simulator_moslb):
    def __init__(self, 
        num_dim:int, 
        priority_level:list, 
        num_arm:int, 
        noise:any= np.random.normal, 
        R:float= 1,
        vary_context:bool= False
        ) -> None:
        """
        Environment for multi-objective stochastic linear bandits
        with mixed Pareto-lexicographic order under priority levels

        Parameters
        ----------
        num_dim : int
            number of dimension of the arms' context
        priority_level: list
            describe the MPL-PL relationship between the objectives
        num_arm : int
            number of arms
        noise : any, optional
            R-sub-Gaussian noise for reward, by default np.random.normal
        R : int, optional
            parameter in noise, by default 1
        vary_context : bool, optional
            whether the contexts at each round are fixed or not
        """
        self.pl = priority_level
        self.l = len(self.pl) 
        self.ml = [len(self.pl[i]) for i in range(self.l)]
        self.m = sum(self.ml)
        super().__init__(self.m,num_dim,num_arm,noise,R,vary_context)

    def _optimal_arms(self) -> None: 
        """
        Assess the optimal arms 
        """
        self.y = np.vstack([self.expected_reward(i) for i in range(self.get_num_arm)])
        opt_ind = [np.arange(self.get_num_arm)]
        for i in range(self.l): 
            opt_ind.append(
                opt_ind[-1][par_non_dominated_sorting(self.y[opt_ind[-1]][:,self.pl[i]])]
            )
        self.opt_ind = opt_ind[1:]

    def _eval_regret_arm(self, arm:int) -> np.ndarray:
        """
        Evaluate the regret for an arm under MPL-PL 

        Parameters
        ----------
        arm : int
            index for the arm

        Returns
        -------
        np.ndarray
            PL regret with length of l
        """
        reg = np.zeros((self.l,))
        arm_y = self.expected_reward(arm)

        for i in range(self.l): 
            delta_x = 0
            for j in self.opt_ind[i]:
                opt_y = self.y[j]
                if par_dominance(opt_y[self.pl[i]], arm_y[self.pl[i]]): 
                    tmp = np.min(opt_y[self.pl[i]]- arm_y[self.pl[i]])
                    if tmp > delta_x: delta_x = tmp
            if delta_x > 0: 
                reg[i] = delta_x
                break
        return reg
    
    def save_env(self, filename: str=None) -> None:
        """
        Save the environment to the "data" folder

        Parameters
        ----------
        filename : str, optional
            saving path, by default None
        """
        if filename is None: 
            filename = '.\\data\\pl_' + '-'.join(str(e) for e in self.ml) + \
                '_d_' + str(self.get_num_dim) + '.pickle'
        data = {
            'theta': self.th, 
            'priority_level': self.pl
                }
        with open(filename, 'wb') as f: 
            pickle.dump(data, f)

    def load_env(self, filename: str=None) -> None:
        """
        Load the existing environment for MOSLB with MPL-PL order

        Parameters
        ----------
        filename : str, optional
            file path, by default None. If default, the file in subfolder 'data' named
            pl_*_d_*.pkl will be loaded. 
        """
        if filename is None: 
            filename = '.\\data\\pl_' + '-'.join(str(e) for e in self.ml) + \
                '_d_' + str(self.get_num_dim) + '.pickle'
        with open(filename, 'rb') as f: 
            data = pickle.load(f)
        self.th = data['theta']
        self.pl = data['priority_level']



class simulator_moslb_pc(simulator_moslb): 
    def __init__(
        self, 
        num_dim, 
        priority_chain, 
        num_arm:int, 
        noise:any= np.random.normal, 
        R:float= 1,
        vary_context:bool= False
        ) -> None:
        """
        Environment for multi-objective stochastic linear bandits
        with mixed Pareto-lexicographic order under priority chains

        Parameters
        ----------
        num_dim : int
            number of dimension of the arms' context
        priority_chain: list
            describe the MPL-PC relationship between the objectives
        num_arm : int
            number of arms
        noise : any, optional
            R-sub-Gaussian noise for reward, by default np.random.normal
        R : int, optional
            parameter in noise, by default 1
        vary_context : bool, optional
            whether the contexts at each round are fixed or not
        """
        self.d = num_dim
        self.pc = priority_chain
        self.c = len(self.pc) 
        self.mc = [len(self.pc[i]) for i in range(self.c)]
        self.c_max = np.max(self.mc)
        self.num_obj = sum(self.mc)
        super().__init__(self.num_obj, num_dim, num_arm, noise, R)

    def expected_reward(self, arm:int) -> np.ndarray: 
        """
        Evaluate the expected reward by the linear model, the reward is rounded 
        within two decimal to increase the number of optimal arms

        Parameters
        ----------
        arm : int
            index of the arm

        Returns
        -------
        np.ndarray
            true reward
        """
        assert isinstance(arm, int)
        return np.dot(self.A[arm], self.th.T).round(decimals=2)

    def _optimal_arms(self) -> None: 
        """
        Access the optimal arms under MPL-PC order 
        """
        self.y = np.vstack([self.expected_reward(i) for i in range(self.get_num_arm)])
        tmp_y = [self.y[:,self.pc[i]].reshape(self.get_num_arm,-1) for i in range(self.c)]
        self.opt_ind = pc_non_dominated_sorting(tmp_y)

    def _eval_regret_arm(self, arm:int) -> np.ndarray:
        """
        Evaluate the MPL-PC regret for an arm

        Parameters
        ----------
        arm : int
            arm's index

        Returns
        -------
        np.ndarray
            PC regret for the arm
        """
        arm_y = [self.expected_reward(arm)[i] for i in self.pc]
        reg = []

        for i in self.opt_ind:
            opt_y = [self.y[i][j] for j in self.pc]
            if pc_dominance(opt_y, arm_y):
                tmp_reg = np.zeros((self.c, self.c_max))
                for j in range(self.c): 
                    for k in range(self.mc[j]):
                        tmp_reg[j][k] = np.maximum(0, opt_y[j][k]-arm_y[j][k])
                ind = np.lexsort([tmp_reg[:, o] for o in reversed(range(self.c_max))])[0]
                reg.append(tmp_reg[ind])

        if len(reg) == 0: 
            return np.zeros((self.c_max))
        elif len(reg) == 1: 
            return reg[0]
        else: 
            reg = np.vstack(reg)
            ind = np.lexsort([reg[:, o] for o in reversed(range(self.c_max))])[0]
            return reg[ind]
        
    def save_env(self, filename: str=None) -> None:
        """
        Save the environment to the "data" folder

        Parameters
        ----------
        filename : str, optional
            saving path, by default None
        """
        if filename is None: 
            filename = '.\\data\\pc_' + '-'.join(str(e) for e in self.mc) + \
                '_d_' + str(self.get_num_dim) + '.pkl'
        data = {
            'theta': self.th, 
            'priority_chain': self.pc
                }
        with open(filename, 'wb') as f: 
            pickle.dump(data, f)

    def load_env(self, filename: str=None) -> None:
        """
        Load the existing environment for MOSLB with MPL-PC order

        Parameters
        ----------
        filename : str, optional
            file path, by default None. If default, the file in subfolder 'data' named
            pc_*_d_*.pkl will be loaded. 
        """
        if filename is None: 
            filename = '.\\data\\pc_' + '-'.join(str(e) for e in self.mc) + \
                '_d_' + str(self.get_num_dim) + '.pkl'
        with open(filename, 'rb') as f: 
            data = pickle.load(f)
        self.th = data['theta']
        self.pc = data['priority_chain']



if __name__ == "__main__": 

    num_obj = 3 
    num_dim = 5
    ###############################################
    ########### example for moslb env #############
    ###############################################
    env = simulator_moslb_pl(num_dim=num_dim,priority_level=[[0,1,2],[3,4]])
    # env.reset()
    # env.observe_context(num_arms=20,verbose=1)
    # env.save_env()

    env.load_env()
    env.observe_context(num_arms=20,verbose=1)

    # env.load_env(verbose=1)
