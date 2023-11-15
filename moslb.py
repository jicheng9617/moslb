import numpy as np 

from utils import par_non_dominated_sorting, prior_free_lexi_filter

class moslb:
    def __init__(self, 
        num_dim:int, 
        num_obj:int, 
        lamda:float=1.,
        delta:float=.05
        ) -> None:
        """
        Multi-objective stochastic linear bandit
        UCB algorithm considering Pareto order

        Parameters
        ----------
        num_dim : int
            number of context's dimension
        num_obj : int
            number of objectives
        lamda : float, optional
            regularization term, lambda, by default 1.0
        delta : float, optional 
            confidence level value, by default 0.05
        """
        self.d = num_dim
        self.m = num_obj
        self.lamda = lamda
        self.delta = delta

    @property
    def get_num_obj(self) -> int:
        return self.m
    
    @property
    def get_num_dim(self) -> int: 
        return self.d
    
    @property
    def get_optimal_arms(self) -> np.ndarray: 
        return self.opt_ind

    def reset(self) -> None: 
        """
        Initialize the parameters 
        """
        self.t = 1
        self.X_list = [] 
        self.Y_list = [] 
        self.V = self.lamda * np.eye(self.d) 
        self.theta = np.zeros((self.m, self.d))
        self.V_inv = 1/self.lamda * self.V 

    def estimate_reward(self, arm:np.ndarray) -> float: 
        """
        Estimate the expected reward for an arm

        Parameters
        ----------
        arm : np.ndarray
            arm's context

        Returns
        -------
        float
            estimated reward
        """
        assert arm.ndim == 1
        return np.dot(arm, self.theta.T) 
    
    def estimate_uncertainty(self, arm:np.ndarray) -> float: 
        """
        Estimate the width of confidence level for an arm

        Parameters
        ----------
        arm : np.ndarray
            arm's context

        Returns
        -------
        float
            estimated variance
        """
        assert arm.ndim == 1
        gamma_t = np.sqrt(self.d * np.log(self.m * (1 + self.t) / self.delta)) + 1
        w_t = gamma_t * np.sqrt(np.dot(arm, self.V_inv).dot(arm))
        return w_t

    def _eval_ucb(self, arm:np.ndarray, alpha: float=1.) -> np.ndarray: 
        """
        Evaluate the upper confidence bound for an arm

        Returns
        -------
        np.ndarray
            upper confidence bound of the estimated reward
        """
        return self.estimate_reward(arm) + alpha*self.estimate_uncertainty(arm)

    def _eval_lcb(self, arm:np.ndarray, alpha: float=1.) -> np.ndarray: 
        """
        Evaluate the lower confidence bound for an arm

        Returns
        -------
        np.ndarray
            lower confidence bound of the estimated reward
        """
        return self.estimate_reward(arm) - alpha*self.estimate_uncertainty(arm)

    def take_action(self, 
        arm:np.ndarray, 
        alpha: float=1.) -> int: 
        """
        Take an action based on P-UCB algorithm

        Parameters
        ----------
        arm : np.ndarray
            arms' context
        alpha : float, optional
            parameter to control the uncertainty level, by default 1.

        Returns
        -------
        int
            index of the selected arm
        """
        arm = np.atleast_2d(arm)
        ucb = np.vstack([self._eval_ucb(arm=arm[i],alpha=alpha) for i in range(arm.shape[0])])
        self.opt_ind = par_non_dominated_sorting(ucb)
        return np.random.choice(self.opt_ind, size=1).item()

    def update_params(self, arm: np.ndarray, reward: np.ndarray) -> None: 
        """
        Update the parameters

        Parameters
        ----------
        arm : np.ndarray
            context of the selected arm
        reward : np.ndarray
            observed reward of the arm
        """
        self.t += 1
        self.X_list.append(arm) 
        self.Y_list.append(reward)
        X = np.vstack(self.X_list) 
        Y = np.vstack(self.Y_list) 
        self.V += np.outer(arm, arm)
        self.V_inv = np.linalg.inv(self.V) 
        for i in range(self.m): 
            self.theta[i] = self.V_inv @ X.T @ Y[:, i]



class moslb_pl(moslb): 
    def __init__(
        self, 
        num_dim:int, 
        priority_level:list, 
        lamda:float=1.,
        delta:float=.05
        ) -> None:
        """
        Multi-objective stochastic linear bandit algorithm 
        considering MPL-PL order

        Parameters
        ----------
        num_dim : int
            number of context's dimension
        priority_level : list
            list of objective index representing priority relation
        lamda : float, optional
            regularization term, lambda, by default 1.0
        delta : float, optional 
            confidence level value, by default 0.05
        """
        self.d = num_dim 
        self.pl = priority_level
        self.l = len(self.pl)
        self.ml = [len(self.pl[i]) for i in range(self.l)]
        self.m = sum(self.ml)
        super().__init__(num_dim=num_dim, num_obj=self.m, lamda=lamda, delta=delta)
    
    @property
    def get_optimal_index(self) -> np.ndarray: 
        if hasattr(self, 'opt_ind'): return self.opt_ind

    def take_action(
        self, 
        arm:np.ndarray, 
        epsilon: float,
        alpha: float=1.
        ) -> int: 
        """
        Take an action

        Parameters
        ----------
        arm : np.ndarray
            arms' context
        alpha : float, optional
            parameter to control the uncertainty level, by default 1.

        Returns
        -------
        int
            index of the selected arm
        """
        arm = np.atleast_2d(arm)
        w_t = np.vstack([self.estimate_uncertainty(arm=arm[i]) for i in range(arm.shape[0])])
        if np.max(w_t) > epsilon:
            opt_ind = np.where(w_t > epsilon)[0]
            return np.random.choice(opt_ind, size=1).item()
        else: 
            ucb = np.vstack([self._eval_ucb(arm=arm[i],alpha=alpha) for i in range(arm.shape[0])])
            self._optimal_arms(ucb)
            return np.random.choice(self.opt_ind[-1], size=1).item()

    def _optimal_arms(self, ucb:np.ndarray) -> None: 
        """
        Evaluate the optimal arm set based on UCB of the arms

        Parameters
        ----------
        ucb : np.ndarray
            upper confidence bound for the arms
        """
        opt_ind = [np.arange(ucb.shape[0])]
        for i in range(self.l): 
            opt_ind.append(
                opt_ind[-1][par_non_dominated_sorting(ucb[opt_ind[-1]][:,self.pl[i]])]
            )
        self.opt_ind = opt_ind[1:]



class moslb_pc(moslb): 
    def __init__(
        self, 
        num_dim:int, 
        priority_chain:list,
        lamda:float=1.,
        delta:float=.05
        ) -> None:
        """
        Multi-objective stochastic linear bandit algorithm 
        considering MPL-PC order

        Parameters
        ----------
        num_dim : int
            number of context's dimension
        priority_chain : list
            list of objective index representing priority relation
        lamda : float, optional
            regularization term, lambda, by default 1.0
        delta : float, optional 
            confidence level value, by default 0.05
        """
        self.d = num_dim 
        self.pc = priority_chain
        self.c = len(priority_chain)
        self.mc = [len(self.pc[i]) for i in range(self.c)]
        self.m = sum(self.mc)
        super().__init__(num_dim=self.d, num_obj=self.m, lamda=lamda, delta=delta)

    @property
    def get_optimal_index(self) -> np.ndarray: 
        if hasattr(self, 'opt_ind'): return self.opt_ind

    def take_action(
        self, 
        arm:np.ndarray, 
        epsilon: float,
        alpha: float=1.
        ) -> int: 
        """
        Take an action

        Parameters
        ----------
        arm : np.ndarray
            arms' context
        alpha : float, optional
            parameter to control the uncertainty level, by default 1.

        Returns
        -------
        int
            index of the selected arm
        """
        arm = np.atleast_2d(arm)
        w_t = np.vstack([self.estimate_uncertainty(arm=arm[i]) for i in range(arm.shape[0])])
        if np.max(w_t) > epsilon:
            opt_ind = np.where(w_t > epsilon)[0]
            return np.random.choice(opt_ind, size=1).item()
        else: 
            ucb = np.vstack([self._eval_ucb(arm=arm[i],alpha=alpha) for i in range(arm.shape[0])])
            lcb = np.vstack([self._eval_lcb(arm=arm[i],alpha=alpha) for i in range(arm.shape[0])])
            self._optimal_arms(ucb, lcb)
            return np.random.choice(self.opt_ind, size=1).item()
            
    def _optimal_arms(self, ucb:np.ndarray, lcb:np.ndarray) -> None: 
        """
        Evaluate the optimal arm set based on UCB and LCB of the estimated rewards

        Parameters
        ----------
        ucb : np.ndarray
            upper confidence bound for the arms
        lcb : np.ndarray
            lower confidence bound for the arms
        """
        candidate_ind = []
        for i in self.pc: 
            candidate_ind.append(prior_free_lexi_filter(ucb[:,i], lcb[:,i]))
        self.opt_ind = np.unique(np.hstack(candidate_ind))
    
    