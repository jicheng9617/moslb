import numpy as np 

from utils import par_non_dominated_sorting, prior_free_lexi_filter

class moslb:
    def __init__(self, dim, num_obj) -> None:

        self.d = dim
        self.m = num_obj

    @property
    def get_num_obj(self) -> int:
        """_summary_

        Returns
        -------
        int
            _description_
        """
        return self.m
    
    @property
    def get_num_arm(self) -> int: 
        """_summary_

        Returns
        -------
        int
            _description_
        """
        return np.atleast_2d(self.A).shape[0]
    
    @property
    def get_num_dim(self) -> int: 
        """_summary_

        Returns
        -------
        int
            _description_
        """
        return self.d

    def reset(self, arms: np.ndarray, delta: float) -> None: 

        self.A = arms
        self.delta = delta
        # initialize parameters
        self.t = 1
        self.X_list = [] 
        self.Y_list = [] 
        self.V = np.eye(self.d) 
        self.theta = np.zeros((self.m, self.d))
        self.V_inv = self.V 

    def _expected_reward(self): 

        self.y_hat = np.matmul(self.A, self.theta.T) 

    def _eval_ucb(self, alpha: float=1.) -> np.ndarray: 

        gamma_t = np.sqrt(self.d * np.log(self.m * (1 + self.t) / self.delta)) + 1
        self.w_t = np.vstack([gamma_t * np.sqrt(np.dot(self.A[i], self.V_inv).dot(self.A[i]))\
                     for i in range(self.get_num_arm)])
        self.u_t = self.y_hat + self.w_t

    def _optimal_arm_set(self) -> np.ndarray: 

        return par_non_dominated_sorting(self.u_t)

    def take_action(self, alpha: float=1.) -> int: 

        self._expected_reward()
        self._eval_ucb(alpha=alpha)
        opt_ind = self._optimal_arm_set()
        return np.random.choice(opt_ind, size=1).item()

    def update_params(self, arm: np.ndarray, reward: np.ndarray) -> None: 

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

    def __init__(self, dim, priority_level) -> None:
        self.d = dim 
        self.pl = priority_level
        self.l = len(self.pl)
        self.ml = [len(self.pl[i]) for i in range(self.l)]
        super().__init__(dim, num_obj=sum(self.ml))

    def reset(self, arms: np.ndarray, delta: float) -> None: 

        self.A = arms
        self.delta = delta
        # initialize parameters
        self.t = 1
        self.X_list = [] 
        self.Y_list = [[] for _ in range(self.l)] 
        self.V = np.eye(self.d) 
        self.theta = [np.zeros((self.ml[i], self.d)) for i in range(self.l)]
        self.V_inv = self.V 

    def _expected_reward(self): 

        self.y_hat = [np.matmul(self.A, self.theta[i].T) for i in range(self.l)] 

    def _eval_ucb(self, alpha: float=1.) -> np.ndarray: 

        gamma_t = np.sqrt(self.d * np.log(self.m * (1 + self.t) / self.delta)) + 1
        self.w_t = np.vstack([alpha * gamma_t * np.sqrt(np.dot(self.A[i], self.V_inv).dot(self.A[i]))\
                     for i in range(self.get_num_arm)])
        self.u_t = [self.y_hat[i] + self.w_t for i in range(self.l)]

    def _optimal_arm_set(self) -> np.ndarray: 

        opt_ind = [] 
        opt_ind.append(par_non_dominated_sorting(self.u_t[0]))
        for i in range(1, self.l): 
            opt_ind.append(opt_ind[i-1][par_non_dominated_sorting(self.u_t[i][opt_ind[i-1]])])
        return opt_ind[-1]

    def take_action(self, epsilon: float, alpha: float=1.) -> int: 

        self._expected_reward()
        self._eval_ucb(alpha=alpha)
        if np.max(self.w_t) > epsilon:
            opt_ind = np.where(self.w_t > epsilon)[0]
        else: 
            opt_ind = self._optimal_arm_set()
        return np.random.choice(opt_ind, size=1).item()

    def update_params(self, arm: np.ndarray, reward: list) -> None: 

        self.t += 1
        self.X_list.append(arm) 
        for i in range(self.l): self.Y_list[i].append(reward[i])
        X = np.vstack(self.X_list) 
        Y = [np.vstack(self.Y_list[i]) for i in range(self.l)] 
        self.V += np.outer(arm, arm)
        self.V_inv = np.linalg.inv(self.V) 

        for i in range(self.l): 
            for j in range(self.ml[i]):
                self.theta[i][j] = self.V_inv @ X.T @ Y[i][:, j]



class moslb_pc(moslb): 

    def __init__(self, dim, priority_chain) -> None:
        self.d = dim 
        self.pc = priority_chain
        self.c = len(priority_chain)
        self.mc = [len(self.pc[i]) for i in range(self.c)]
        self.m = sum(self.mc)
        super().__init__(dim=self.d, num_obj=self.m)

    def reset(self, arms: np.ndarray, delta: float) -> None: 

        self.A = arms
        self.delta = delta
        # initialize parameters
        self.t = 1
        self.X_list = [] 
        self.Y_list = [[] for _ in range(self.c)] 
        self.V = np.eye(self.d) 
        self.theta = [np.zeros((self.mc[i], self.d)) for i in range(self.c)]
        self.V_inv = self.V 

    def _expected_reward(self): 

        self.y_hat = [np.matmul(self.A, self.theta[i].T).round(decimals=2) \
                      for i in range(self.c)] 

    def _eval_ucb(self, alpha: float=1.) -> np.ndarray: 

        gamma_t = np.sqrt(self.d * np.log(self.m * (1 + self.t) / self.delta)) + 1
        self.w_t = np.vstack([alpha * gamma_t * np.sqrt(np.dot(self.A[i], self.V_inv).dot(self.A[i]))\
                     for i in range(self.get_num_arm)])
        self.u_t = [self.y_hat[i] + self.w_t for i in range(self.c)]
        self.l_t = [self.y_hat[i] - self.w_t for i in range(self.c)]

    def _pc_filter(self) -> np.ndarray: 

        candidate_ind = []
        for i in range(self.c): 
            candidate_ind.append(prior_free_lexi_filter(ucb=self.u_t[i], lcb=self.l_t[i]))

        return np.unique(np.hstack(candidate_ind))
    
    def take_action(self, epsilon: float, alpha: float=1.) -> int: 

        self._expected_reward()
        self._eval_ucb(alpha=alpha)
        if np.max(self.w_t) > epsilon:
            opt_ind = np.where(self.w_t > epsilon)[0]
        else: 
            opt_ind = self._pc_filter()
        return np.random.choice(opt_ind, size=1).item()
    
    def update_params(self, arm: np.ndarray, reward: list) -> None: 

        self.t += 1
        self.X_list.append(arm) 
        for i in range(self.c): self.Y_list[i].append(reward[i])
        X = np.vstack(self.X_list) 
        Y = [np.vstack(self.Y_list[i]) for i in range(self.c)] 
        self.V += np.outer(arm, arm)
        self.V_inv = np.linalg.inv(self.V) 

        for i in range(self.c): 
            for j in range(self.mc[i]):
                self.theta[i][j] = self.V_inv @ X.T @ Y[i][:, j]