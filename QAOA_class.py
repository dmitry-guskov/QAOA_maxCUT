
import numpy as np
from scipy.optimize import *
import random
import time


class QAOA:
    
    def __init__(self,depth,H):             # Class initialization. Arguments are "depth", 
                                            # and a Diagonal Hamiltonian,"H".    
        
        diagonal = np.asarray(H)
        if (diagonal.ndim != 1 or len(diagonal) < 2 or len(diagonal) & (len(diagonal)-1)
                or not np.isrealobj(diagonal) or not np.isfinite(diagonal).all()):
            raise ValueError('Expected a finite real power-of-two diagonal.')
        if not isinstance(depth,(int,np.integer)) or depth < 1:
            raise ValueError('Expected a positive integer depth.')
        self.H = diagonal.astype(float,copy=True)
        self.n = int(np.log2(int(len(self.H)))) # Calculates the number of qubits. 
        
        #______________________________________________________________________________________________________
        self.X = self.new_mixerX()              # Executes a sequence of array manipulations to encapsulate the 
                                                # effect of standard one body driver hamiltonian, \Sum \sigma_x,
                                                # acting on any state.
        #______________________________________________________________________________________________________
        
        
        self.min = min(self.H)                     # Calculates minimum of the Hamiltonain, Ground state energy.
        
        self.deg = len(self.H[self.H == self.min]) # Calculates the degeneracy of Ground states. 
        self.p = depth                             # Standard qaoa depth written as "p".
        
        self.heruistic_LW_seed1 = 50
        self.heruistic_LW_seed2 = 25
        
        
        #______________________________________________________________________________________________________   
                    
                    # The sequence of array manipulations that return action of the driver,
                    # in terms of permutation indices.
    
    def new_mixerX(self):
        """Single-qubit X indices, qubit zero most significant."""
        indices = np.arange(2**self.n)
        return [indices ^ (1 << (self.n-1-q)) for q in range(self.n)]
    #__________________________________________________________________________________________________________   
        
        
    def U_gamma(self,angle,state):       # applies exp{-i\gamma H_z}, here as "U_gamma", on a "state".
        
        t = -1j*angle
        state = np.asarray(state)
        phase = np.exp(t*self.H)
        state = state * (phase if state.ndim == 1 else phase[:, None])
        
        return state
            
    
    
    
    def V_beta(self,angle,state):        # applies exp{-i\beta H_x}, here as "V_beta", on a "state".
        c = np.cos(angle)
        s = np.sin(angle)
        
        for i in range(self.n):
            t = self.X[i]
            st = state[t]
            state = c*state + (-1j*s*st)
            
        return state
  
    #__________________________________________________________________________________________________________
    
                        # This step creates the qaoa_ansatz w.r.t to "angles" that are passed. 
                        # "angles" are passed as [gamma_1,gamma_2,...,gamma_p,beta_1,beta2,....beta_p].
    
    def qaoa_ansatz(self, angles):
        
        state = np.ones((2**self.n,1),dtype = 'complex128')*(1/np.sqrt(2**self.n))
        if len(angles) % 2:
            raise ValueError("Expected complete gamma/beta angle pairs.")
        p = len(angles)//2
        for i in range(p):
            state = self.U_gamma(angles[i],state)
            state = self.V_beta(angles[p + i],state)
        
        return state 
    
    #__________________________________________________________________________________________________________
    
    
    def apply_ansatz(self, angles,state):
        if len(angles) % 2:
            raise ValueError("Expected complete gamma/beta angle pairs.")
        p = len(angles)//2
        for i in range(p):
            state = self.U_gamma(angles[i],state)
            state = self.V_beta(angles[p + i],state)
        
        return state
    

        
    
    #__________________________________________________________________________________________________________
    
    
    def expectation(self,angles):   # Calculates expected value of the Hamiltonian w.r.t qaoa_ansatz state,
                                    # defined by the specific choice of "angles".
        
        state = self.qaoa_ansatz(angles)
        
        ex = np.vdot(state,state*(self.H).reshape((2**self.n,1)))
        
        return np.real(ex)
            
    
    
    
    def overlap(self,state):        # Calculates ground state overlap for any "state",
                                    # passed to it. Usually the final state  or "f_state" returned, 
                                    # after optimization.
        
        g_ener = min(self.H)
        olap = 0
        for i in range(len(self.H)):
            if self.H[i] == g_ener:
                olap+= np.absolute(state[i])**2
        
        return olap
    
   #__________________________________________________________________________________________________________ 
    
    
                    # Main execution of the algorithm. 
                    # 1) Create "initial_angles", this would be the guess or  
                    #    starting point for the optimizer.
                    # 2) Optimizer "L-BFGS-B" then takes "initial_angles" and calls "expectation".
                    # 3) "expectation" then returns a number and the optimizer tries to minimize this,
                    #     by doing finite differences. Thermination returns optimized angles, 
                    #     stored here as "res.x".
                    # 4) Treating the optimized angles as being global minima for "expectation",
                    #    we calculate and store (as class attributes) the qaoa energy, here as "q_energy",
                    #    energy error, here as "q_error",
                    #    ground state overlap, here as "olap"
                    #    and also the optimal state, here as "f_state" 
    
    def run_RI(self, *, seed=None):
        rng = random if seed is None else random.Random(seed)
        t_start = time.time()
        initial_angles=[]
        bds= [(0,2*np.pi)]*self.p + [(0,1*np.pi)]*self.p
        for i in range(2*self.p):
            if i < self.p:
                initial_angles.append(rng.uniform(0,2*np.pi))
            else:
                initial_angles.append(rng.uniform(0,np.pi))
            
        res = minimize(self.expectation,initial_angles,method='L-BFGS-B',                        jac=None, bounds=bds, options={'maxfun': 150000})
        
        t_end = time.time()
        self.opt_angles = res.x
        self.exe_time = float(t_end - t_start)
        self.opt_iter = int(res.nfev)+1
        self.q_energy = self.expectation(res.x)
        self.q_error = self.q_energy - self.min
        self.f_state = self.qaoa_ansatz(res.x)
        self.olap = self.overlap(self.f_state)[0]
        self.log = (f' Depth: {self.p} \n Error: {self.q_error} \n QAOA_Eg: {self.q_energy} \n'
                    f' Exact_Eg: {self.min} \n Overlap: {self.olap} \n Exe_time: {self.exe_time} \n'
                    f' Iternations: {self.opt_iter}')
        
     #__________________________________________________________________________________________________________
        
    def run_heuristic_LW(self, *, seed=None):
        rng = random if seed is None else random.Random(seed)
        if any(not isinstance(v,(int,np.integer)) or v < 1
               for v in (self.heruistic_LW_seed1,self.heruistic_LW_seed2)):
            raise ValueError('Expected positive restart counts.')
        total_nfev = 0
        
        initial_guess = lambda x: ([rng.uniform(0,2*np.pi) for _ in range(x) ] +                                    [rng.uniform(0,np.pi) for _ in range(x)])
        bds = lambda x: [(0.1,2*np.pi)]*x + [(0.1,1*np.pi)]*x
        
        def combine(a,b):

            a = list(a)
            b = list(b)
            a1 = a[0:int(len(a)/2)]
            a2 = a[int(len(a)/2)::]
            b1 = b[0:int(len(b)/2)]
            b2 = b[int(len(b)/2)::]
            a = a1+b1
            b = a2+b2
            
            return a + b 
        
        
        
        temp = [] 
        t_start = time.time()
        
        for _ in range(self.heruistic_LW_seed1):
            initial_guess_p1 = initial_guess(1)
            res = minimize(self.expectation,initial_guess_p1,method='L-BFGS-B',                           jac=None, bounds=bds(1), options={'maxfun': 150000})
            total_nfev += res.nfev
            temp.append([res.fun, res.x.copy()])
            
        temp = np.asarray(temp,dtype=object)
        idx = np.argmin(temp[:,0])
        opt_angles = temp[idx][1]
       
        
        t_state = np.ones((2**self.n,1),dtype = 'complex128')*(1/np.sqrt(2**self.n))
        
        
        while len(opt_angles) < 2*self.p:
            ts1 = time.time()
            t_state = self.qaoa_ansatz(opt_angles)
            
            
            def ex(x):
                evolved = self.apply_ansatz(x,t_state)
                return float(np.vdot(evolved,evolved*self.H[:,None]).real)
            temp = [] 
            
            for _ in range(self.heruistic_LW_seed2):
                
                res = minimize(ex,initial_guess(1),method='L-BFGS-B', jac=None, bounds=bds(1),                                options={'maxfun': 150000})
                total_nfev += res.nfev
                temp.append([res.fun, res.x])
            temp = np.asarray(temp,dtype=object)
            idx = np.argmin(temp[:,0])
            lw_angles = temp[idx][1]
            opt_angles = combine(opt_angles,lw_angles)
            

            
            res = minimize(self.expectation,opt_angles,method='L-BFGS-B', jac=None,                            bounds=bds(int(len(opt_angles)/2)), options={'maxfun': 150000})    
            total_nfev += res.nfev
            opt_angles = res.x
        self.opt_angles = opt_angles    
       
        

            
        t_end = time.time()
        self.exe_time = float(t_end - t_start)
        self.opt_iter = int(total_nfev)+1
        self.q_energy = self.expectation(self.opt_angles)
        self.q_error = self.q_energy - self.min
        self.f_state = self.qaoa_ansatz(self.opt_angles)
        self.olap = self.overlap(self.f_state)[0]
        self.log = (f' Depth: {self.p} \n Error: {self.q_error} \n QAOA_Eg: {self.q_energy} \n'
                    f' Exact_Eg: {self.min} \n Overlap: {self.olap} \n Exe_time: {self.exe_time} \n'
                    f' Iternations: {self.opt_iter}')

