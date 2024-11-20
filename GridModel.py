from utils import * 
import jax.numpy as jnp 
import jax 
import jax.random as jar 
from functools import partial 
from tqdm import tqdm 
from functools import reduce
from operator import mul




class GridModel: 
    """
    Base class for a PartialDiefferentialEquation on a grid. 
    """

    def __init__(self, Li, Ni): 
            self.Li = Li 
            self.Ni = Ni 
            self.d = len(Li) 
            self.Volume = reduce(mul, Li)


            if len(Li) != len(Ni): 
                raise ValueError("Length of L and N must be equal")
        
            self.grid, self.k, self.dx = createGrid(Li, Ni)
            self.k2 = jnp.sum(self.k**2, axis=0)
            

    def noise(self, rng):
        """
        Generates noise in the Fourier Domaain. The noise is with noise delta correlated in space. 
        rng (jax.random.PRNGKey): A random number generator key.
        Returns:
        jnp.ndarray: The noise in the frequency domain.
        """
        noise_x = jar.normal(rng, self.Ni) 
        noise_k = jnp.fft.fftn(noise_x)
        return noise_k
    


    def Gradient(self, field):
        """
        Compute the gradient of a scalr field in the Fourier domanin 

        Args:
            field (jax.numpy.ndarray): The input field for which the gradient is to be computed.
        Returns:
            jax.numpy.ndarray: The gradient of the input field.
        """

        return jax.vmap( lambda ki, psi:  1j * ki * psi, in_axes=(0, None, ))(self.k, field)   


    def return_real(self, field): 
         """
         Transform a d-dimensional field from Fourier to real space.  
         In case the input is a multicompoent filed returns the FFT compoinent wise.

            Args: 
                field: The input field to be transformed in the form of a .
         """
         if len(field.shape) == self.d: 
            return RealSpace(field) 
         else : 
             return jax.vmap(RealSpace)(field)
         


    def Defaul_callback(self, field, par, t): 
        """
        Default callback function that returns the fields in real space
        
        Args:
            field: The input data structure to be transformed.
            par: The parameters of the model
            t: time 

        Returns:
            Transformed data structure with the same shape as the input field. The output, as the fileds, is a pytree
        """
        return jax.tree.map( self.return_real, field) 
    


    def simulate(self, field, p, rng, T, ShowProgress = False, callback = None, SaveStep = None):
        """
        Simulates the evolution of a field over time.
        Parameters:
        -----------
        field : pytree 
            The initial state of the fields to be simulated (in Fourier space), in the form of a pytree/dictionary.
            For instance if the model has a scalar field psi and a n-components vector field A, fields would be: 
            field = {'psi': psi, 'A': A}, where psi.shape = self.Ni and A.shape = (self.d, *self.Ni).

        p : dict
            A dictionary of parameters required for the simulation. ( it has to contain dt ) 

        rng : jax.random.PRNGKey
            A random number generator key to generate the stochastic noise

        T : float
            Total simulation time.

        ShowProgress : bool, optional
            If True, displays a progress bar during the simulation. Default is False.

        callback : callable, optional
            A function to be called at intervals during the simulation. Default is None.

        SaveStep : float, optional
            The time interval at which to save the simulation state using the callback. Default is None. If SaveStep is None the callback will not be called. 

            If SaveStep is not None, the callback will be called at intervals of SaveStep.

            If SaveStep is None the simulation will return just the final configuration. 

        Returns:
        --------
        field : array-like
            The final state of the field after simulation.
        output_dictionary : dict, optional
            A dictionary containing the saved states at intervals if a callback is provided. The keys are:
            - 't': array of time points at which the states were saved.
            - other keys: arrays of saved states corresponding to the keys returned by the callback.
        """

        num_steps =  int(T/p['dt']) +1
        
        
        if SaveStep is not None: 
            delta = int(SaveStep/p['dt'])
            output = [] 
        

        rngs = jax.random.split(rng, num_steps)
        run = jax.jit(partial(self.step, p=p))

        #if ShowProgress is true then the simulate function will show a tqdm progress bar
        if ShowProgress: 
            pbar = tqdm(range( num_steps))
        else: 
            pbar = range(num_steps)
        
        
        #simulation loop 
        for t in pbar:
            
            if (SaveStep is not None) and (t % delta == 0):
                output.append(callback(field, p, t)) 
            
            field = run(field, rngs[t])
            

        if SaveStep is not None: 
            #Formatting the output 
            output_dictionary = {'t' : jnp.arange(0, len(output))*p['dt']*delta} 
            for key in output[0].keys(): 
                    output_dictionary[key] =  jnp.stack([o[key] for o in output])
            return jax.tree.map( lambda x : jnp.real(jnp.fft.ifftn(x)), field), output_dictionary

        if SaveStep is None : 
            #if save step is None, return just the last configuration 
            return jax.tree.map( lambda x : jnp.real(jnp.fft.ifftn(x)), field)
