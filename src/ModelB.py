import jax.numpy as jnp 
import jax 
import jax.random as jar 
from functools import partial 
from tqdm import tqdm 
from src.utils import * 
import src.GridModel as GridModel




class ModelB(GridModel.LatticeModel):


    def __init__(self, Li, Ni): 
        super().__init__(Li, Ni)

        self.mask3 = BuildAntiAliasingMask(self.k, self.dx, 3) 


    def InitAt(self, rng, phi0): 
        phi = jar.normal(rng, self.Ni)+phi0
        return {'phi' : jnp.fft.fftn(phi)}
    

    def Init(self, rng): 
        phi = jar.normal(rng, self.Ni)
        return {'phi' : jnp.fft.fftn(phi)}
    

    def step(self, fields,  rng, p): 
        phi = fields['phi'] 


        # computing the dealiased fields in reals space 
        phi1 = RealSpace(phi) 

        phi3 = RealSpace(phi*self.mask3 ) 

        #computin the noise term 
        noise_phi = self.noise(rng) * ( 2 * p['Gamma_t'] * p['dt']*self.k2)**0.5

        #computing the non-implicit part of the force 
        Force_non_implicit = self.k2 * KSpace( - p['r']*phi1-p['J'] * phi3**3) * p['dt'] + noise_phi
        Implicit_term =  - p['dt'] * (self.k2)**2 * p['Gamma'] 

        #computing the new fields 
        new_phi = (phi + Force_non_implicit + noise_phi)/(1-Implicit_term) 
        return {'phi' : new_phi} 
    
    