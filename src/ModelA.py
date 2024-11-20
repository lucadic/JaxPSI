import jax.numpy as jnp 
import jax 
import jax.random as jar 
from functools import partial 
from tqdm import tqdm 
from src.utils import * 
import src.GridModel as GridModel





class ModelA(GridModel.LatticeModel):

    def __init__(self, Li, Ni): 
        super().__init__(Li, Ni)

        self.mask3 = BuildAntiAliasingMask(self.k, self.dx, 3) 


    def Init(self, rng): 
        phi = jar.normal(rng, self.Ni)
        return {'phi' : jnp.fft.fftn(phi)}
    

    def step(self, fields,  rng, p): 
        phi = fields['phi'] 
        #       computing the dealiased fields in reals space 
        phi1 = jnp.real(jnp.fft.ifftn(phi) )
        phi3 = jnp.real(jnp.fft.ifftn(phi*self.mask3 ) )
        #       computin the noise term 
        noise_phi = self.noise(rng) * ( 2 * p['Gamma_t'] * p['dt'] )**0.5
        '''computing the non-implicit part of the force '''
        Force_non_implicit = jnp.fft.fftn( - p['r']*phi1-p['J'] * phi3**3) * p['dt'] + noise_phi
        Implicit_term =  - p['dt'] * self.k2 * p['Gamma'] 
        #          computing the new fields 
        new_phi = (phi + Force_non_implicit + noise_phi)/(1-Implicit_term) 
        return {'phi' : new_phi} 
    

