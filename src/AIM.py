
import jax.numpy as jnp 
import jax 
import jax.random as jar 
from functools import partial 
from tqdm import tqdm 

from src.utils import * 
import src.GridModel as GridModel







class AIM(GridModel.LatticeModel):
    def __init__(self, Li, Ni): 
        super().__init__(Li, Ni)
        self.mask2 = BuildAntiAliasingMask(self.k, self.dx, 2) 
        self.mask3 = BuildAntiAliasingMask(self.k, self.dx, 3) 

    def BuildKernel(self, p ): 

        G = 1./( 1 +( p['Gamma'] + p['D'])* self.k2 * p['D_t']   +  (self.k[0]**2*p['v']**2 + p['Gamma']*p['D'] * self.k2**2 )*p['D_t']**2 ) 

        G_m_m  = G * ( 1 + p['D']*self.k2*p['D_t'])
        
        G_m_rho = - G * 1j *self.k[0] * p['v'] * p['D_t']
        
        G_rho_rho = G* ( 1 + p['Gamma']*self.k2*p['D_t'])
        

        return G_m_m, G_m_rho, G_rho_rho


    def Init(self, rng): 
        m = jar.normal(rng, self.Ni)
        rho = RemoveAverage(jar.normal(rng, self.Ni))
        return {'m' : jnp.fft.fftn(m), 'rho' : jnp.fft.fftn(rho)}
    

    def step(self, fields,  rng, p): 


        m, rho = fields['m'] , fields['rho']


        # computing the dealiased fields in reals space 
        rho2 = RealSpace(rho*self.mask2 ) 
        m1 = RealSpace(m ) 
        m2 = RealSpace(m*self.mask2 ) 
        m3 = RealSpace(m*self.mask3 ) 

        #computing the noise terms 
        rng_m, rng_rho = jax.random.split(rng, 2)
        noise_m = self.noise(rng_m) * ( 2 * p['Gamma_t'] * p['D_t'] )**0.5

        noise_rho = self.noise(rng_rho)  * ( 2 * p['D_t'] * self.k2 * p['D_t'] )**0.5

        #computing the non-implicit part of the force 
        Force_m_non_implicit = (KSpace( - p['r']*m1-p['J'] * m3**3 + p['g']* rho2*m2 ) ) * p['D_t'] + noise_m
        Force_rho_non_implicit = noise_rho

        #computing the kernels for the implicit part of the integration
        G_m_m, G_m_rho, G_rho_rho = self.BuildKernel(p)

        #computing the new fields 
        m_new = G_m_m *  ( m+ Force_m_non_implicit )  + G_m_rho * (rho+Force_rho_non_implicit  )
        rho_new = G_m_rho * ( m + Force_m_non_implicit ) + G_rho_rho * (rho+Force_rho_non_implicit  )

        return {'m' : m_new, 'rho' : rho_new }
    

