import jax.numpy as jnp 
import jax 
import jax.random as jar 
from functools import partial 
from functools import reduce

from operator import mul
from tqdm import tqdm 

from src.utils import * 
import src.GridModel as GridModel


class ModelH(GridModel.LatticeModel):


    def __init__(self, Li, Ni): 
        super().__init__(Li, Ni)

        self.mask2 = BuildAntiAliasingMask(self.k, self.dx, 2) 
        self.mask3 = BuildAntiAliasingMask(self.k, self.dx, 3) 
        self.mask4 = BuildAntiAliasingMask(self.k, self.dx, 4) 
        self.khat = jnp.where(self.k2 !=0 , self.k/self.k2**0.5, 0)
        
    
    
    def T(self, field): 
        return field - self.khat * jnp.sum(self.khat * field, axis=0, keepdims = True)



    def ModelH_noise(self, rng): 
        rng_psi, rng_J = jar.split(rng, 2) 
        noise_psi = jar.normal(rng_psi, self.Ni)
        noise_J = jar.normal(rng_J, self.k.shape)
        return jnp.fft.fftn(noise_psi), jax.vmap(jnp.fft.fftn)(noise_J)


    def Init(self, rng, J0 = 0, psi0 = 0 ): 
        rng_psi, rng_J = jar.split(rng, 2)  
        psi = jar.normal(rng_psi, tuple(self.Ni) )
        J = jar.normal(rng_J, self.k.shape )

        psi = (psi - psi.mean()) + psi0/self.Volume
        J =  self.T ((J - J.mean()) + J0/ self.Volume)

        return {'psi' : jnp.fft.fftn(psi), 'J' : jnp.fft.fftn(J) }
    
    

    def step(self, fields,  rng, p): 
        psi, J = fields['psi'], fields['J']
        # #computing the dealiased fields in reals space 

        psi3 = RealSpace(psi*self.mask3 ) 
        psi2 = RealSpace(psi*self.mask2 ) 
        grad_psi2 = RealSpace( self.Gradient(psi*self.mask2 )) 
        nabla2_psi2 = RealSpace( -self.k2*self.mask2 )
        psi1 = RealSpace(psi ) 
        psi4 = RealSpace(psi*self.mask4 ) 
        grad_psi4 = RealSpace( self.Gradient(psi*self.mask4 )) 
        J2 = jax.vmap(RealSpace)(J*self.mask2 ) 


        # #compuitn the implicit intergration kernels 
        Implicit_term_psi =   - p['dt'] * (self.k2)**2 * p['lambda'] 
        Implicit_term_J =  - ((self.k2) * p['eta']) * p['dt']

        # #computing the noise terms 
        noise_psi, noise_J = self.ModelH_noise(rng)
        noise_psi = noise_psi * (2 * p['lambda_t'] * self.k2  * p['dt'])**0.5
        noise_J = self.T ( noise_J * (2 * p['eta_t'] * self.k2* p['dt'])**0.5) 

        # #computing the force terms 
        Explicit_term_psi =   (self.k2) * p['lambda']  *  KSpace( - p['r'] * psi1  - p['u'] * psi3**3)* p['dt']
        MC_temrm_psi = - p['g'] *  KSpace( (grad_psi2 * J2).sum(0)) * p['dt']


        MC_term_J = - p['g'] * self.T( jax.vmap(KSpace)( grad_psi2  * ( p['r'] *psi2 - nabla2_psi2) +  p['u'] *grad_psi4 * psi4**3 )) * p['dt']


        new_J = (J + MC_term_J + noise_J)/(1-Implicit_term_J)

        new_psi = (psi + Explicit_term_psi + MC_temrm_psi + noise_psi)/(1-Implicit_term_psi)


        return {'psi' : new_psi, 'J' : new_J}
    