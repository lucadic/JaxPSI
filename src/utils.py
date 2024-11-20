
import numpy as np 
import jax 
import jax.numpy as jnp 
import matplotlib.pyplot as plt 
import jax.random as jar 
from importlib import reload 

from functools import reduce 
import operator


def createGrid(Li, Ni):
    if len(Li) != len(Ni): 
        raise ValueError("Length of L and N must be equal")
    
    x = [ jnp.linspace(0, l, n, endpoint=False) for l, n in zip(Li, Ni) ]
    grid = jnp.meshgrid(*x, indexing='ij')
    dxi= [ l/n for l, n in zip(Li, Ni) ]
    ki = [ jnp.fft.fftfreq(n, d=dx)*2*jnp.pi for n, dx in zip(Ni, dxi) ]
    K = jnp.meshgrid(*ki, indexing='ij')
    return jnp.stack(grid), jnp.stack(K), jnp.array(dxi)


def BuildAntiAliasingMask(k, dx, n): 
    masks = jax.vmap( lambda x, y: x< y, in_axes=(0, 0))(k**2, ( 2*jnp.pi/((n+1)*dx) )** 2)
    mask = jnp.prod(masks, axis=0)
    return mask 

    

def RealSpace(field): 
    return jnp.real(jnp.fft.ifftn(field))


def KSpace(field): 
    return (jnp.fft.fftn(field))

def RemoveAverage(phi): 
    return phi - phi.mean()