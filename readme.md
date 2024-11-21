# JaxPSI 

This is bare-bone code to simulates stochastic partial differential equations using a pseudoSpectral, implicit-explicit integration scheme. 

The code is written using the jax-library [https://jax.readthedocs.io/en/latest/index.html] and it can run on both cpu and accelerators (GPU). 

 


#### Fully differentiable
Being written in jax the simulation routine is end-to-end differentiable. This means that we can easily take derivative of the results of the simulation with respect to the parameters or with respect to the initial condition. This can be done by just using the jax build-in auto-grad function jax.grad. (this can be quite memory intensive depending on the details of the simulation)

#### Vmap support:

The code base supports the jax.vmap function, meaning that the simulate method can be vectorizes. This can be used to run many simulations with different random seeds or different parameters. (but this can me memory intensive)

#### What kind of systems does this code simulate


The code is meant to integrate equations such as model A of critical dynamics [1]

$$
\partial_t \varphi(x) = -\Gamma_0 \nabla^2 \varphi - r \varphi - J \varphi^3 + \xi
$$
$$
\langle \xi(t, x) \xi(t^\prime, x^\prime)\rangle = 2 \tilde \Gamma \delta^d(x- x^\prime) \delta(t -t ^\prime)
$$

using a pseudo spectral partially implicit scheme[2].

You will find a very simple LatticeModel superclass to define specific models. This class is initialized with the right discretized momentum mesh. The class has also  a simulate method that can be used to simulate the system, which can accept a call back function to extract observables. Crucially the class has to be equipped with a simulation step function, which specifies the details of the simulation. 

I have implemented some standard models, such as Model A, Model B and Model H of critical dynamics [1] and the Active Ising Model [4]. From these example classes it should be quite clear how to define your own model. 


### Why finite differences is delicate and we need to use a semi implicit scheme
To understand why these kind of equations are stiff and hard to integrate, let's consider the same equation in Fourier space.
$$ 
\partial _t \hat  \varphi(k,t) = -( r + \Gamma k^2)\hat \varphi(k,t) + \mathcal {F} [ \varphi^3(x)](k,t) +\hat \xi(k,t) 
$$
Where in a discrete system, the momentum $k$ assumes only discrete in the interval $k \in [ -\frac \pi {\Delta x}, \frac \pi {\Delta x}$ ]. 

If we naively simulate this equation by discretizing time we would get, 
$$
\hat \varphi(k, t+\Delta _t) = \hat \varphi(k,t) - ( r + \Gamma k^2)\hat \varphi(k,t) \Delta t \ + \mathcal F [ \varphi^3(x)](k,t) \Delta t \ +  \ \sqrt{ 2 T \Gamma \Delta t} \eta(k, t)  
$$

This equation seems harmless, but it is not. In order for the integration to succeed we need $\Delta t \ll k_{\rm max}^{-2}  \simeq L^{-2}$. 

We can solve this problem by considering a partially implicit integration scheme. We consider integrate the  $\Gamma k^2$ term in the equation of motion implicitly by using a partially backward integration scheme: 
$$
\hat \varphi(k, t+\Delta _t) = \hat \varphi(k,t) - r \hat \varphi(k,t) \Delta t + \Gamma k^2 \hat \varphi(k,t+\Delta t)\Delta t \ + \mathcal F [ \varphi^3(x)](k,t)\Delta t \ +  \ \sqrt{ 2 T \Gamma \Delta t} \eta(k, t)  
$$
Crucially now the $k^2$ term is computed at the next time step, leading to the following update equation. 
$$
\hat \varphi(k, t +\Delta t) = \frac{ \hat \varphi(k,t) - r \hat \varphi(k,t) \Delta t \ + \mathcal F [ \varphi^3(x)](k,t) \Delta t \ + \sqrt{2  T \tilde \Gamma \Delta t} \ \eta(k,t)}{1 - \Gamma k^2 \Delta t}
$$

Using this implicit explicit scheme the stiff term $\Gamma k^2$ is know in the denominator solving the problematics that we had with the explicit integration. 

### De - aliasing 
We work in with the Fourier transformed field $\hat \varphi(k,t)$, so that we can integrate the Laplacian term using a semi-implicit scheme. In this representation, non-linear terms such as $\varphi^3(x,t)$  are computed as convolutions,
$$
\mathcal F [ \varphi^3(x,t)]= \int   \varphi(q,t) \hat \varphi(p,t) \hat \varphi(k-p-q,t)\ dq  \ dp
$$ 
which are computationally expensive. To avoid this, we compute the non linear terms in real space: 
$$
 \mathcal F [ \varphi^3(x,t)]= \mathcal F[ \mathcal F^{-1}[ \hat \varphi]^3 ]
$$


This approach presents some problematics when dealing with discrete systems, in which the momenta in Fourier space are discrete and are defined up to $2 \pi / \Delta x$ in a periodic domain.  
$$
q = q + \frac{ 2 \pi}{ \Delta x}
$$ 

When we compute the non-linear terms, such as $\varphi(x,t)^2$ going in real space we are automatically considering all the possible pairs of moments, 
$$
\mathcal F[\varphi^2] = \sum_k \sum_q e^{i ( k+q) x } \hat \varphi(k,t) \hat \varphi(q,t)
$$
In this way we are considering pairs of moments that sums up to more than $2 \pi / 2 \Delta x $.

In general, every time we compute a non-linear term of order $m$, we have to set to zero all the modes with $|k|<\pi/(m \Delta x)$. This process is called <strong> de-aliasing</strong> [3]. 

In the code we do this by applying a mask to the fields before applying the inverse Fourier Transform to go back to the reals space domain. 

## References
1. Hohenberg, P. C., & Halperin, B. I. (1977). Theory of dynamic critical phenomena. Reviews of Modern Physics, 49(3), 435.
2. Fornberg, B. (1998). A practical guide to pseudospectral methods. Cambridge university press.

3. Orszag, S. A. (1971). On the elimination of aliasing in finite-difference schemes by filtering high-wavenumber components. Journal of Atmospheric Sciences, 28(6), 1074-1074.

4. Solon, A. P., & Tailleur, J. (2015). Flocking with discrete symmetry: The two-dimensional active Ising model. Physical Review E, 92(4), 042119.
 