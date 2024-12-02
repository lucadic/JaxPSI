# **JaxPSI**  

JaxPSI is a lightweight codebase for simulating stochastic partial differential equations using a pseudo-spectral, implicit-explicit integration scheme.  

The code is implemented using [JAX](https://jax.readthedocs.io/en/latest/index.html) and is compatible with both CPUs and accelerators (GPUs).  

---

## **Key Features**  

### **1. Fully Differentiable**  
Thanks to JAX, the simulation routine is end-to-end differentiable. This allows you to compute derivatives of the simulation results with respect to parameters or initial conditions effortlessly using JAX's built-in automatic differentiation function, `jax.grad`.  
> **Note:** Differentiation can be memory-intensive depending on the details of the simulation.  

### **2. `vmap` Support**  
The code supports JAX's `vmap` function, enabling vectorization of the simulation method. This functionality allows you to run multiple simulations simultaneously with different random seeds or parameters.  
> **Note:** Using `vmap` can also be memory-intensive.  

---

## **Supported Systems**  

JaxPSI is designed to integrate equations such as Model A of critical dynamics [1]:  

$$
\partial_t \varphi(x) = -\Gamma_0 \nabla^2 \varphi - r \varphi - J \varphi^3 + \xi
$$  

$$
\langle \xi(t, x) \xi(t', x') \rangle = 2 \tilde{\Gamma} \delta^d(x - x') \delta(t - t')
$$  

using a pseudo-spectral, partially implicit scheme [2].  

### **Defining Models**  
The code includes a simple `LatticeModel` superclass for defining specific models. This class initializes the discretized momentum mesh and provides:  
- A `simulate` method for running simulations (accepts callback functions to extract observables).  
- A customizable simulation step function for implementing model-specific dynamics.  

Pre-implemented models include:  
- Model A, Model B, and Model H of critical dynamics [1].  
- The Active Ising Model [4].  

You can use these examples as templates to define your own models.  

---

## **Why Use a Semi-Implicit Scheme?**  

### **The Problem with Explicit Schemes**  
Consider the equation in Fourier space:  

$$
\partial_t \hat{\varphi}(k, t) = -(r + \Gamma k^2)\hat{\varphi}(k, t) + \hat{\xi}(k, t)
$$  

Here, the momentum $` k `$ is discrete and lies in the interval $` k \in (-\pi / \Delta x, \pi / \Delta x) `$.  

A naive time discretization yields:  

$$
\hat{\varphi}(k, t + \Delta t) = \hat{\varphi}(k, t) - (r + \Gamma k^2) \hat{\varphi}(k, t) \Delta t + \sqrt{2 T \Gamma \Delta t} \eta(k, t)
$$  

For stability, this requires $` \Delta t \ll k_{\text{max}}^{-2} \simeq L^{-2} `$, which can be highly restrictive.  

### **The Semi-Implicit Solution**  
To address this stiffness, we integrate the $` \Gamma k^2 `$ term implicitly using a partially backward scheme:  

$$
\hat{\varphi}(k, t + \Delta t) = \frac{\hat{\varphi}(k, t) - r \hat{\varphi}(k, t) \Delta t + \mathcal{F}(\varphi^3(x))(k, t) \Delta t + \sqrt{2 T \tilde{\Gamma} \Delta t} \eta(k, t)}{1 - \Gamma k^2 \Delta t}
$$  

This approach moves the stiff term $` \Gamma k^2 `$ into the denominator, improving numerical stability.  

---

## **De-Aliasing**  

In Fourier space, non-linear terms such as $` \varphi^3(x, t) `$ are computed as convolutions:  

$$
\mathcal{F}(\varphi^3(x, t)) = \int \varphi(q, t) \hat{\varphi}(p, t) \hat{\varphi}(k - p - q, t) \, dq \, dp
$$  

To simplify computations, non-linear terms are instead calculated in real space:  

$$
\mathcal{F}(\varphi^3(x, t)) = \mathcal{F}(\mathcal{F}^{-1}(\hat{\varphi})^3)
$$  

However, this approach introduces aliasing issues in discrete systems with periodic domains where $q = q + 2\pi / \Delta x $.  

For non-linear terms of order $m$, modes with  $|k| > \pi / (m \Delta x) $ must be filtered. This process, known as **de-aliasing**, is implemented in the code using masks applied before the inverse Fourier Transform.  

---

## **References**  

1. Hohenberg, P. C., & Halperin, B. I. (1977). *Theory of dynamic critical phenomena*. Reviews of Modern Physics, 49(3), 435.  
2. Fornberg, B. (1998). *A practical guide to pseudospectral methods*. Cambridge University Press.  
3. Orszag, S. A. (1971). *On the elimination of aliasing in finite-difference schemes by filtering high-wavenumber components*. Journal of Atmospheric Sciences, 28(6), 1074-1074.  
4. Solon, A. P., & Tailleur, J. (2015). *Flocking with discrete symmetry: The two-dimensional active Ising model*. Physical Review E, 92(4), 042119.  

---
