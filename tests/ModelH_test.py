import unittest
from src.jaxPSI import *
import src.jaxPSI as jaxPSI
import jax 
import jax.numpy as jnp 
import jax.random as jar 


class TestExample(unittest.TestCase):
    def setUp(self):
        """
        Set up any preconditions or reusable objects before each test.
        """
        # Example setup (if needed)
        self.value = 42

  

    def test_model_H_Projection_Function(self): 
        #define and plot a 2dimensional vector field with non zero divergence 
        L = 64
        Li = (L, L) 
        Ni = (256, 256) 

        model = ModelH(Li, Ni) 
        X, Y = model.grid
        a = 1.0
        b = 1.0
        Udf = b* (Y-L/2) 
        Vdf = -b* (X-L/2)
        ### define a 2 d fiele explicitly dividing the divergent and indivergent parts 
        U = a*  (X-L/2) + Udf
        V = a*  (Y-L/2) +Vdf
        J = jnp.stack([U, V], axis=0)
        J_indivergent = jnp.stack([Udf, Vdf], axis=0)

        Jk = jax.vmap(jnp.fft.fftn)(J)
        J_indivergent_prediction = jnp.real(jax.vmap(jnp.fft.ifftn)(model.T(Jk)) )
        err = (jnp.mean( (J_indivergent_prediction - J_indivergent)**2)/jnp.mean(J_indivergent**2))**0.5


        self.assertTrue( err<0.01) 

    def test_run_modelH_CustomCallBack(self):
        """
        A sample test case.
        """
        L = 16
        N = 32
        Li = (L, L) 
        Ni = (N, N) 
        model = ModelH(Li, Ni)
        par = {'r' : -0.5, 'u' : 0.3, 'lambda' : 1.0, 'lambda_t' : 1.0 , 'dt' : 0.1, 
               'eta' : 0.5, 'eta_t' : 0.5, 'g' : 0.5} 
        
        rng = jar.PRNGKey(0)  
        rng_init, rng_sim = jar.split(rng)
        field = model.Init(rng_init)   

        
        def CallBack(field, par, t): 
            psi = field['psi'] 
            psi_r = jaxPSI.RealSpace(psi)
            return {'psi2' : (psi_r**2).sum()}
        
        out = model.simulate(field, par, rng_sim, 100, SaveStep = 0.5, callback=CallBack)
        self.assertIn('psi2', out[1])
        self.assertEqual(out[1]['psi2'].shape, (201,))
        final =jnp.isnan(out[1]['psi2'][-1])
        self.assertFalse(final)
        self.assertIn('t', out[1])

        



  

    def tearDown(self):
        """
        Clean up after each test, if needed.
        """
        # Example cleanup (if needed)
        self.value = None

if __name__ == '__main__':
    unittest.main()