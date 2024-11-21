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

  

   


    def test_run_AIM_3d(self):
        """
        A sample test case.
        """
        L = 16
        N = 32
        Li = (L, L) 
        Ni = (N, N) 
        model = AIM(Li, Ni)
        par = {'r' : -0.5, 'J' : 0.3, 'Gamma' : 1.0, 'Gamma_t' : 1.0 , 'dt' : 0.1, 
               'D' : 0.5, 'D_t' : 0.5, 'v' : 0.25, 'g' : 0.1} 
        
        rng = jar.PRNGKey(0)  
        rng_init, rng_sim = jar.split(rng)
        field = model.Init(rng_init)   

        
        def CallBack(field, par, t): 
            m = field['m'] 
            m_r = jaxPSI.RealSpace(m)
            return {'m2' : (m_r**2).sum()}
        
        out = model.simulate(field, par, rng_sim, 100, SaveStep = 0.5, callback=CallBack)
        self.assertIn('m2', out[1])
        self.assertEqual(out[1]['m2'].shape, (201,))
        final =jnp.isnan(out[1]['m2'][-1])
        self.assertFalse(final)
        self.assertIn('t', out[1])

    def test_run_AIM_3d(self):
        """
        A sample test case.
        """
        L = 16
        N = 32
        Li = (L, L, L) 
        Ni = (N, N, N) 
        model = AIM(Li, Ni)
        par = {'r' : -0.5, 'J' : 0.3, 'Gamma' : 1.0, 'Gamma_t' : 1.0 , 'dt' : 0.1, 
               'D' : 0.5, 'D_t' : 0.5, 'v' : 0.25, 'g' : 0.1} 
        
        rng = jar.PRNGKey(0)  
        rng_init, rng_sim = jar.split(rng)
        field = model.Init(rng_init)   

        
        def CallBack(field, par, t): 
            m = field['m'] 
            m_r = jaxPSI.RealSpace(m)
            return {'m2' : (m_r**2).sum()}
        
        out = model.simulate(field, par, rng_sim, 100, SaveStep = 0.5, callback=CallBack)
        self.assertIn('m2', out[1])
        self.assertEqual(out[1]['m2'].shape, (201,))
        final =jnp.isnan(out[1]['m2'][-1])
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