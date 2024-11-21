import unittest

from src.jaxPSI import *
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

    def test_create_model(self):
        """
        A sample test case.
        """
        Li = (20, 28) 
        Ni = (30, 41) 
        model = ModelA(Li, Ni)
        model = LatticeModel(Li, Ni)
        

        self.assertEqual(1, 1) 

    def test_run_modelA(self):
        """
        A sample test case.
        """
        Li = (32, 32) 
        Ni = (32, 32) 
        model = ModelA(Li, Ni)
        par = {'r' : -0.5, 'J' : 0.3, 'Gamma' : 1.0, 'Gamma_t' : 1.0 , 'dt' : 0.01} 
        rng = jar.PRNGKey(0)  
        rng_init, rng_sim = jar.split(rng)
        field = model.Init(rng_init)   
        out = model.simulate(field, par, rng_sim, 100)
        self.assertEqual(1, 1) 


    def test_run_modelA_simpleCallBack(self):
        """
        A sample test case.
        """
        L = 32
        N = 39
        Li = (L, L) 
        Ni = (N, N) 
        model = ModelA(Li, Ni)
        par = {'r' : -0.5, 'J' : 0.3, 'Gamma' : 1.0, 'Gamma_t' : 1.0 , 'dt' : 0.01} 
        rng = jar.PRNGKey(0)  
        rng_init, rng_sim = jar.split(rng)
        field = model.Init(rng_init)   
        out = model.simulate(field, par, rng_sim, 100, SaveStep = 0.2)

        
        self.assertEqual(out[1]['phi'].shape, (501, N, N)) 

    def test_run_modelA_CustomCallBack(self):
        """
        A sample test case.
        """
        L = 32
        N = 39
        Li = (L, L) 
        Ni = (N, N) 
        model = ModelA(Li, Ni)
        par = {'r' : -0.5, 'J' : 0.3, 'Gamma' : 1.0, 'Gamma_t' : 1.0 , 'dt' : 0.01} 
        rng = jar.PRNGKey(0)  
        rng_init, rng_sim = jar.split(rng)
        field = model.Init(rng_init)   

        
        def CallBack(field, par, t): 
            phi = field['phi'] 
            phi_real = jnp.fft.ifftn(phi).real
            return {'m1' : (phi_real**1).sum(), 'm2' : (phi_real**2).sum() }
        
        out = model.simulate(field, par, rng_sim, 100, SaveStep = 0.5, callback=CallBack)
        self.assertIn('m1', out[1])
        self.assertIn('m2', out[1])
        self.assertEqual(out[1]['m1'].shape, (201,))
        final =jnp.isnan(out[1]['m2'][-1])
        self.assertFalse(final)
        self.assertIn('t', out[1])

        
    def test_case_2(self):
        """
        Another sample test case.
        """
        self.assertTrue(isinstance(self.value, int))

    def tearDown(self):
        """
        Clean up after each test, if needed.
        """
        # Example cleanup (if needed)
        self.value = None

if __name__ == '__main__':
    unittest.main()