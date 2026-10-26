import unittest
import numpy as np
from models.iris.truth.response_operator_v1 import central_kinematic_derivative, measured_pair_response_tensor


class ResponseOperatorTests(unittest.TestCase):
    def setUp(self):
        self.j=np.array([[[1.,2],[0,1],[3,-1]],[[2.,0],[1,3],[-1,2]]])
        self.b=np.array([[.3,.1],[.0,.7]])
        self.pairs=np.array([[0,0],[0,1],[1,0],[1,1]])
        self.valid=np.ones((2,2),bool)

    def test_nonorthogonal_control_basis_requires_transport_and_preserves_magnitude(self):
        base,_=measured_pair_response_tensor(self.j,self.b,self.pairs,self.valid)
        transform=np.array([[2.,.3],[0,.5]])
        changed,_=measured_pair_response_tensor(self.j@transform,np.linalg.solve(transform,self.b),self.pairs,self.valid)
        np.testing.assert_allclose(changed,base,atol=1e-14)
        wrong,_=measured_pair_response_tensor(self.j@transform,self.b,self.pairs,self.valid)
        self.assertFalse(np.allclose(wrong,base))
        scaled,_=measured_pair_response_tensor(self.j,2*self.b,self.pairs,self.valid)
        np.testing.assert_allclose(scaled,4*base)

    def test_duplicate_control_does_not_change_object_when_measure_is_transported(self):
        base,_=measured_pair_response_tensor(self.j,self.b,self.pairs,self.valid)
        transform=np.array([[1.,1,0],[0,0,1]])
        transported=np.stack((self.b[0]/2,self.b[0]/2,self.b[1]))
        duplicate,_=measured_pair_response_tensor(self.j@transform,transported,self.pairs,np.ones((2,3),bool))
        np.testing.assert_allclose(duplicate,base)

    def test_root_rotation_is_covariant_and_pair_transpose_and_psd_hold(self):
        base,_=measured_pair_response_tensor(self.j,self.b,self.pairs,self.valid)
        rotation=np.array([[0.,-1,0],[1,0,0],[0,0,1]])
        rotated,_=measured_pair_response_tensor(rotation@self.j,self.b,self.pairs,self.valid)
        np.testing.assert_allclose(rotated,rotation@base@rotation.T,atol=1e-14)
        np.testing.assert_allclose(base[1],base[2].T)
        whole=np.block([[base[0],base[1]],[base[2],base[3]]])
        self.assertGreaterEqual(np.linalg.eigvalsh(whole).min(),-1e-14)

    def test_missing_excited_control_invalidates_pair_without_renormalizing(self):
        valid=self.valid.copy();valid[0,1]=False
        j=self.j.copy();j[0,:,1]=np.nan
        result,mask=measured_pair_response_tensor(j,self.b,self.pairs,valid)
        self.assertEqual(mask.tolist(),[False,False,False,True])
        self.assertTrue(np.isnan(result[1]).all())
        b=self.b.copy();b[1]=0
        result,mask=measured_pair_response_tensor(j,b,self.pairs,valid)
        self.assertTrue(mask.all())

    def test_central_difference_keeps_units_and_no_excitation_cannot_teach_rank(self):
        h=np.array([.01,.02])
        positive=(self.j*h[None,None,:]).transpose(0,2,1)
        negative=-positive
        actual,valid=central_kinematic_derivative(positive,negative,h,self.valid)
        np.testing.assert_allclose(actual,self.j)
        with self.assertRaisesRegex(ValueError,'NONZERO_MEASURE'):
            measured_pair_response_tensor(actual,np.zeros_like(self.b),self.pairs,valid)
