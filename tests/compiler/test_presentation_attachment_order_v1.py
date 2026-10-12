import unittest
import numpy as np
from compiler.realsas_compiler_core.rigid_attachment_surface_anchor_v1 import (
 derive_surface_binding,apply_surface_rigid_3d,SurfaceAddressV1,AttachmentSurfaceBindingV1)
from compiler.realsas_compiler_core.single_m_caa_v1 import render_m_depth
from compiler.realsas_compiler_core.presentation_part_order_v1 import compile_part_ids
from tools.demo.render_knight_source_motion_v3 import preview_samples

class PresentationTests(unittest.TestCase):
 def test_contact_address_and_3d_rigidity_survive_wrong_initial_offset(self):
  rest=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,0],[.5,0,0],[0,.5,0]],float)
  faces=np.array([[0,1,2],[3,4,5]]);R=np.array([[0,-1,0],[1,0,0],[0,0,1]],float)
  pose=rest.copy();pose[:3]=rest[:3]@R.T+[2,3,4];pose[3:]+=17
  binding=AttachmentSurfaceBindingV1(1,np.arange(3,6),SurfaceAddressV1(1,(1.,0.,0.)),SurfaceAddressV1(0,(1.,0.,0.)),0.)
  out,proof=apply_surface_rigid_3d(pose,rest=rest,faces=faces,binding=binding,rotation=R)
  np.testing.assert_allclose(out[3:],rest[3:]@R.T+[2,3,4]);np.testing.assert_array_equal(out[:3],pose[:3])
  self.assertLess(proof['anchor_transport_residual_world'],1e-12)
  self.assertAlmostEqual(np.linalg.norm(out[4]-out[3]),.5)
 def test_holder_address_cannot_attach_to_unrelated_closer_body_surface(self):
  rest=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,.1],[1,0,.1],[0,1,.1],[0,0,0],[.5,0,0],[0,.5,0]],float)
  faces=np.array([[0,1,2],[3,4,5],[6,7,8]]);owners=np.array([0]*6+[1]*3)
  binding=derive_surface_binding(rest_xyz=rest,faces=faces,attachment_vertex_owner=owners,attachment_index=1,holder_face_indices=[1])
  self.assertEqual(binding.holder.face_index,1);self.assertAlmostEqual(binding.setup_surface_gap,.1)
 def render(self,ranks=None,depths=None):
  pos=np.array([[1,1],[6,1],[1,6],[1,1],[6,1],[1,6]],float)
  faces=np.array([[0,1,2],[3,4,5]])
  def sample(fi,b):return np.array([[0,0,1,1] if i==0 else [1,0,0,1] for i in fi],float)
  return render_m_depth(positions=pos,depths=np.array([1]*3+[2]*3,float) if depths is None else depths,faces=faces,uv=np.zeros((6,2)),texture=np.zeros((8,8,4),np.uint8),resolution=8,sample_material=sample,face_draw_rank=ranks)
 def test_whole_region_precedence_is_explicit_and_geometric_default_survives(self):
  a=self.render();b=self.render([1,0]);np.testing.assert_array_equal(a.straight_rgba_u8[2,2],[0,0,255,255]);np.testing.assert_array_equal(b.straight_rgba_u8[2,2],[255,0,0,255])
 def test_artistic_order_does_not_erase_geometric_depth_tie_diagnostics(self):
  r=self.render([1,0],np.ones(6));self.assertGreater(r.unresolved_depth_tie_count,0)
 def test_nested_components_with_same_skin_owner_keep_local_depth(self):
  faces=np.array([[0,1,2],[3,4,5],[6,7,8]]);W=np.zeros((9,2));W[:6,0]=1;W[6:,1]=1
  vp,fp,_=compile_part_ids(9,faces,np.zeros(9,dtype=int),W);self.assertEqual(fp[0],fp[1]);self.assertNotEqual(fp[0],fp[2])
 def test_30fps_four_second_playback_holds_nonloop_terminal_without_slowdown(self):
  playback,times,indices=preview_samples(1.2,loop=False);self.assertEqual(len(playback),120);self.assertAlmostEqual(playback[30],1.);self.assertAlmostEqual(times[indices[-1]],1.2)
  playback,times,indices=preview_samples(5/6,loop=True);self.assertAlmostEqual(times[indices[25]],0.);self.assertEqual(len(times),25)

if __name__=='__main__':unittest.main()
