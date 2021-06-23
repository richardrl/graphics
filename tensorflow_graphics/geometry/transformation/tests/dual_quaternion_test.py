# Copyright 2020 The TensorFlow Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for dual quaternion."""

from absl.testing import flagsaver
from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow_graphics.geometry.transformation import dual_quaternion
from tensorflow_graphics.geometry.transformation.tests import test_helpers
from tensorflow_graphics.util import test_case


class DualQuaternionTest(test_case.TestCase):

  @parameterized.parameters(
      ((8,),),
      ((None, 8),),
  )
  def test_conjugate_exception_not_raised(self, *shape):
    self.assert_exception_is_not_raised(dual_quaternion.conjugate, shape)

  @parameterized.parameters(
      ("must have exactly 8 dimensions", (3,)),)
  def test_conjugate_exception_raised(self, error_msg, *shape):
    self.assert_exception_is_raised(dual_quaternion.conjugate, error_msg, shape)

  @flagsaver.flagsaver(tfg_add_asserts_to_graph=False)
  def test_conjugate_jacobian_preset(self):
    x_init = test_helpers.generate_preset_test_dual_quaternions()
    self.assert_jacobian_is_correct_fn(dual_quaternion.conjugate, [x_init])

  @flagsaver.flagsaver(tfg_add_asserts_to_graph=False)
  def test_conjugate_jacobian_random(self):
    x_init = test_helpers.generate_random_test_dual_quaternions()
    self.assert_jacobian_is_correct_fn(dual_quaternion.conjugate, [x_init])

  @flagsaver.flagsaver(tfg_add_asserts_to_graph=False)
  def test_conjugate_preset(self):
    x_init = test_helpers.generate_preset_test_dual_quaternions()
    x = tf.convert_to_tensor(value=x_init)
    y = tf.convert_to_tensor(value=x_init)

    x = dual_quaternion.conjugate(x)
    x_real, x_dual = tf.split(x, (4, 4), axis=-1)

    y_real, y_dual = tf.split(y, (4, 4), axis=-1)
    xyz_y_real, w_y_real = tf.split(y_real, (3, 1), axis=-1)
    xyz_y_dual, w_y_dual = tf.split(y_dual, (3, 1), axis=-1)
    y_real = tf.concat((-xyz_y_real, w_y_real), axis=-1)
    y_dual = tf.concat((-xyz_y_dual, w_y_dual), axis=-1)

    self.assertAllEqual(x_real, y_real)
    self.assertAllEqual(x_dual, y_dual)

  @parameterized.parameters(
      ((8,), (8,)),
      ((None, 8), (None, 8)),
  )
  def test_multiply_exception_not_raised(self, *shapes):
    self.assert_exception_is_not_raised(dual_quaternion.multiply, shapes)

  @parameterized.parameters(
      ("must have exactly 8 dimensions", (5,), (6,)),
      ("must have exactly 8 dimensions", (7,), (8,)),
  )
  def test_multiply_exception_raised(self, error_msg, *shape):
    self.assert_exception_is_raised(dual_quaternion.multiply, error_msg, shape)

  @flagsaver.flagsaver(tfg_add_asserts_to_graph=False)
  def test_multiply_jacobian_preset(self):
    x_1_init = test_helpers.generate_preset_test_dual_quaternions()
    x_2_init = test_helpers.generate_preset_test_dual_quaternions()

    self.assert_jacobian_is_correct_fn(dual_quaternion.multiply,
                                       [x_1_init, x_2_init])

  @flagsaver.flagsaver(tfg_add_asserts_to_graph=False)
  def test_multiply_jacobian_random(self):
    x_1_init = test_helpers.generate_random_test_dual_quaternions()
    x_2_init = test_helpers.generate_random_test_dual_quaternions()

    self.assert_jacobian_is_correct_fn(dual_quaternion.multiply,
                                       [x_1_init, x_2_init])

  @parameterized.parameters(
      ((8,),),
      ((None, 8),),
  )
  def test_inverse_exception_not_raised(self, *shape):
    """Tests that the shape exceptions are raised."""
    self.assert_exception_is_not_raised(dual_quaternion.inverse, shape)

  @parameterized.parameters(
      ("must have exactly 8 dimensions", (3,)),)
  def test_inverse_exception_raised(self, error_msg, *shape):
    """Tests that the shape exceptions are raised."""
    self.assert_exception_is_raised(dual_quaternion.inverse, error_msg, shape)

  @flagsaver.flagsaver(tfg_add_asserts_to_graph=False)
  def test_inverse_jacobian_preset(self):
    """Test the Jacobian of the inverse function."""
    x_init = test_helpers.generate_preset_test_dual_quaternions()

    self.assert_jacobian_is_correct_fn(dual_quaternion.inverse, [x_init])

  @flagsaver.flagsaver(tfg_add_asserts_to_graph=False)
  def test_inverse_jacobian_random(self):
    """Test the Jacobian of the inverse function."""
    x_init = test_helpers.generate_random_test_dual_quaternions()

    self.assert_jacobian_is_correct_fn(dual_quaternion.inverse, [x_init])

  def test_inverse_random(self):
    """Tests that multiplying with the inverse gives identity."""
    rand_dual_quaternion = test_helpers.generate_random_test_dual_quaternions()

    inverse_dual_quaternion = dual_quaternion.inverse(rand_dual_quaternion)
    final_dual_quaternion = dual_quaternion.multiply(rand_dual_quaternion,
                                                     inverse_dual_quaternion)
    tensor_shape = rand_dual_quaternion.shape[:-1]
    identity_dual_quaternion = np.array(
        (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0), dtype=np.float32)
    identity_dual_quaternion = np.tile(identity_dual_quaternion,
                                       tensor_shape + (1,))

    self.assertAllClose(
        final_dual_quaternion, identity_dual_quaternion, rtol=1e-3)

  @parameterized.parameters(
      ((8,),),
      ((None, 8),),
  )
  def test_norm_exception_not_raised(self, *shape):
    self.assert_exception_is_not_raised(dual_quaternion.norm, shape)

  @parameterized.parameters(
      ("must have exactly 8 dimensions", (3,)),)
  def test_norm_exception_raised(self, error_msg, *shape):
    self.assert_exception_is_raised(dual_quaternion.norm, error_msg, shape)

  @flagsaver.flagsaver(tfg_add_asserts_to_graph=False)
  def test_norm_jacobian_preset(self):
    x_init = test_helpers.generate_preset_test_dual_quaternions()
    self.assert_jacobian_is_correct_fn(dual_quaternion.norm, [x_init])

  @flagsaver.flagsaver(tfg_add_asserts_to_graph=False)
  def test_norm_jacobian_random(self):
    x_init = test_helpers.generate_random_test_dual_quaternions()
    self.assert_jacobian_is_correct_fn(dual_quaternion.norm, [x_init])


if __name__ == "__main__":
  test_case.main()
