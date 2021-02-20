from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.keras.optimizer_v2 import optimizer_v2 
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_resource_variable_ops
from tensorflow.python.training import gen_training_ops
from tensorflow.python.util.tf_export import keras_export


@keras_export("keras.optimizers.SGD")
class SGD_TUD(optimizer_v2.OptimizerV2):
 
  _HAS_AGGREGATE_GRAD = True

  def __init__(self,
               learning_rate=0.01,
               name="SGD_TUD",
               **kwargs):
    super(SGD_TUD, self).__init__(name, **kwargs)
    self._set_hyper("learning_rate", kwargs.get("lr", learning_rate))

  def _resource_apply_dense(self, grad, var, apply_state=None):
    var_device, var_dtype = var.device, var.dtype.base_dtype
    coefficients = ((apply_state or {}).get((var_device, var_dtype))
                    or self._fallback_apply_state(var_device, var_dtype))

    return gen_training_ops.ResourceApplyGradientDescent(
        var=var.handle,
        alpha=coefficients["lr_t"],
        delta=grad,
        use_locking=self._use_locking)

  def _resource_apply_sparse_duplicate_indices(self, grad, var, indices,
                                               **kwargs):
    var_device, var_dtype = var.device, var.dtype.base_dtype
    coefficients = (kwargs.get("apply_state", {}).get((var_device, var_dtype))
                    or self._fallback_apply_state(var_device, var_dtype))

    return gen_resource_variable_ops.ResourceScatterAdd(
        resource=var.handle,
        indices=indices,
        updates=-grad * coefficients["lr_t"])

  def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
    var_device, var_dtype = var.device, var.dtype.base_dtype
    coefficients = ((apply_state or {}).get((var_device, var_dtype))
                    or self._fallback_apply_state(var_device, var_dtype))

    return gen_training_ops.ResourceSparseApplyKerasMomentum(
        var=var.handle,
        lr=coefficients["lr_t"],
        grad=grad,
        indices=indices,
        use_locking=self._use_locking,
        )

  def get_config(self):
    config = super(SGD_TUD, self).get_config()
    config.update({
        "learning_rate": self._serialize_hyperparameter("learning_rate"),
    })
    return config