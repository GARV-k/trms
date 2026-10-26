"""CPU/AOT correctness and full-driver restart tests; not CUDA-graph validation."""
import copy
import unittest
from unittest.mock import patch
import torch
from scripts.single4_compile import ManagedCompile, own_tensors
from scripts.experiment import configuration
from scripts.repro_config import configuration as baseline
from tests import test_repro as original


class WrapperTests(unittest.TestCase):
    def test_owned_outputs_keep_gradients_and_do_not_alias(self):
        x = torch.tensor([2.0], requires_grad=True)
        output = dict(loss=x.square().sum(), carry=x.detach())
        owned = own_tensors(output)
        owned['loss'].backward()
        self.assertEqual(x.grad.item(),4)
        with torch.no_grad(): x.fill_(9)
        self.assertEqual(owned['carry'].item(),2)
        self.assertFalse(owned['carry'].requires_grad)

    def test_copy_does_not_keep_callable_bound_to_old_weights(self):
        raw = torch.nn.Linear(2,2)
        model = ManagedCompile(raw,'eager')
        inputs = torch.ones(1,2)
        model(inputs)
        clone = copy.deepcopy(model)
        self.assertIsNone(clone._compiled_call)
        with torch.no_grad(): clone._orig_mod.weight.add_(1)
        self.assertFalse(torch.equal(model(inputs),clone(inputs)))
        self.assertIs(clone._compiled_call,clone._orig_mod)
        self.assertEqual(set(model.state_dict()),{'_orig_mod.weight','_orig_mod.bias'})

    def test_aot_training_loss_gradients_and_recurrent_carry(self):
        torch.set_num_threads(1)
        raw = original.ACTLossHead(original.Native(original.configuration()),'stablemax_cross_entropy')
        wrapped = ManagedCompile(copy.deepcopy(raw),'reduce-overhead')
        raw.train(); wrapped.train()
        batch = original.batch()
        ca = raw.initial_carry(batch); cb = wrapped.initial_carry(batch)
        compile_model = torch.compile
        with patch.object(torch,'compile',lambda model,**kw: compile_model(model,backend='aot_eager')), \
             patch.object(torch.compiler,'cudagraph_mark_step_begin'):
            for step in range(3):
                raw.zero_grad(); wrapped.zero_grad()
                torch.manual_seed(19+step)
                ca, la, *_ = raw(carry=ca,batch=batch,return_keys=[])
                (la/2).backward()
                torch.manual_seed(19+step)
                cb, lb, *_ = wrapped(carry=cb,batch=batch,return_keys=[])
                (lb/2).backward()
                torch.testing.assert_close(la,lb)
                torch.testing.assert_close(ca.inner_carry.z_H,cb.inner_carry.z_H)
                torch.testing.assert_close(ca.inner_carry.z_L,cb.inner_carry.z_L)
                for a,b in zip(raw.parameters(),wrapped.parameters()):
                    if a.grad is not None: torch.testing.assert_close(a.grad,b.grad)
        torch.compiler.reset()


class WrappedDriverTests(original.DriverTests):
    def run_driver(self, *args, **kwargs):
        loss_head = original.ACTLossHead
        def wrap(*a, **kw): return ManagedCompile(loss_head(*a, **kw),'reduce-overhead')
        # The inherited driver fixture replaces torch.device to run on CPU;
        # that replacement is incompatible with Dynamo's isinstance checks.
        # Test real AOT compilation separately above, outside that fixture.
        def cpu_compile(model, **kw): return model
        with patch.object(original,'ACTLossHead',wrap), patch.object(torch,'compile',cpu_compile), \
             patch.object(torch.compiler,'cudagraph_mark_step_begin'):
            return super().run_driver(*args, **kwargs)


if __name__=='__main__': unittest.main()
