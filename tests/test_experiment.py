import copy
import json
from pathlib import Path
import tempfile
import unittest
import torch
from study import bootstrap
from study.sudoku_controls import Model
from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1 as Native
from models.losses import ACTLossHead
from scripts.experiment import configuration, make_config, specs
from scripts.launch import parser, resolve
from scripts.profile_plan import budget_epochs, safe
from scripts.repro_diagnostics import UpdateObserver
from tests.test_design import configuration as tiny, batch


class ExperimentTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)

    def test_mapping_depth_and_reused_preliminary(self):
        self.assertEqual(configuration('preliminary'),configuration('axis1-k1'))
        for k in (1,2,4,6):
            cfg=configuration(f'axis1-k{k}')
            self.assertEqual(cfg['arch']['L_cycles'],2*k)
            self.assertEqual(cfg['study']['attention_blocks_per_supervision'],6*(2*k+1))
            self.assertEqual(cfg['arch']['hidden_size'],256)
            self.assertFalse(cfg['arch']['mlp_t'])
        self.assertEqual(len(specs()),17)

    def test_argparse_one_configuration_and_overrides(self):
        a=parser().parse_args(['--axis','2','--k','4','--t','2','--batch','768','--epochs','20000'])
        cfg=resolve(a)
        self.assertEqual(cfg['arch']['L_cycles'],8);self.assertEqual(cfg['arch']['H_cycles'],2)
        self.assertEqual(cfg['eval_interval'],2000)
        a=parser().parse_args(['--axis','3','--batch','768','--epochs','20000'])
        with self.assertRaises(ValueError):resolve(a)
        with self.assertRaises(ValueError):make_config(batch=769)

    def test_depth_matched_forward_identity_and_every_untied_layer_trains(self):
        torch.manual_seed(17);tied=Model(tiny(untied=False))
        torch.manual_seed(17);untied=Model(tiny(untied=True))
        torch.manual_seed(17);native=Native(tiny())
        data=batch()
        outputs=[]
        for m in (tied,untied,native):
            m.eval();_,out=m(m.initial_carry(data),data);outputs.append(out['logits'])
        torch.testing.assert_close(outputs[0],outputs[1],rtol=0,atol=0)
        torch.testing.assert_close(outputs[0],outputs[2],rtol=0,atol=0)
        self.assertEqual(len(untied.inner.L_level.layers),2*2*(2+1))
        self.assertNotEqual(untied.inner.L_level.layers[0].mlp.gate_up_proj.weight.data_ptr(),
                            untied.inner.L_level.layers[2].mlp.gate_up_proj.weight.data_ptr())
        head=ACTLossHead(untied,'stablemax_cross_entropy');head.train()
        opt=torch.optim.AdamW(head.parameters(),lr=1e-4)
        with UpdateObserver(head,opt,2) as observer:
            _,loss,*_=head(carry=head.initial_carry(data),batch=data,return_keys=[])
            (loss/2).backward();opt.step()
        stats=observer.finish()
        self.assertEqual(len(stats['layers']),12)
        self.assertEqual(set(stats['latents']),{'z','y'})
        for layer in untied.inner.L_level.layers:
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in layer.parameters()))

    def test_runtime_budget_keeps_ten_evals_and_minimum_training(self):
        r=dict(projected_training_hours=6,projected_evaluation_hours=1)
        self.assertEqual(budget_epochs([r],768),50000)
        self.assertEqual(budget_epochs([dict(projected_training_hours=12,projected_evaluation_hours=1)],768),29160)
        with self.assertRaises(RuntimeError):budget_epochs([dict(projected_training_hours=1,projected_evaluation_hours=9)],768)
        self.assertFalse(safe(dict(status='passed',peak_memory_fraction=.81,projected_compute_hours=5)))
        self.assertFalse(safe(dict(status='passed',peak_memory_fraction=.6,projected_compute_hours=float('nan'))))


if __name__=='__main__':unittest.main()
