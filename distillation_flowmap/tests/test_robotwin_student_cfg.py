import os
import sys
import types
import unittest

import torch


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FLOWMAP_DIR = os.path.join(REPO_ROOT, "distillation_flowmap")
for path in (REPO_ROOT, FLOWMAP_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)


class _FlexAttnFunc:
    attention_mask = object()
    cross_attention_mask = object()


class _DummyStudentModel:
    def __init__(self):
        self.call_batch_sizes = []
        self.call_text_values = []

    def __call__(self, input_dict, train_mode=True, r_timestep=None, action_r_timestep=None):
        text_emb = input_dict["latent_dict"]["text_emb"]
        self.call_batch_sizes.append(text_emb.shape[0])
        self.call_text_values.append(text_emb.detach().clone())
        value = text_emb[:, :1, :1].reshape(text_emb.shape[0], 1, 1, 1, 1)
        return value, None


class _Harness:
    def __init__(self, serial):
        self.config = types.SimpleNamespace(opd_serial_student_cfg=serial)

    def _extract_video_v(self, value, ref_shape, batch_size):
        return value


class StudentCfgForwardTest(unittest.TestCase):
    def setUp(self):
        self.old_modules = sys.modules.get("modules")
        self.old_modules_model = sys.modules.get("modules.model")
        modules_stub = types.ModuleType("modules")
        modules_model_stub = types.ModuleType("modules.model")
        modules_model_stub.FlexAttnFunc = _FlexAttnFunc
        sys.modules["modules"] = modules_stub
        sys.modules["modules.model"] = modules_model_stub
        sys.modules.pop("distillation_flowmap.flowmap_step", None)
        from distillation_flowmap.flowmap_step import FlowMapStepMixin

        self.method = FlowMapStepMixin._student_cfg_forward

    def tearDown(self):
        if self.old_modules is None:
            sys.modules.pop("modules", None)
        else:
            sys.modules["modules"] = self.old_modules
        if self.old_modules_model is None:
            sys.modules.pop("modules.model", None)
        else:
            sys.modules["modules.model"] = self.old_modules_model

    def _input(self):
        text = torch.tensor([[[3.0]]])
        empty = torch.tensor([[[1.0]]])
        latent_dict = {
            "noisy_latents": torch.zeros(1, 1, 1, 1, 1),
            "latent": torch.zeros(1, 1, 1, 1, 1),
            "timesteps": torch.zeros(1, 1),
            "cond_timesteps": torch.zeros(1, 1),
            "text_emb": text,
        }
        action_dict = {
            "noisy_latents": torch.zeros(1, 1, 1, 1, 1),
            "latent": torch.zeros(1, 1, 1, 1, 1),
            "timesteps": torch.zeros(1, 1),
            "cond_timesteps": torch.zeros(1, 1),
            "text_emb": text,
        }
        input_dict = {
            "latent_dict": latent_dict,
            "action_dict": action_dict,
            "chunk_size": 1,
            "window_size": 1,
        }
        return input_dict, empty

    def test_serial_cfg_matches_batched_cfg_without_doubling_batch(self):
        input_dict, empty = self._input()
        r_timestep = torch.zeros(1, 1)
        action_r_timestep = torch.zeros(1, 1)
        cfg_scale = 2.0

        batched_model = _DummyStudentModel()
        batched_out = self.method(
            _Harness(serial=False),
            batched_model,
            input_dict,
            empty,
            cfg_scale,
            1,
            None,
            r_timestep,
            action_r_timestep,
            force_cfg=True,
        )

        serial_model = _DummyStudentModel()
        serial_out = self.method(
            _Harness(serial=True),
            serial_model,
            input_dict,
            empty,
            cfg_scale,
            1,
            None,
            r_timestep,
            action_r_timestep,
            force_cfg=True,
        )

        self.assertTrue(torch.allclose(serial_out, batched_out))
        self.assertTrue(torch.allclose(serial_out, torch.tensor([[[[[5.0]]]]])))
        self.assertEqual(batched_model.call_batch_sizes, [2])
        self.assertEqual(serial_model.call_batch_sizes, [1, 1])


if __name__ == "__main__":
    unittest.main()
