import importlib
import os
import sys
import types
import unittest

import torch


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FLOWMAP_DIR = os.path.join(REPO_ROOT, "distillation_flowmap")
WANVA_ROOT = os.path.join(REPO_ROOT, "wan_va")
for path in (FLOWMAP_DIR, WANVA_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)


def test_module_does_not_put_wanva_utils_directory_on_sys_path():
    assert os.path.join(REPO_ROOT, "wan_va", "utils") not in sys.path


def load_config(module_name, **env):
    old_env = {key: os.environ.get(key) for key in env}
    try:
        for key, value in env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        sys.modules.pop(module_name, None)
        return importlib.import_module(module_name).cfg
    finally:
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


class IdentityScheduler:
    def apply_shift(self, value):
        return value


class RobotWinAblationPairModeConfigTest(unittest.TestCase):
    def test_stage1_parses_adjacent_grid_mode(self):
        cfg = load_config(
            "distillation_flowmap.config_robotwin_fullfinetune_stage1_warmup",
            FLOWMAP_PAIR_MODE="adjacent_grid",
            OPD_PAIR_MODE=None,
            FLOWMAP_ADJACENT_GRID="1000,750,500,250,0",
        )

        self.assertEqual(cfg.flowmap_pair_mode, "adjacent_grid")
        self.assertEqual(cfg.opd_pair_mode, "adjacent_grid")
        self.assertEqual(cfg.flowmap_adjacent_grid, [1000, 750, 500, 250, 0])

    def test_stage2_adjacent_opd_disables_low_noise_bias_by_default(self):
        cfg = load_config(
            "distillation_flowmap.config_robotwin_fullfinetune_stage2_anyflow",
            FLOWMAP_PAIR_MODE="adjacent_grid",
            OPD_PAIR_MODE="adjacent_grid",
            FLOWMAP_ADJACENT_GRID="1000,750,500,250,0",
            OPD_QUERY_BIAS=None,
        )

        self.assertEqual(cfg.flowmap_pair_mode, "adjacent_grid")
        self.assertEqual(cfg.opd_pair_mode, "adjacent_grid")
        self.assertEqual(cfg.opd_query_bias, "none")


class RobotWinAblationPairModeSamplerTest(unittest.TestCase):
    def _build_mixin(self, *, flowmap_pair_mode="adjacent_grid", opd_pair_mode=None):
        utils_stub = types.ModuleType("utils")
        utils_stub.data_seq_to_patch = lambda *args, **kwargs: None
        utils_stub.logger = None
        old_utils = sys.modules.get("utils")
        sys.modules["utils"] = utils_stub
        try:
            from distillation_flowmap.flowmap_step import FlowMapStepMixin
        finally:
            if old_utils is None:
                sys.modules.pop("utils", None)
            else:
                sys.modules["utils"] = old_utils

        class DummySampler(FlowMapStepMixin):
            pass

        sampler = DummySampler()
        sampler.diffusion_ratio = 0.0
        sampler.consistency_ratio = 0.0
        sampler.flowmap_ratio = 1.0
        sampler.config = types.SimpleNamespace(
            num_train_timesteps=1000,
            flowmap_pair_mode=flowmap_pair_mode,
            opd_pair_mode=opd_pair_mode or flowmap_pair_mode,
            flowmap_adjacent_grid=[1000, 750, 500, 250, 0],
        )
        sampler.train_scheduler_latent = IdentityScheduler()
        return sampler

    def test_adjacent_grid_sampler_uses_only_neighbor_edges(self):
        sampler = self._build_mixin()
        torch.manual_seed(7)

        t, r, is_diffusion = sampler.sample_timestep_mixed(
            batch_size=64,
            num_frames=3,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )

        observed = {
            (int(t_i), int(r_i))
            for t_i, r_i in zip(t[:, 0].tolist(), r[:, 0].tolist())
        }
        allowed = {
            (1000, 750),
            (750, 500),
            (500, 250),
            (250, 0),
        }
        self.assertTrue(observed)
        self.assertTrue(observed.issubset(allowed), observed)
        self.assertTrue(torch.equal(t, t[:, :1].expand_as(t)))
        self.assertTrue(torch.equal(r, r[:, :1].expand_as(r)))
        self.assertFalse(is_diffusion.any().item())

    def test_opd_pair_mode_can_override_flowmap_pair_mode(self):
        sampler = self._build_mixin(flowmap_pair_mode="arbitrary", opd_pair_mode="adjacent_grid")
        torch.manual_seed(11)

        t, r, _ = sampler.sample_timestep_mixed(
            batch_size=32,
            num_frames=2,
            dtype=torch.float32,
            device=torch.device("cpu"),
            pair_mode=sampler.config.opd_pair_mode,
        )

        observed = {
            (int(t_i), int(r_i))
            for t_i, r_i in zip(t[:, 0].tolist(), r[:, 0].tolist())
        }
        self.assertTrue(observed.issubset({
            (1000, 750),
            (750, 500),
            (500, 250),
            (250, 0),
        }))


if __name__ == "__main__":
    unittest.main()
