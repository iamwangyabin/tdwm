import unittest

from tdwm.training.td_jepa import apply_tdjepa_cube_overrides


class TDJEPATrainingTest(unittest.TestCase):
    def test_applies_released_cube_sweep_defaults(self):
        base = {
            "data": {"dataset_root": "old", "load_n_episodes": 1000},
            "agent": {"compile": True, "train": {"batch_size": 256}},
        }

        config = apply_tdjepa_cube_overrides(
            base,
            data_root="data",
            work_dir="outputs/td_jepa",
            seed=3917,
            train_steps=100,
            load_episodes=10,
            loader_workers=2,
            log_every_updates=10,
            checkpoint_every_steps=50,
            compile_model=False,
            implementation_revision="abc",
        )

        self.assertEqual(base["data"]["dataset_root"], "old")
        self.assertEqual(config["seed"], 3917)
        self.assertEqual(config["data"]["load_n_episodes"], 10)
        self.assertEqual(config["data"]["num_workers"], 2)
        self.assertFalse(config["agent"]["compile"])
        self.assertEqual(config["agent"]["train"]["bc_coeff"], 3.0)
        self.assertEqual(config["agent"]["train"]["psi_ortho_coef"], 1.0)
        self.assertEqual(config["agent"]["train"]["lr_psi"], 1.0e-4)
        self.assertEqual(config["evaluations"], [])


if __name__ == "__main__":
    unittest.main()
