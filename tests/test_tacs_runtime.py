import unittest

from comet_tasmoe.runtime.overlay_types import WindowView
from comet_tasmoe.runtime.replica_state import ReplicaState
from comet_tasmoe.runtime.tacs_runtime import Telemetry, _bank_for_core, tacs_step


ARCH = {"mesh": {"cores": [8, 8]}, "dram": {"banks": [32, 32, 8]}}


class BankMappingTest(unittest.TestCase):
    def test_projects_core_coordinates_and_preserves_layer(self):
        self.assertEqual(_bank_for_core(21, "B(12,7,3)", ARCH), "B(22,8,3)")

    def test_projects_last_core_to_last_bank(self):
        self.assertEqual(_bank_for_core(63, "B(0,0,7)", ARCH), "B(31,31,7)")

    def test_invalid_bank_returns_no_destination(self):
        self.assertEqual(_bank_for_core(1, "", ARCH), "")

    def test_tacs_migration_changes_bank(self):
        tele = Telemetry([90.0, 60.0], [55.0], [0.9, 0.1], 0.1, 100)
        placement = {
            "experts": {"E0": {"cores": [0]}},
            "tensors": {"W_E0_0": {"dram_bank": "B(0,0,3)", "size_bytes": 1024}},
        }
        _, migrations = tacs_step(
            tele,
            placement,
            ReplicaState(),
            {"max_parallel_migrations": 1},
            WindowView(0, [0]),
            ARCH,
        )
        row = migrations.to_dict(orient="records")[0]
        self.assertEqual(row["src"], "B(0,0,3)")
        self.assertEqual(row["dst"], "B(4,0,3)")
        self.assertNotEqual(row["src"], row["dst"])

    def test_tacs_skips_no_op_migration(self):
        arch = {"mesh": {"cores": [2, 1]}, "dram": {"banks": [1, 1, 1]}}
        tele = Telemetry([90.0, 60.0], [55.0], [0.9, 0.1], 0.1, 100)
        placement = {
            "experts": {"E0": {"cores": [0]}},
            "tensors": {"W_E0_0": {"dram_bank": "B(0,0,0)", "size_bytes": 1024}},
        }
        _, migrations = tacs_step(
            tele,
            placement,
            ReplicaState(),
            {"max_parallel_migrations": 1},
            WindowView(0, [0]),
            arch,
        )
        self.assertTrue(migrations.empty)


if __name__ == "__main__":
    unittest.main()
