import tempfile
import unittest
from pathlib import Path

try:
    import h5py
    import hdf5plugin
    import numpy as np
except ImportError:
    h5py = None


@unittest.skipIf(h5py is None, "HDF5 dependencies are not installed")
class RechunkCubeHDF5Test(unittest.TestCase):
    def test_rechunk_is_lossless_and_preserves_columns(self):
        from scripts.rechunk_cube_hdf5 import rechunk_cube_hdf5

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.h5"
            output = root / "output.h5"
            rng = np.random.default_rng(42)
            with h5py.File(source, "w") as handle:
                handle.create_dataset(
                    "pixels",
                    data=rng.integers(0, 256, (12, 8, 8, 3), dtype=np.uint8),
                    chunks=(6, 8, 8, 3),
                    **hdf5plugin.Blosc(cname="lz4", clevel=5),
                )
                handle.create_dataset("action", data=rng.normal(size=(12, 5)))
                handle.create_dataset(
                    "observation", data=rng.normal(size=(12, 28))
                )
                handle.create_dataset("ep_len", data=np.array([6, 6]))
                handle.create_dataset("ep_offset", data=np.array([0, 6]))

            result = rechunk_cube_hdf5(
                source,
                output,
                pixel_chunk=3,
                copy_rows=6,
                verify_samples=12,
            )

            self.assertEqual(result["verification"]["episodes"], 2)
            self.assertEqual(result["verification"]["transitions"], 12)
            self.assertEqual(len(result["output_sha256"]), 64)
            with h5py.File(source, "r") as left, h5py.File(output, "r") as right:
                self.assertEqual(right["pixels"].chunks, (3, 8, 8, 3))
                for name in left:
                    np.testing.assert_array_equal(left[name][:], right[name][:])


if __name__ == "__main__":
    unittest.main()
