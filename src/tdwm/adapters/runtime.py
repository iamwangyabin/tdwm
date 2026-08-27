from __future__ import annotations

import importlib.metadata


def prepare_cloud_runtime() -> dict[str, object] | None:
    """Bridge known version gaps in the managed cloud image."""

    compatibility: dict[str, object] = {}
    try:
        pyarrow_version = importlib.metadata.version("pyarrow")
        hotfix_version = importlib.metadata.version("pyarrow-hotfix")
    except importlib.metadata.PackageNotFoundError:
        pass
    else:
        if int(pyarrow_version.split(".", 1)[0]) >= 21 and hotfix_version == "0.6":
            import pyarrow as pa

            class LegacyArrowPyExtension(pa.ExtensionType):
                def __init__(self) -> None:
                    super().__init__(pa.null(), "arrow.py_extension_type")

                def __arrow_ext_serialize__(self) -> bytes:
                    return b""

                @classmethod
                def __arrow_ext_deserialize(cls, storage_type, serialized):
                    return cls()

            try:
                pa.register_extension_type(LegacyArrowPyExtension())
            except pa.ArrowKeyError:
                pass
            compatibility["pyarrow_hotfix"] = {
                "reason": "pyarrow-hotfix 0.6 expects a legacy extension removed by PyArrow",
                "pyarrow": pyarrow_version,
                "version": hotfix_version,
            }

    return compatibility or None
