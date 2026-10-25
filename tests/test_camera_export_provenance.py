"""A policy score from a different camera must not authorize the current export."""
import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest


def test_export_rejects_old_camera_then_accepts_matching_camera(tmp_path, monkeypatch):
    path = Path(__file__).parents[1] / "tools/knockdown/export_policy.py"
    spec = importlib.util.spec_from_file_location("export_camera_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"not loaded: validation must happen before environment creation")
    evaluation = dict(checkpoint_sha256=module.sha(checkpoint), contract_id=module.C.CONTRACT_ID,
                      contract_source_sha256=module.C.source_file_sha256(),
                      perception_files={name: module.sha(Path(module.C.__file__).with_name(name))
                                        for name in ("egg_vision.py", "camera_geometry.py")},
                      source_model_sha256=module.file_sha(module.SOURCE_XML),
                      perception="rendered_rgb_detector", episodes=200,
                      nominal_dynamics=False, meets_target=True)
    summary = tmp_path / "summary.json"
    monkeypatch.setattr(sys, "argv", [str(path), "--checkpoint", str(checkpoint), "--out", str(tmp_path / "export"),
                                     "--evaluation-summary", str(summary), "--checkpoint-contract-id", module.C.CONTRACT_ID])
    class ReachedEnvironment(Exception):
        pass
    def stop_before_physics(*args, **kwargs):
        raise ReachedEnvironment
    monkeypatch.setattr(module, "ManagerBasedRlEnv", stop_before_physics)
    # Old pre-calibration summaries lack the view entirely; altered views also fail.
    for camera in (None, copy.deepcopy(module.camera_manifest())):
        if camera is not None:
            camera["fovy_deg"] -= 2
        evaluation["rendered_camera"] = camera
        summary.write_text(json.dumps(evaluation))
        with pytest.raises(SystemExit) as caught:
            module.main()
        assert caught.value.code == 2
    evaluation["rendered_camera"] = module.camera_manifest()
    summary.write_text(json.dumps(evaluation))
    with pytest.raises(ReachedEnvironment):
        module.main()
