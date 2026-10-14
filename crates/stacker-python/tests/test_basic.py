"""Tests of the Python API. Run after `maturin develop --release`:

    cd crates/stacker-python
    pip install pytest
    python -m pytest tests/test_basic.py

They need numpy and pytest only; the PNG frames are written by a small helper below.
"""

import struct
import zlib

import numpy as np
import pytest

import zstackr


# ── helpers ──────────────────────────────────────────────────────────────────


def half_sharp(sharp_left, dtype=np.uint8, height=32, width=48):
    """A frame whose left (or right) half is a checkerboard and the rest flat gray."""
    y, x = np.mgrid[0:height, 0:width]
    sharp = (x < width // 2) if sharp_left else (x >= width // 2)
    checker = np.where((x + y) % 2 == 0, 230, 30)
    plane = np.where(sharp, checker, 110).astype(np.uint8)
    frame = np.repeat(plane[:, :, None], 3, axis=2)
    if dtype == np.uint16:
        return frame.astype(np.uint16) * 257
    if dtype == np.float32:
        return (frame / 255.0).astype(np.float32)
    return frame


def write_png(path, rgb):
    """Write an 8-bit RGB PNG without any imaging library."""
    height, width, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[row].tobytes() for row in range(height))

    def chunk(tag, data):
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


@pytest.fixture
def quiet():
    """Settings that make a tiny synthetic stack deterministic."""
    return zstackr.PyStackingSettings(
        alignment_mode="None",
        correct_brightness=False,
        auto_cull=False,
        pyramid_levels=3,
        strata_base_radius=8,
    )


@pytest.fixture
def frame_files(tmp_path):
    folder = tmp_path / "frames"
    folder.mkdir()
    paths = []
    for k in range(3):
        rgb = half_sharp(k % 2 == 0)
        path = folder / f"frame_{k}.png"
        write_png(path, rgb)
        paths.append(str(path))
    return paths


# ── the module ───────────────────────────────────────────────────────────────


def test_module_exposes_the_documented_api():
    for name in (
        "stack_files",
        "batch_stack",
        "load_config",
        "save_config",
        "load_image",
        "stack_arrays",
        "PyPipelineParams",
        "PyStackingSettings",
        "PyPreprocessingSettings",
        "PyImageSavingSettings",
    ):
        assert hasattr(zstackr, name), name
    assert isinstance(zstackr.__version__, str)


# ── settings ─────────────────────────────────────────────────────────────────


def test_default_settings_have_the_documented_values():
    s = zstackr.PyStackingSettings()
    assert s.alignment_mode == "Registration"
    assert s.optimizer == "Auto"
    assert s.compute_device == "CPU"
    assert s.pyramid_levels == 8
    assert s.settings_version == 2
    assert s.relief_use_multigrid is False
    assert s.image_saving.output_format == "TIFF"
    assert s.image_saving.bit_depth == 16
    assert s.preprocessing.pre_rotation == 0


def test_relief_engine_and_use_gpu_are_views_of_the_real_keys():
    s = zstackr.PyStackingSettings(relief_engine="multigrid", use_gpu=True)
    assert s.relief_use_multigrid is True
    assert s.compute_device == "GPU"
    s.relief_engine = "guided_filter"
    s.use_gpu = False
    assert s.relief_use_multigrid is False
    assert s.compute_device == "CPU"
    assert s.relief_engine == "guided_filter"
    assert s.use_gpu is False


def test_settings_equality_and_nested_assignment():
    a = zstackr.PyStackingSettings()
    b = zstackr.PyStackingSettings()
    assert a == b
    b.pyramid_levels = 5
    assert a != b
    b.image_saving = zstackr.PyImageSavingSettings(output_format="PNG", bit_depth=8)
    assert b.image_saving.output_format == "PNG"


def test_invalid_settings_raise_value_error_naming_the_problem(quiet, frame_files, tmp_path):
    params = zstackr.PyPipelineParams(frame_files, str(tmp_path / "out.png"), tile_size=0)
    for bad, needle in (
        (zstackr.PyStackingSettings(alignment_mode="Sideways"), "Sideways"),
        (zstackr.PyStackingSettings(optimizer="fast"), "fast"),
        (zstackr.PyStackingSettings(pyramid_levels=99), "pyramid_levels"),
        (zstackr.PyStackingSettings(strata_detail_focus=0), "strata_detail_focus"),
        (zstackr.PyStackingSettings(tile_size=10), "tile_size"),
    ):
        with pytest.raises(ValueError, match=needle):
            zstackr.stack_files(params, bad)


def test_config_round_trip(tmp_path):
    path = str(tmp_path / "config.toml")
    settings = zstackr.PyStackingSettings(
        alignment_mode="Affine",
        pyramid_levels=6,
        relief_engine="multigrid",
        auto_cull_threshold_pct=2.5,
        compute_device="GPU",
        image_saving=zstackr.PyImageSavingSettings(output_format="JPEG", jpeg_quality=80),
    )
    zstackr.save_config(path, settings)
    assert zstackr.load_config(path) == settings


def test_load_config_accepts_legacy_keys_and_warns_about_unknown_ones(tmp_path):
    path = tmp_path / "old.toml"
    path.write_text('dmap_estimation_radius = 9\nuse_gpu = true\nflavour = "mint"\n')
    with pytest.warns(UserWarning, match="flavour"):
        settings = zstackr.load_config(str(path))
    assert settings.relief_estimation_radius == 9
    assert settings.compute_device == "GPU"


def test_load_config_reports_missing_and_malformed_files(tmp_path):
    with pytest.raises(RuntimeError, match="failed to read"):
        zstackr.load_config(str(tmp_path / "missing.toml"))
    bad = tmp_path / "bad.toml"
    bad.write_text("this is = = not toml")
    with pytest.raises(ValueError, match="failed to parse"):
        zstackr.load_config(str(bad))


# ── arrays ───────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32])
@pytest.mark.parametrize("mode", ["apex", "relief", "strata"])
def test_stack_arrays_keeps_dtype_and_shape(quiet, dtype, mode):
    frames = [half_sharp(True, dtype), half_sharp(False, dtype)]
    out = zstackr.stack_arrays(frames, quiet, mode=mode)
    assert out.dtype == dtype
    assert out.shape == (32, 48, 3)


def test_stack_arrays_apex_keeps_both_sharp_halves(quiet):
    out = zstackr.stack_arrays([half_sharp(True), half_sharp(False)], quiet)
    left = out[8:24, 4:20].astype(float)
    right = out[8:24, 28:44].astype(float)
    assert left.std() > 20 and right.std() > 20


def test_identical_flat_frames_come_back_unchanged_in_both_encodings(quiet):
    flat = np.full((24, 24, 3), 0.25, dtype=np.float32)
    for encoding in ("srgb", "linear"):
        out = zstackr.stack_arrays([flat, flat], quiet, encoding=encoding)
        assert np.allclose(out, flat, atol=1e-4), encoding


def test_a_dark_flat_stack_is_a_fixed_point_of_both_encodings(quiet):
    # Each encoding returns what it was given: the output is encoded with the function the
    # input was decoded with, so a flat frame comes back unchanged either way.
    dark = np.full((24, 24, 3), 0.1, dtype=np.float32)
    srgb = zstackr.stack_arrays([dark, dark], quiet, encoding="srgb")
    linear = zstackr.stack_arrays([dark, dark], quiet, encoding="linear")
    assert np.allclose(srgb, dark, atol=1e-4) and np.allclose(linear, dark, atol=1e-4)


def test_stack_arrays_rejects_bad_input(quiet):
    good = half_sharp(True)
    with pytest.raises(ValueError, match="must not be empty"):
        zstackr.stack_arrays([], quiet)
    with pytest.raises(ValueError, match="not available"):
        zstackr.stack_arrays([good], quiet, mode="ai")
    with pytest.raises(ValueError, match="encoding"):
        zstackr.stack_arrays([good], quiet, encoding="gamma")
    with pytest.raises(ValueError, match="dtype"):
        zstackr.stack_arrays([good, half_sharp(False, np.float32)], quiet)
    with pytest.raises(ValueError, match="C-contiguous"):
        zstackr.stack_arrays([good[:, ::-1]], quiet)
    with pytest.raises(ValueError, match="same size"):
        zstackr.stack_arrays([good, good[:, :40].copy()], quiet)
    with pytest.raises(ValueError):
        zstackr.stack_arrays([good[:, :, 0].copy()], quiet)


def test_stack_arrays_applies_stack_every_nth(quiet):
    # Frames 0 and 2 are both sharp on the left only; with every second frame the right half
    # of the result stays flat.
    frames = [half_sharp(True), half_sharp(False), half_sharp(True)]
    quiet.stack_every_nth = 2
    out = zstackr.stack_arrays(frames, quiet)
    assert out.shape == (32, 48, 3)
    assert out[8:24, 28:44].astype(float).std() < 5
    assert out[8:24, 4:20].astype(float).std() > 20


def test_stack_arrays_progress_callback(quiet):
    seen = []
    zstackr.stack_arrays(
        [half_sharp(True), half_sharp(False)],
        quiet,
        progress=lambda stage, current, total: seen.append((stage, current, total)),
    )
    stages = [s for s, _, _ in seen]
    assert stages[0] == "align" and "fuse" in stages
    assert all(total == 1000 for _, _, total in seen)
    assert seen[-1] == ("fuse", 1000, 1000)


def test_a_failing_progress_callback_does_not_abort_the_stack(quiet, capsys):
    def broken(stage, current, total):
        raise RuntimeError("boom")

    out = zstackr.stack_arrays([half_sharp(True), half_sharp(False)], quiet, progress=broken)
    assert out.shape == (32, 48, 3)
    assert "boom" in capsys.readouterr().err


# ── files ────────────────────────────────────────────────────────────────────


def test_load_image_returns_uint16_codes(tmp_path):
    path = tmp_path / "one.png"
    write_png(path, half_sharp(True))
    image = zstackr.load_image(str(path))
    assert image.dtype == np.uint16 and image.shape == (32, 48, 3)
    assert np.array_equal(image, half_sharp(True, np.uint16))
    with pytest.raises(RuntimeError):
        zstackr.load_image(str(tmp_path / "missing.png"))


@pytest.mark.parametrize("tile_size", [0, 64])
@pytest.mark.parametrize("name", ["stacked.png", "stacked.tif", "stacked.jpg"])
def test_stack_files_writes_the_format_of_the_extension(quiet, frame_files, tmp_path, name, tile_size):
    output = tmp_path / "out" / name
    params = zstackr.PyPipelineParams(frame_files, str(output), mode="apex", tile_size=tile_size)
    zstackr.stack_files(params, quiet)
    assert output.exists()
    if name.endswith(".png"):
        assert zstackr.load_image(str(output)).shape == (32, 48, 3)


def test_stack_files_reports_progress_in_stage_order(quiet, frame_files, tmp_path):
    seen = []
    params = zstackr.PyPipelineParams(frame_files, str(tmp_path / "p.png"), tile_size=0)
    zstackr.stack_files(
        params, quiet, progress=lambda stage, current, total: seen.append((stage, current, total))
    )
    stages = []
    for stage, _, _ in seen:
        if not stages or stages[-1] != stage:
            stages.append(stage)
    assert stages == ["load", "align", "fuse", "save"]
    assert seen[-1] == ("save", 1000, 1000)


def test_stack_files_rejects_what_cannot_be_honoured(quiet, frame_files, tmp_path):
    with pytest.raises(ValueError, match="mode"):
        zstackr.stack_files(
            zstackr.PyPipelineParams(frame_files, str(tmp_path / "a.png"), mode="blend"), quiet
        )
    with pytest.raises(ValueError, match=r"\.tif"):
        zstackr.stack_files(zstackr.PyPipelineParams(frame_files, str(tmp_path / "a.bmp")), quiet)
    with pytest.raises(ValueError, match="64"):
        zstackr.stack_files(
            zstackr.PyPipelineParams(frame_files, str(tmp_path / "a.png"), tile_size=10), quiet
        )
    with pytest.raises(RuntimeError):
        zstackr.stack_files(
            zstackr.PyPipelineParams([str(tmp_path / "gone.png")], str(tmp_path / "a.png"), tile_size=0),
            quiet,
        )


def test_batch_stack_stacks_each_folder_and_survives_a_bad_one(quiet, tmp_path):
    root = tmp_path / "batches"
    for name in ("beetle", "ant"):
        folder = root / name
        folder.mkdir(parents=True)
        for k in range(2):
            write_png(folder / f"f{k}.png", half_sharp(k == 0))
    broken = root / "broken"
    broken.mkdir()
    (broken / "x.png").write_bytes(b"not a png")
    (root / "empty").mkdir()

    results = zstackr.batch_stack(str(root), str(tmp_path / "results"), quiet, tile_size=0)
    assert [name for name, _, _ in results] == ["ant", "beetle", "broken"]
    ok = {name: (good, detail) for name, good, detail in results}
    assert ok["ant"][0] and ok["beetle"][0] and not ok["broken"][0]
    assert (tmp_path / "results" / "ant_stacked.tif").exists()
