import io

import pytest
import imageio.v2 as iio
import imageio.v3 as iio3
import numpy as np
from PIL import Image


def test_reader(test_images):
    with iio.get_reader(test_images / "chelsea.png") as file:
        assert file.request.extension == ".png"

        img = file.get_data(0)
        assert img.shape == (300, 451, 3)

        # read using alternative method
        file.set_image_index(0)
        img2 = file.get_next_data()

        assert np.allclose(img2, img)


def test_get_reader_format(test_images):
    with iio.get_reader(test_images / "chelsea.png") as file:
        with pytest.raises(TypeError):
            file.format  # v3 Pillow Plugin


def test_get_length(test_images):
    with iio.get_reader(test_images / "chelsea.png") as file:
        v3_count = file.instance.properties(index=...).shape[0]
        assert len(file) == v3_count


def test_iterating(test_images):
    with iio.get_reader(test_images / "newtonscradle.gif") as file:
        for count, img in enumerate(file):
            assert img.shape == (150, 200, 3)

        assert count == len(file) - 1


def test_writer(test_images, tmp_path):
    img = iio.imread(test_images / "chelsea.png")

    with iio.get_writer(tmp_path / "test.png") as file:
        assert file.request.extension == ".png"
        file.append_data(img)


def test_get_writer_format(tmp_path):
    with iio.get_writer(tmp_path / "test.png") as file:
        with pytest.raises(TypeError):
            file.format  # v3 Pillow Plugin


def test_warnings(tmp_path, test_images):
    img = iio.imread(test_images / "chelsea.png")

    with iio.get_writer(tmp_path / "test.png") as writer:
        with pytest.warns(UserWarning):
            writer.append_data(img, meta={"meta_key": "meta_value"})


def test_mimwrite_exception(tmp_path):
    img = np.zeros((4, 4), dtype=np.uint8)

    with pytest.raises(ValueError):
        iio.mimwrite(tmp_path / "test.png", img)


def test_kwarg_forwarding(tmp_path):
    rng = np.random.default_rng()

    # Write the frames to an animated GIF
    with iio.get_writer(
        tmp_path / "tmp.gif", duration=1000, loop=0, pilmode="RGB"
    ) as writer:
        for _ in range(10):
            frame = rng.integers(0, 255, (100, 100, 3), dtype=np.uint8)
            writer.append_data(frame)

    imgs = np.stack(iio.mimread(tmp_path / "tmp.gif"))
    assert imgs.shape == (10, 100, 100, 3)

    with iio.get_reader(tmp_path / "tmp.gif", pilmode="L") as file:
        img = file.get_next_data()

        assert img.shape == (100, 100)
        assert img.meta["duration"] == 1000
        assert img.meta["loop"] == 0


def _make_frames(n=4, shape=(32, 48, 3)):
    rng = np.random.default_rng(42)
    return [rng.integers(0, 255, shape, dtype=np.uint8) for _ in range(n)]


@pytest.mark.parametrize("writer", ["mimwrite", "get_writer"])
def test_gif_default_loops_forever(tmp_path, writer):
    """Regression test: the default v2 GIF export must loop indefinitely.

    The v2 API documents ``loop`` as defaulting to 0 (loop forever), and the
    default plugin used for ``.gif`` writes (the v3 ``pillow`` plugin) only
    emits a NETSCAPE loop extension when it is given a ``loop`` value. Without
    the v2 default the written animations play exactly once and stop on their
    last frame.
    """
    frames = _make_frames()
    path = tmp_path / "animated.gif"

    if writer == "mimwrite":
        iio.mimwrite(path, frames)
    else:
        with iio.get_writer(path) as file:
            for frame in frames:
                file.append_data(frame)

    data = path.read_bytes()
    assert b"NETSCAPE2.0" in data

    with Image.open(io.BytesIO(data)) as im:
        assert im.info["loop"] == 0
        assert im.n_frames == len(frames)
        # the loop extension has to be read back by imageio as well
        frames_back = iio.mimread(path)
        assert len(frames_back) == len(frames)


def test_gif_explicit_loop_is_respected(tmp_path):
    """An explicitly requested loop count must not be overridden."""
    frames = _make_frames(n=3)
    path = tmp_path / "explicit.gif"
    iio.mimwrite(path, frames, loop=2)

    with Image.open(path) as im:
        assert im.info["loop"] == 2


def test_v3_gif_write_still_defaults_to_no_loop(tmp_path):
    """The fix must not leak into the imageio.v3 write behavior."""
    frames = np.stack(_make_frames(n=3))
    data = iio3.imwrite("<bytes>", frames, extension=".gif")

    assert b"NETSCAPE2.0" not in data
    with Image.open(io.BytesIO(data)) as im:
        assert im.info.get("loop") is None
