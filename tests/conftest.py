import sys
import os

import pytest

# Add src/ to the path so tests can import project modules directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class FakeFrameSource:
    """Deterministic camera substitute for stream lifecycle tests."""

    def __init__(self, frames=(), read_error=None):
        self._frames = iter(frames)
        self._read_error = read_error
        self.enter_count = 0
        self.read_count = 0
        self.release_count = 0

    def __enter__(self):
        self.enter_count += 1
        return self

    def read(self):
        self.read_count += 1
        if self._read_error is not None:
            raise self._read_error
        return next(self._frames, (False, None))

    def __exit__(self, exc_type, exc_value, traceback):
        self.release_count += 1
        return False


@pytest.fixture
def fake_frame_source_factory():
    sources = []

    def factory(*, frames=(), read_error=None):
        source = FakeFrameSource(frames=frames, read_error=read_error)
        sources.append(source)
        return source

    return factory, sources
