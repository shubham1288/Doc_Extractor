"""
Shared test configuration and fixtures for the KYC Document Extraction system.

Provides:
- Hypothesis settings profiles (default and CI)
- FastAPI test client fixture
- Sample image fixtures for preprocessing tests
- Test file bytes fixtures (JPEG and PNG)
"""

import io
import struct

import numpy as np
import pytest
from httpx import ASGITransport, AsyncClient

from hypothesis import HealthCheck, Phase, settings

from app.main import app


# ---------------------------------------------------------------------------
# Hypothesis Settings Profiles
# ---------------------------------------------------------------------------

# Default profile: balanced speed vs coverage for local development
settings.register_profile(
    "default",
    max_examples=100,
    deadline=500,  # milliseconds
    suppress_health_check=[HealthCheck.too_slow],
    phases=[Phase.explicit, Phase.reuse, Phase.generate, Phase.shrink],
)

# CI profile: thorough testing with no deadline pressure
settings.register_profile(
    "ci",
    max_examples=500,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
    phases=[Phase.explicit, Phase.reuse, Phase.generate, Phase.shrink],
)

# Load the default profile (CI systems can override via --hypothesis-profile=ci)
settings.load_profile("default")


# ---------------------------------------------------------------------------
# FastAPI Test Client Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def async_client() -> AsyncClient:
    """Async HTTP client for testing FastAPI endpoints."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


# ---------------------------------------------------------------------------
# Sample Image Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_grayscale_image() -> np.ndarray:
    """A small 100x100 grayscale image (uint8) for preprocessing tests."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, size=(100, 100), dtype=np.uint8)


@pytest.fixture
def sample_color_image() -> np.ndarray:
    """A small 100x100 BGR color image (uint8) for preprocessing tests."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, size=(100, 100, 3), dtype=np.uint8)


@pytest.fixture
def sample_document_image() -> np.ndarray:
    """A 300x200 grayscale image simulating a document scan with some text-like contrast."""
    image = np.full((300, 200), 240, dtype=np.uint8)  # Light background
    # Add some dark regions to simulate text
    image[50:60, 20:180] = 30
    image[80:90, 20:150] = 30
    image[110:120, 20:170] = 30
    return image


# ---------------------------------------------------------------------------
# Test File Bytes Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def valid_jpeg_bytes() -> bytes:
    """Minimal valid JPEG file bytes for upload testing."""
    # Minimal JPEG: SOI + APP0 + minimal scan data + EOI
    # This is a 1x1 white pixel JPEG
    return (
        b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
        b"\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t"
        b"\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a"
        b"\x1f\x1e\x1d\x1a\x1c\x1c $.\' ',#\x1c\x1c(7),01444\x1f\'9=82<.342"
        b"\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00"
        b"\xff\xc4\x00\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00\x00"
        b"\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b"
        b"\xff\xc4\x00\xb5\x10\x00\x02\x01\x03\x03\x02\x04\x03\x05\x05\x04"
        b"\x04\x00\x00\x01}\x01\x02\x03\x00\x04\x11\x05\x12!1A\x06\x13Qa\x07"
        b"\x22q\x142\x81\x91\xa1\x08#B\xb1\xc1\x15R\xd1\xf0$3br\x82\t\n\x16"
        b"\x17\x18\x19\x1a%&\'()*456789:CDEFGHIJSTUVWXYZcdefghijstuvwxyz"
        b"\x83\x84\x85\x86\x87\x88\x89\x8a\x92\x93\x94\x95\x96\x97\x98\x99"
        b"\x9a\xa2\xa3\xa4\xa5\xa6\xa7\xa8\xa9\xaa\xb2\xb3\xb4\xb5\xb6\xb7"
        b"\xb8\xb9\xba\xc2\xc3\xc4\xc5\xc6\xc7\xc8\xc9\xca\xd2\xd3\xd4\xd5"
        b"\xd6\xd7\xd8\xd9\xda\xe1\xe2\xe3\xe4\xe5\xe6\xe7\xe8\xe9\xea\xf1"
        b"\xf2\xf3\xf4\xf5\xf6\xf7\xf8\xf9\xfa"
        b"\xff\xda\x00\x08\x01\x01\x00\x00?\x00T\xdb\x9e\xa7\x8bB\xb2\x03"
        b"\xff\xd9"
    )


@pytest.fixture
def valid_png_bytes() -> bytes:
    """Minimal valid PNG file bytes (1x1 white pixel) for upload testing."""
    # PNG file structure: signature + IHDR + IDAT + IEND
    # This creates a minimal 1x1 white pixel PNG
    buf = io.BytesIO()

    # PNG signature
    buf.write(b"\x89PNG\r\n\x1a\n")

    def write_chunk(chunk_type: bytes, data: bytes) -> None:
        buf.write(struct.pack(">I", len(data)))
        buf.write(chunk_type)
        buf.write(data)
        import zlib
        crc = zlib.crc32(chunk_type + data) & 0xFFFFFFFF
        buf.write(struct.pack(">I", crc))

    # IHDR: width=1, height=1, bit_depth=8, color_type=2 (RGB)
    ihdr_data = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    write_chunk(b"IHDR", ihdr_data)

    # IDAT: compressed image data (filter byte 0 + RGB white pixel)
    import zlib
    raw_data = b"\x00\xff\xff\xff"  # filter=None, R=255, G=255, B=255
    compressed = zlib.compress(raw_data)
    write_chunk(b"IDAT", compressed)

    # IEND
    write_chunk(b"IEND", b"")

    return buf.getvalue()


@pytest.fixture
def invalid_file_bytes() -> bytes:
    """Random bytes that don't match any valid file format."""
    return b"\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09" * 100
