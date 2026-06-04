"""
Request ID generation utility.

Generates unique identifiers for each incoming API request,
used for log correlation and response tracing.
"""

import uuid


def generate_request_id() -> str:
    """Generate a unique request identifier.

    Returns:
        A string in the format "req_<uuid4>" where uuid4 is a random
        UUID without hyphens. Example: "req_a1b2c3d4e5f6..."
    """
    return f"req_{uuid.uuid4().hex}"
