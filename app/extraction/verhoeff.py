"""Verhoeff checksum algorithm for Aadhaar number validation.

The Verhoeff algorithm is a checksum formula used to validate 12-digit
Aadhaar numbers issued by UIDAI. It detects all single-digit errors and
all adjacent transposition errors.
"""

# Verhoeff multiplication table (Dihedral group D5)
_d = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
    [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
    [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
    [4, 0, 1, 2, 3, 9, 5, 6, 7, 8],
    [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
    [6, 5, 9, 8, 7, 1, 0, 4, 3, 2],
    [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
    [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
    [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
]

# Verhoeff permutation table
_p = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
    [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
    [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
    [9, 4, 5, 3, 1, 2, 6, 8, 7, 0],
    [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
    [2, 7, 9, 3, 8, 0, 6, 4, 1, 5],
    [7, 0, 4, 6, 9, 1, 3, 2, 5, 8],
]

# Verhoeff inverse table
_inv = [0, 4, 3, 2, 1, 5, 6, 7, 8, 9]


def validate_verhoeff(number: str) -> bool:
    """Validate a 12-digit Aadhaar number using the Verhoeff checksum algorithm.

    The algorithm processes digits from right to left, applying permutation
    and multiplication tables. A valid number produces a final checksum of 0.

    Args:
        number: A string of exactly 12 digits to validate.

    Returns:
        True if the Verhoeff checksum is valid (final value == 0), False otherwise.

    Raises:
        ValueError: If the input is not exactly 12 digits.
    """
    if not number.isdigit() or len(number) != 12:
        raise ValueError(f"Expected 12-digit string, got: '{number}'")

    checksum = 0
    reversed_digits = [int(x) for x in reversed(number)]

    for i, digit in enumerate(reversed_digits):
        # INVARIANT: checksum == accumulated Verhoeff value for digits[0..i-1]
        checksum = _d[checksum][_p[i % 8][digit]]

    return checksum == 0
