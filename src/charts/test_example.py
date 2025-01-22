import pytest


def add(a: int, b: int) -> int:
    return a + b


def test_adding() -> None:
    assert add(2, 2) == 4

    with pytest.raises(AssertionError):
        assert add(2, 3) == 4
