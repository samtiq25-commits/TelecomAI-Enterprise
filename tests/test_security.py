from src.security import can_access
def test_roles():
    assert can_access("admin","voice")
    assert can_access("viewer","dashboard")
    assert not can_access("viewer","voice")
