import pytest
from cardiovortex.demo import main

def test_demo_execution(capsys):
    main()
    captured = capsys.readouterr()
    assert "CARDIOVORTEX" in captured.out
    assert "Fibrillation Terminated" in captured.out
