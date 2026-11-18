import numpy as np
import pytest
from cardiovortex.models.aliev_panfilov import AlievPanfilovModel, APParameters
from cardiovortex.models.fitzhugh_nagumo import FitzHughNagumoModel, FHNParameters

def test_aliev_panfilov_resting():
    model = AlievPanfilovModel()
    u, v = model.initial_state((10, 10))
    du, dv = model.reaction_rates(u, v)
    # At resting (u=0, v=0), du/dt = 0, dv/dt = 0
    assert np.allclose(du, 0.0, atol=1e-7)
    assert np.allclose(dv, 0.0, atol=1e-7)

def test_aliev_panfilov_excitation():
    model = AlievPanfilovModel()
    # Depolarized state above threshold a=0.15
    u = np.full((5, 5), 0.5)
    v = np.zeros((5, 5))
    du, dv = model.reaction_rates(u, v, i_ext=0.1)
    # Should produce strong positive upstroke du/dt > 0
    assert np.all(du > 0)

def test_fitzhugh_nagumo_dynamics():
    model = FitzHughNagumoModel()
    u, v = model.initial_state((4, 4))
    du, dv = model.reaction_rates(u, v, i_ext=0.5)
    assert du.shape == (4, 4)
    assert dv.shape == (4, 4)
    assert np.all(np.isfinite(du))
    assert np.all(np.isfinite(dv))
