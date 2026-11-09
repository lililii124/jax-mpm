import pytest

from tools.compare_multicell_binary_blast import verify_physical_model_compatibility


def settings():
    return {'cell': [8, 8, 20], 'cell_size': [.1, .1, .02], 'skin': 0., 'skin_layers': 0,
            'volume_fraction': .3, 'skin_young': 70e9, 'core_young': 70e9, 'solid_density': 2700.,
            'pressure': 2e6, 'pulse': 2e-5, 'blast_radius': .08, 'dt': 1e-7, 'steps': 1000}


def test_same_horizon_allows_different_timesteps():
    proxy, binary = settings(), settings()
    binary.update(dt=5e-8, steps=2000, fixed_edge_width=[.012625, .012625])
    verify_physical_model_compatibility(proxy, binary)


def test_horizon_and_fixed_width_mismatches_are_rejected():
    proxy, binary = settings(), settings()
    binary['steps'] = 900
    with pytest.raises(ValueError, match='time horizons'):
        verify_physical_model_compatibility(proxy, binary)
    binary = settings(); binary['fixed_edge_width'] = [.006, .012625]
    with pytest.raises(ValueError, match='fixed-edge widths'):
        verify_physical_model_compatibility(proxy, binary)
