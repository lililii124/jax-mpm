"""Two closed-contact histories with an exact unit-mass normal projection."""
import json
import numpy as np

from tools.metal_affine_dynamic_numpy import midpoint_contact_velocity


class NormalContact:
    def apply(self, velocity):
        impulse = max(float(velocity[0]-velocity[1]), 0.)/2
        physical = np.array([-impulse, impulse])
        return velocity+physical, {}, np.zeros(2), physical


def check():
    contact = NormalContact()
    incoming = np.array([3., -1.])
    initial = contact.apply(incoming)[0]
    endpoint, _, _, physical = midpoint_contact_velocity(initial, np.zeros(2), 1., contact)
    final = contact.apply(endpoint)[0]
    np.testing.assert_allclose(final, [1., 1.], rtol=0, atol=1e-15)
    assert final[0]-final[1] == 0 and np.sum(final) == np.sum(incoming)
    assert .5*(incoming@incoming-final@final) == 4.
    separating = np.array([-.5, .5])
    force = np.array([2., -2.])
    endpoint, _, _, physical = midpoint_contact_velocity(separating, force, 1., contact)
    midpoint = .5*(separating+endpoint)
    final, _, _, final_impulse = contact.apply(endpoint)
    np.testing.assert_array_equal(midpoint, np.zeros(2))
    np.testing.assert_array_equal(final, np.zeros(2))
    assert midpoint@physical == 0 and midpoint@force == 0
    final_loss = -.5*(endpoint+final)@final_impulse
    assert final_loss == .25
    assert .5*(final@final-separating@separating)+final_loss == 0
    wrong_end, _, _, wrong_impulse = contact.apply(separating+force)
    old_positive_work = .5*(separating+wrong_end)@wrong_impulse
    assert old_positive_work == .75
    return {'passed': True, 'closing_impact_restitution': 0.,
            'reclosing_midpoint_contact_work': 0., 'final_inelastic_loss': final_loss,
            'endpoint_only_positive_work': old_positive_work,
            'scope': 'Two exact closed-contact histories; no general collision-geometry validation'}


if __name__ == '__main__':
    print(json.dumps(check()), flush=True)
