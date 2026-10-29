import numpy as np

from examples.blast_sandwich_design import Settings
from examples.blast_sandwich_reduced import checked


def test_fixed_mass_feasibility_rejects_underweight_response_feasible_field():
    record=dict(cases=np.zeros((3,7)),mass_kg=.9,
        actual_peak_back_m=.001,actual_peak_skin_Pa=100e6)
    assert checked(record,Settings(stage='min-mass'),1.)==(True,True)
    assert checked(record,Settings(stage='fixed-mass'),1.)==(True,False)
    record['mass_kg']=1.
    assert checked(record,Settings(stage='fixed-mass'),1.)==(True,True)
