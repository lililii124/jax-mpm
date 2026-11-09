from tools.audit_cut_interface import audit


def test_knot_split_cut_quadrature_balances_displaced_and_inclined_cracks():
    for angle in (0., .31, .68):
        for offset in (-.375, 0., .25):
            result = audit(12, offset, angle)
            assert result['plus_relative_interior_residual'] < 2e-10
            assert result['minus_relative_interior_residual'] < 2e-10
            assert result['face_partition_error'] < 2e-11
            assert result['face_first_moment_error'] < 2e-11
            assert result['bulk_partition_error'] < 2e-11
            assert result['pair_resultant_error'] < 2e-10
            assert result['pair_moment_error'] < 2e-10
            assert result['face_resultant_error'] < 2e-10
            assert result['face_moment_error'] < 2e-10
