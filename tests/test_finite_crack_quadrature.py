from tools.audit_finite_crack import audit


def test_finite_crack_endpoint_tie_removes_shadow_residual():
    for angle, offset in ((0., 0.), (.31, .25)):
        result = audit(12, angle, offset)
        assert result['legacy']['relative_interior_residual'] > 1e-4
        assert result['tied']['relative_interior_residual'] < 2e-11
        assert result['legacy']['pair_resultant_error'] < 2e-11
        assert result['tied']['pair_resultant_error'] < 2e-11
        assert result['legacy']['face_resultant_error'] < 2e-11
        assert result['tied']['face_resultant_error'] < 2e-11
        assert result['legacy']['face_moment_error'] < 2e-11
        assert result['tied']['face_moment_error'] < 2e-11
