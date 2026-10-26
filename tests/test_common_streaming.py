from tools.check_common_streaming import check


def test_streaming_retains_restart_state_and_frame_history(tmp_path):
    assert check(tmp_path/'streaming')['status'] == 'passed'
