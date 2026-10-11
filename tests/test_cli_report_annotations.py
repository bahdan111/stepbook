from typing import Any, get_type_hints

from stepbook.cli.inspection import write_report_md


def test_report_annotations_resolve_for_env_and_result():
    annotations = get_type_hints(write_report_md)

    assert annotations["env"] is Any
    assert annotations["result"] is Any
