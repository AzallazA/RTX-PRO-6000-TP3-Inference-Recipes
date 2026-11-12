# CPU checks for the compose.tools.yaml overlay (see ../check_tools.sh).
# DS41_TOOLS_EXPECT=unpatched: tests marked `fixed_by_port` become strict xfail, i.e. the run passes only if
# every one of them FAILS on the stock image (and every other test still passes).
import os

import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "fixed_by_port(pr): expectation added by an upstream PR this overlay ports"
    )


def pytest_collection_modifyitems(config, items):
    if os.environ.get("DS41_TOOLS_EXPECT", "patched") != "unpatched":
        return
    for item in items:
        mark = item.get_closest_marker("fixed_by_port")
        if mark is not None:
            item.add_marker(
                pytest.mark.xfail(
                    strict=True, reason=f"stock image lacks upstream #{mark.args[0]}"
                )
            )
