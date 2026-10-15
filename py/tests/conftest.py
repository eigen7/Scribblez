"""Suite-wide pytest setup: one CPU thread per math library and for
torch.compile, and per-file progress lines for the default (non -v/-q) output.

Tests are often run beside other CPU work. Torch's CPU ops default to one
thread per core, and on the tiny tensors tests use those threads spend their
time synchronizing; when other processes hold most cores, that turns a 1 s test
into a 90 s one (test_rack_best's overfit loop, measured with 24 of 28 cores
busy). One thread is faster even on an idle machine. Likewise torch.compile
starts a pool of one compile process per core (29 here) in every process that
compiles, for no measurable gain on test-sized graphs. The caps are set through
the environment before anything imports torch or numpy, so subprocesses
spawned by tests inherit them too.

For the progress lines: stock pytest right-aligns each file's percentage to
the terminal edge, which leaves a wide gap on a wide terminal, and reports no
time. This replaces the terminal reporter with one that prints each test file
as

    [ 69%] [   7.241s] py/tests/test_transfer_test_evaluate.py ......

where the time is the file's wall clock, module fixtures included. On a
terminal the line is drawn live with a blank prefix that is overwritten in
place once the file finishes; otherwise (piped output, or -s, whose prints
would interleave) the line is written once, complete. -v and -q keep the
stock output.
"""

import os
import sys
import time

import pytest
from _pytest.config import Config
from _pytest.pathlib import bestrelpath
from _pytest.reports import TestReport
from _pytest.terminal import TerminalReporter, TestShortLogReport

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"

PREFIX_WIDTH = len("[100%] [9999.999s] ")


@pytest.hookimpl(trylast=True)
def pytest_configure(config):
    # trylast: the stock reporter is registered by _pytest.terminal's own
    # pytest_configure, which would otherwise run after this one.
    if config.option.verbose != 0:
        return
    stock = config.pluginmanager.getplugin("terminalreporter")
    config.pluginmanager.unregister(stock)
    config.pluginmanager.register(FileTimingReporter(config), "terminalreporter")


def _letter_markup(report: TestReport) -> dict[str, bool]:
    if report.failed:
        return {"red": True}
    if report.skipped or hasattr(report, "wasxfail"):
        return {"yellow": True}
    return {"green": True}


class FileTimingReporter(TerminalReporter):
    def __init__(self, config: Config):
        super().__init__(config, sys.stdout)
        self._live = self.isatty() and config.getoption("capture") != "no"
        self._file_start = 0.0
        self._letters: list[tuple[str, dict[str, bool]]] = []

    def pytest_runtest_logstart(self, nodeid, location):
        fspath = self.config.rootpath / nodeid.split("::", maxsplit=1)[0]
        if fspath == self.currentfspath:
            return
        self._finish_file_line()
        self.currentfspath = fspath
        self._file_start = time.perf_counter()
        self._letters = []
        if self._live:
            self._tw.line()
            self._tw.write(" " * PREFIX_WIDTH + self._relpath() + " ", flush=True)

    def pytest_runtest_logreport(self, report):
        self._tests_ran = True
        status = TestShortLogReport(
            *self.config.hook.pytest_report_teststatus(report=report, config=self.config)
        )
        self._add_stats(status.category, [report])
        if not status.letter and not status.word:
            return  # Passed setup/teardown.
        self._progress_nodeids_reported.add(report.nodeid)
        letter = (status.letter, _letter_markup(report))
        self._letters.append(letter)
        if self._live and self._tw.width_of_current_line < self._tw.fullwidth - 1:
            self._tw.write(letter[0], flush=True, **letter[1])

    @pytest.hookimpl(wrapper=True)
    def pytest_runtestloop(self):
        result = yield
        self._finish_file_line()
        return result

    def _finish_file_line(self):
        if not self._letters:
            return
        color, _ = self._get_main_color()
        if self._live:
            self._tw.write("\r")
        else:
            self._tw.line()
        self._tw.write(self._prefix(), **{color: True})
        if not self._live:
            self._tw.write(self._relpath() + " ")
            for letter, markup in self._letters:
                self._tw.write(letter, **markup)
        self._tw.flush()
        self._letters = []

    def _prefix(self) -> str:
        assert self._session
        percent = self.reported_progress * 100 // max(self._session.testscollected, 1)
        elapsed = time.perf_counter() - self._file_start
        return f"[{percent:3d}%] [{elapsed:8.3f}s] "

    def _relpath(self) -> str:
        return bestrelpath(self.startpath, self.currentfspath)
