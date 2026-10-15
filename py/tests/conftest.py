"""Suite-wide pytest setup: one CPU thread per math library and for
torch.compile, seeded RNGs per test, and per-file progress lines for the
default (non -v/-q) output.

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

where the time is the file's wall clock, module fixtures included. A file's
line is due when its last collected test finishes, and the percentage counts
tests finished across the whole run. Under xdist the workers only forward
their test events to the controller, whose reporter is this one: files finish
out of order there, so each line is written once, complete, as its file
finishes. In a serial run on a terminal the line is instead drawn live with a
blank prefix that is overwritten in place once the file finishes (but not
under -s, whose prints would interleave). -v and -q keep the stock output.
"""

import os

# Before numpy or torch load (both below), which read them once.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"

import random  # noqa: E402
import time  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402
from _pytest.pathlib import bestrelpath  # noqa: E402
from _pytest.reports import TestReport  # noqa: E402
from _pytest.terminal import TerminalReporter, TestShortLogReport  # noqa: E402

PREFIX_WIDTH = len("[100%] [9999.999s] ")


@pytest.fixture(autouse=True)
def _seed_rngs():
    """Every test starts from the same RNG state, whatever ran before it in
    its process: under xdist that set of tests changes from run to run."""
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)


@pytest.hookimpl(trylast=True)
def pytest_configure(config):
    # torch.utils.mkldnn, which torch imports itself, decorates 14 methods with
    # torch.jit.script_method, which torch now deprecates: 14 warnings per
    # process that no code of ours can address.
    config.addinivalue_line(
        "filterwarnings", "ignore:`torch.jit.script_method` is deprecated:DeprecationWarning"
    )
    # trylast: the stock reporter is registered by _pytest.terminal's own
    # pytest_configure, which would otherwise run after this one. The reporter
    # object itself is kept, its class swapped, because other plugins hold it
    # by then (xdist's controller looks it up in its own pytest_configure);
    # re-registering rebinds its hooks to the new class.
    if config.option.verbose != 0 or hasattr(config, "workerinput"):
        return
    reporter = config.pluginmanager.getplugin("terminalreporter")
    config.pluginmanager.unregister(reporter)
    reporter.__class__ = FileTimingReporter
    reporter._files = {}
    config.pluginmanager.register(reporter, "terminalreporter")


def _file_of(nodeid: str) -> str:
    return nodeid.split("::", maxsplit=1)[0]


def _letter_markup(report: TestReport) -> dict[str, bool]:
    if report.failed:
        return {"red": True}
    if report.skipped or hasattr(report, "wasxfail"):
        return {"yellow": True}
    return {"green": True}


@dataclass
class _FileProgress:
    """One test file's progress, as the controller hears of it."""

    expected: int  # tests collected in the file
    start: float = 0.0  # perf_counter at its first test's start; 0 before
    finished: int = 0
    letters: list[tuple[str, dict[str, bool]]] = field(default_factory=list)


class FileTimingReporter(TerminalReporter):
    """Never constructed: pytest_configure swaps it in as the stock
    reporter's class."""

    _files: dict[str, _FileProgress]

    def pytest_collection_finish(self, session):
        super().pytest_collection_finish(session)
        self._expect([item.nodeid for item in session.items])

    @pytest.hookimpl(optionalhook=True)
    def pytest_xdist_node_collection_finished(self, node, ids):
        if not self._files:  # every worker collects the same ids
            self._expect(ids)

    def pytest_runtest_logstart(self, nodeid, location):
        name = _file_of(nodeid)
        progress = self._files[name]
        if progress.start:
            return
        progress.start = time.perf_counter()
        if self._live:
            self._start_line(name)
            self._tw.write(" " * PREFIX_WIDTH + self._relpath(name) + " ", flush=True)

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
        self._files[_file_of(report.nodeid)].letters.append(letter)
        if self._live and self._tw.width_of_current_line < self._tw.fullwidth - 1:
            self._tw.write(letter[0], flush=True, **letter[1])

    def pytest_runtest_logfinish(self, nodeid, location):
        name = _file_of(nodeid)
        progress = self._files[name]
        progress.finished += 1
        if progress.finished == progress.expected:
            self._finish_line(name)

    @pytest.hookimpl(wrapper=True)
    def pytest_runtestloop(self):
        try:
            return (yield)
        finally:
            # Files cut short (-x, a crashed worker) never reach their count;
            # xdist ends a -x run by raising out of the loop.
            for name, progress in self._files.items():
                if progress.letters:
                    self._finish_line(name)

    @property
    def _xdist(self) -> bool:
        return self.config.pluginmanager.has_plugin("dsession")

    @property
    def _live(self) -> bool:
        return self.isatty() and self.config.getoption("capture") != "no" and not self._xdist

    def _expect(self, nodeids: list[str]):
        for nodeid in nodeids:
            name = _file_of(nodeid)
            self._files.setdefault(name, _FileProgress(expected=0)).expected += 1

    def _start_line(self, name: str):
        # The stock serial reporter opens each file's line with a newline; under
        # xdist the controller's status output already ends with a blank line.
        if self._tw.width_of_current_line or not self._xdist:
            self._tw.line()
        self.currentfspath = self.config.rootpath / name  # for ensure_newline

    def _finish_line(self, name: str):
        progress = self._files[name]
        color, _ = self._get_main_color()
        if self._live:
            self._tw.write("\r")
        else:
            self._start_line(name)
        self._tw.write(self._prefix(progress), **{color: True})
        if not self._live:
            self._tw.write(self._relpath(name) + " ")
            for letter, markup in progress.letters:
                self._tw.write(letter, **markup)
        self._tw.flush()
        progress.letters = []

    def _prefix(self, progress: _FileProgress) -> str:
        assert self._session
        percent = self.reported_progress * 100 // max(self._session.testscollected, 1)
        elapsed = time.perf_counter() - progress.start
        return f"[{percent:3d}%] [{elapsed:8.3f}s] "

    def _relpath(self, name: str) -> str:
        return bestrelpath(self.startpath, self.config.rootpath / name)
