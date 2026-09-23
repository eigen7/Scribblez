"""The blind_spots workload: what a cycle keeps and delivers."""

import json
from pathlib import Path

from scribblez.sim_candidate_survey import SURVEY_SUFFIX, load_survey, slim_survey_file
from scribblez.workloads import blind_spots
from tests.test_sim_candidate_survey import position


class RecordingSink:
    """Records deliveries as (data-relative destination, file name) and consumes
    the file, as a real sink does."""

    kind = "local"

    def __init__(self):
        self.delivered = []

    def deliver(self, src: Path, data_rel: str) -> int:
        self.delivered.append(data_rel)
        size = src.stat().st_size
        src.unlink()
        return size


def write_game(work_dir: Path, stem: str, positions: list[dict]) -> Path:
    path = work_dir / f"{stem}{SURVEY_SUFFIX}"
    path.write_text(json.dumps({"version": 3, "cut": 10, "positions": positions}))
    (work_dir / f"{stem}.slog").write_bytes(b"game")
    (work_dir / blind_spots.GCG_DIR).mkdir(exist_ok=True)
    for p in positions:
        name = f"{stem}-g{p['game']}-turn{p['turn'] + 1}.gcg"
        (work_dir / blind_spots.GCG_DIR / name).write_text("#note a game\n")
    return path


def test_slimming_keeps_the_found_positions_and_their_confirmed_candidates(tmp_path):
    found = position([40], {62: 55}) | {"turn": 7}
    found["candidates"].append({"move": "M99", "display": "M99", "equity_rank": 99})  # screened out
    path = write_game(tmp_path, "g", [found, position([40], {62: 41}) | {"turn": 9}])
    assert slim_survey_file(path) == [(0, 7)]
    slim = json.loads(path.read_text())
    assert slim["positions_surveyed"] == 2
    (kept,) = slim["positions"]
    assert [c["move"] for c in kept["candidates"]] == ["M0", "M62"]
    # The slimmed file still reads as a survey: same finding, same position count.
    survey = load_survey([path])
    assert (survey.positions, len(survey.winners)) == (2, 1)
    assert survey.winners[0].outside.move == "M62"


def test_a_cycle_delivers_found_positions_and_clears_the_game(tmp_path):
    found, dull = position([40], {62: 55}) | {"turn": 7}, position([40], {62: 41}) | {"turn": 9}
    write_game(tmp_path, "123-w1", [found, dull])
    sink = RecordingSink()
    positions, nbytes, _ = blind_spots.deliver_surveyed(sink, tmp_path)
    assert positions == 1 and nbytes > 0
    # The kept position's game first, then the survey file; nothing for the dull one.
    assert sink.delivered == ["gcg/123-w1-g0-turn8.gcg", f"survey/123-w1{SURVEY_SUFFIX}"]
    assert list(tmp_path.iterdir()) == []  # the .slog and the unkept .gcg are cleared away


class StubSpec:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir

    def paths(self, tag: str):
        return self


def test_progress_counts_the_tags_store(tmp_path):
    (tmp_path / "gcg").mkdir()
    (tmp_path / "survey").mkdir()
    (tmp_path / "gcg" / "a.gcg").touch()
    (tmp_path / "survey" / f"a{SURVEY_SUFFIX}").touch()
    counts = blind_spots.progress(StubSpec(tmp_path), "t")
    assert counts == [("positions found", 1), ("games surveyed", 1)]


class Task:
    tag = "t"

    def __init__(self, target: int):
        self.params = {"target_positions": target}


def gate_after_tick(tmp_path, target: int, found: int):
    (tmp_path / "gcg").mkdir(exist_ok=True)
    for i in range(found):
        (tmp_path / "gcg" / f"{i}.gcg").touch()
    gates = []
    spec = StubSpec(tmp_path)
    spec.params_cls = blind_spots.BlindSpotsParams
    hooks = type(
        "Hooks", (), {"gate": staticmethod(lambda role, reason: gates.append((role, reason)))}
    )
    blind_spots.tick(spec, Task(target), hooks)
    return gates


def test_the_scheduler_parks_the_surveyors_at_the_target(tmp_path):
    assert gate_after_tick(tmp_path, target=3, found=2) == [("generate", None)]
    assert gate_after_tick(tmp_path, target=3, found=3) == [
        ("generate", "target reached: 3 of 3 positions")
    ]
    assert gate_after_tick(tmp_path, target=0, found=3) == [("generate", None)]  # 0: never


def test_the_survey_seed_is_fixed_by_the_game_so_a_restart_can_resume(tmp_path):
    (tmp_path / "123-w1.slog").write_bytes(b"game")
    seed = blind_spots.survey_seed(tmp_path)
    assert blind_spots.survey_seed(tmp_path) == seed and 0 <= seed < 2**62
    (tmp_path / "123-w1.slog").rename(tmp_path / "456-w1.slog")
    assert blind_spots.survey_seed(tmp_path) != seed
