"""The known positions (positions/NWL23/known-positions): every one replays,
and HastyBot's move at each is the one its manifest entry names, so the
manifest's notation is the pick tool's and each record still reaches the
position it was cut for."""

from pathlib import Path

import pytest

_ENGINE_DIR = Path(__file__).resolve().parents[2] / "target" / "engine"
LEAVES = Path("/workspace/mount/macondo/data/strategy/NWL23/leaves.klv2")
if not (_ENGINE_DIR / "position_pick_tool").is_file():
    pytest.skip("engine not built -- run py/build.py first", allow_module_level=True)
if not LEAVES.exists():
    pytest.skip("HastyBot leave values not installed", allow_module_level=True)

from scribblez.deep_bot import known_positions  # noqa: E402
from scripts.deep_bot import known_position_picks  # noqa: E402


@pytest.fixture(scope="module")
def hasty_picks(tmp_path_factory):
    positions = known_positions.load()
    work = tmp_path_factory.mktemp("known")
    return positions, known_positions.ask(positions, "--type=hastybot", work, threads=2)


def test_every_position_replays_to_hastybots_recorded_move(hasty_picks):
    positions, picks = hasty_picks
    assert len(positions) == len(picks) >= 15
    assert len({p.id for p in positions}) == len(positions)
    for p, pick in zip(positions, picks, strict=True):
        assert pick["pick"] == p.static_equity_best, p.id
        assert p.read(pick["pick"]) == known_positions.EQUITY, p.id
        # What makes a position one: its answer is outside HastyBot's ten
        # best, and is not a pass.
        assert p.reference and p.reference_equity_rank > 10, p.id
        assert "-" not in p.reference, p.id


def test_a_pick_is_read_against_its_positions_answer(hasty_picks):
    positions, picks = hasty_picks
    by_id = {p.id: p for p in positions}
    assert by_id["EELLT-exch"].read("-EELLT") == known_positions.REFERENCE
    assert by_id["CLINKER"].read("O1 ZZZ") == known_positions.OTHER
    text = known_position_picks.table(positions, picks)
    assert text.splitlines()[-1] == (
        f"{len(positions)} positions: 0 reference, {len(positions)} equity, 0 other"
    )
