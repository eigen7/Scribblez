"""Tests for the evidence fusion stage (roadmap item 2): the exactness
guarantees the deployment loop rests on -- empty evidence degrades to the
plain one-pass model bit-for-bit, padding is inert, and the staged
(cached-per-round) path reproduces the monolithic forward -- plus the .sobs ->
EvidenceInputs builder (FFI-backed, so it needs the engine built)."""

import dataclasses

import numpy as np
import pytest
import torch
from scribblez.evidence_fusion import (
    NUM_EVIDENCE_PLANES,
    NUM_EVIDENCE_SCALARS,
    EvidenceInputs,
)
from scribblez.move_set_eval import moves as move_enc
from test_move_set_eval_train import _FORWARD_KEYS, _MOVE_KEYS, _ragged_batch


def _model(batch, seed: int = 0):
    from scribblez.move_set_eval.model import MoveSetEvalModel

    torch.manual_seed(seed)
    model = MoveSetEvalModel(
        spatial_planes=batch["input_spatial"].shape[1],
        scalar_size=batch["input_scalar"].shape[1],
        trunk_channels=8,
        num_blocks=2,
        num_heads=2,
    )
    return model.double().eval()


def _randomize_fusion(model, seed: int = 1):
    """The fusion projections are zero-initialized, so its effect on outputs
    is invisible until the weights are perturbed the way training would."""
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in model.evidence_fusion.parameters():
            p.copy_(torch.randn(p.shape, generator=gen, dtype=p.dtype) * 0.1)


def _random_evidence(p: int, counts: list[int], seed: int = 2, max_e: int | None = None):
    """A padded (P, E, ...) evidence batch with `counts[i]` real tokens for
    position i. Padded rows hold random (valid-range) garbage, which the mask
    is supposed to make unreachable."""
    max_tiles, num_scalars, letter_vocab, cells = move_enc.move_encoding_dims()
    e = max_e if max_e is not None else max(max(counts), 1)
    gen = torch.Generator().manual_seed(seed)
    mask = torch.zeros(p, e, dtype=torch.bool)
    for i, k in enumerate(counts):
        mask[i, :k] = True
    return EvidenceInputs(
        letters=torch.randint(0, letter_vocab, (p, e, max_tiles), generator=gen),
        blanks=torch.randint(0, 2, (p, e, max_tiles), generator=gen),
        squares=torch.randint(0, cells, (p, e, max_tiles), generator=gen),
        tile_mask=(torch.rand(p, e, max_tiles, generator=gen) > 0.4).double(),
        scalars=torch.randn(p, e, num_scalars, generator=gen, dtype=torch.float64),
        obs_planes=torch.rand(
            p, e, NUM_EVIDENCE_PLANES, 15, 15, generator=gen, dtype=torch.float64
        ),
        obs_scalars=torch.randn(p, e, NUM_EVIDENCE_SCALARS, generator=gen, dtype=torch.float64),
        mask=mask,
    )


def _forward(model, batch, evidence=None):
    with torch.no_grad():
        return model(*(batch[k] for k in _FORWARD_KEYS), evidence=evidence)


def _assert_equal(a: dict, b: dict, moves=slice(None), atol=0.0):
    for name in a:
        torch.testing.assert_close(a[name][moves], b[name][moves], rtol=0, atol=atol)


def test_zero_initialized_fusion_is_the_plain_model():
    """A fresh model with a full evidence set computes exactly the plain
    one-pass model: every fusion output projection starts at zero."""
    counts = [2, 3]
    batch = _ragged_batch([4, 6])
    model = _model(batch)
    _assert_equal(_forward(model, batch), _forward(model, batch, _random_evidence(2, counts)))


def test_empty_evidence_passes_through_at_any_weights():
    """An all-empty evidence set is bit-identical to passing no evidence,
    whatever the (trained) fusion weights -- the hard per-position gate, not
    the zero initialization, is what carries this guarantee."""
    batch = _ragged_batch([4, 6, 1])
    model = _model(batch)
    _randomize_fusion(model)
    plain = _forward(model, batch)
    empty = _random_evidence(3, [0, 0, 0], max_e=2)
    assert not empty.mask.any()
    _assert_equal(plain, _forward(model, batch, empty))


def test_evidence_free_positions_are_untouched_in_a_mixed_batch():
    """Training batches mix prefix sizes including zero; a position with no
    evidence must score exactly as it does in an evidence-free batch even
    when its neighbors carry evidence."""
    counts = [3, 5, 2]
    evidence_counts = [2, 0, 1]
    batch = _ragged_batch(counts)
    model = _model(batch)
    _randomize_fusion(model)
    plain = _forward(model, batch)
    mixed = _forward(model, batch, _random_evidence(3, evidence_counts))
    bare_position = slice(counts[0], counts[0] + counts[1])  # position 1's candidate rows
    _assert_equal(plain, mixed, moves=bare_position)
    # And the conditioning is not vacuous: evidenced positions did change.
    assert not torch.equal(plain["wld"][: counts[0]], mixed["wld"][: counts[0]])


def test_padded_evidence_rows_are_inert():
    """The same evidence set padded wider -- pad rows full of garbage -- must
    not change the outputs. Tolerance is the kernel-shape epsilon, not zero:
    a wider padded batch routes the same rows through differently-blocked
    kernels (the same effect the cross-position independence test absorbs);
    content leaking from pad rows would show at full magnitude."""
    counts = [2, 1]
    batch = _ragged_batch([3, 4])
    model = _model(batch)
    _randomize_fusion(model)
    narrow = _random_evidence(2, counts, max_e=2)
    wide = _random_evidence(2, counts, max_e=5, seed=99)
    for f in dataclasses.fields(EvidenceInputs):
        getattr(wide, f.name)[:, :2] = getattr(narrow, f.name)
    assert torch.equal(wide.mask[:, :2], narrow.mask) and not wide.mask[:, 2:].any()
    _assert_equal(_forward(model, batch, narrow), _forward(model, batch, wide), atol=1e-12)


def test_staged_path_matches_the_monolithic_forward():
    """The deployment loop's caching split: board, move encodings, and
    per-candidate tokens encoded once (tokens even one candidate at a time,
    as the loop appends them), only fusion + scoring re-run -- bit-identical
    to forward() with the full set."""
    counts = [4, 6]
    batch = _ragged_batch(counts)
    model = _model(batch)
    _randomize_fusion(model)
    evidence = _random_evidence(2, [2, 2], max_e=2)

    full = _forward(model, batch, evidence)

    with torch.no_grad():
        board, g = model.encode_board(batch["input_spatial"], batch["input_scalar"])
        e = model.encode_moves(board, *(batch[k] for k in _MOVE_KEYS), batch["move_pos_id"])
        # Tokens arrive one loop iteration at a time; each is encoded alone
        # and never touched again.
        token_parts, feat_parts = [], []
        for i in range(2):
            one = EvidenceInputs(
                **{
                    f.name: getattr(evidence, f.name)[:, i : i + 1]
                    for f in dataclasses.fields(EvidenceInputs)
                }
            )
            tokens_i, feats_i = model.encode_evidence(board, one)
            token_parts.append(tokens_i)
            feat_parts.append(feats_i)
        tokens = torch.cat(token_parts, dim=1)
        feats = torch.cat(feat_parts, dim=1)
        board_c, g_c = model.evidence_fusion(board, g, tokens, feats, evidence.mask)
        staged = model.score_moves(board_c, g_c, e, batch["move_pos_id"])

    for name in full:
        torch.testing.assert_close(staged[name], full[name], rtol=0, atol=1e-12)


def test_gradients_stay_finite_on_mixed_prefix_batches():
    """Empty evidence rows must not poison training: an all-masked attention
    row softmaxes to NaN, and the output gate's multiply-by-zero would not
    scrub NaN out of the backward pass."""
    counts = [3, 4]
    batch = _ragged_batch(counts)
    model = _model(batch).train()
    _randomize_fusion(model)
    evidence = _random_evidence(2, [2, 0], max_e=2)

    out = model(*(batch[k] for k in _FORWARD_KEYS), evidence=evidence)
    loss = out["wld"].sum() + out["score_diff"].sum() + out["planes"].sum()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads
    for grad in grads:
        assert torch.isfinite(grad).all()
    # The fusion actually trains: some of its parameters received gradient.
    fusion_grads = [p.grad for p in model.evidence_fusion.parameters() if p.grad is not None]
    assert any(float(g.abs().sum()) > 0 for g in fusion_grads)


# --- the .sobs -> EvidenceInputs builder (FFI-backed) ---


def _synthetic_sobs(k: int):
    from scribblez.sim_evidence.sobs import MOVE_PLAY, RECORD_DTYPE

    rec = np.zeros(k, dtype=RECORD_DTYPE)
    for i in range(k):
        move = rec["move"][i]
        move["type"] = MOVE_PLAY
        move["horizontal"] = 1
        move["start"] = 7
        move["square_mask"] = (1 << (3 + i)) | (1 << (4 + i))
        move["num_played"] = 2
        move["glyphs"][:2] = [1 + i, 2 + i]
        move["score"] = 10 * (i + 1)
        obs = rec["obs"][i]
        obs["n"] = 40
        obs["wins"], obs["draws"], obs["losses"] = 25, 5, 10
        obs["delta_sum"] = 40 * 12
        obs["delta_sq_sum"] = 40 * (12**2 + 25)
        obs["opp_next_count"][:] = np.arange(225) % 41
        obs["self_next_count"][:] = 3
    return rec["move"], rec["obs"]


def _first_pass(k: int, seed: int = 5):
    gen = torch.Generator().manual_seed(seed)
    return {
        "wld": torch.randn(k, 3, generator=gen),
        "score_diff": torch.randn(k, 2, generator=gen),
        "planes": torch.randn(k, 4, 225, generator=gen),
    }


def test_builder_assembles_both_halves():
    from scribblez.move_set_eval.evidence import build_evidence_inputs
    from scribblez.sim_evidence.sobs import move_footprint

    k, max_e = 2, 4
    moves, obs = _synthetic_sobs(k)
    first_pass = _first_pass(k)
    ev = build_evidence_inputs(moves, obs, 30, first_pass, max_e=max_e)

    assert ev.mask.shape == (1, max_e)
    assert int(ev.mask.sum()) == k
    assert ev.obs_planes.shape == (1, max_e, NUM_EVIDENCE_PLANES, 15, 15)
    assert ev.obs_scalars.shape == (1, max_e, NUM_EVIDENCE_SCALARS)

    # Observed planes are counts normalized by the rollout count...
    expected = (np.arange(225) % 41).reshape(15, 15) / 40.0
    np.testing.assert_allclose(ev.obs_planes[0, 0, 0].numpy(), expected, atol=1e-6)
    # ...the prediction half is the first pass squashed to probabilities...
    np.testing.assert_allclose(
        ev.obs_planes[0, 0, 4:8].numpy(),
        torch.sigmoid(first_pass["planes"][0]).reshape(4, 15, 15).numpy(),
        atol=1e-6,
    )
    # ...and the last channel is the candidate's own footprint.
    np.testing.assert_array_equal(ev.obs_planes[0, 1, 8].numpy(), move_footprint(moves[1]))

    np.testing.assert_allclose(
        ev.obs_scalars[0, 0, :3].numpy(), [25 / 40, 5 / 40, 10 / 40], atol=1e-6
    )
    wld = torch.softmax(first_pass["wld"], dim=1)
    np.testing.assert_allclose(ev.obs_scalars[0, :k, 6:9].numpy(), wld.numpy(), atol=1e-6)

    # The move half is the engine's encoding of the same records.
    enc = move_enc.encode_moves(np.asarray(moves), np.full(k, 30, dtype=np.int32))
    np.testing.assert_array_equal(ev.letters[0, :k].numpy(), enc["letters"])
    assert not ev.letters[0, k:].any()  # padding rows are zeroed

    # Padded rows are unmasked and zero.
    assert not ev.mask[0, k:].any()


def test_builder_rejects_sets_that_do_not_fit():
    from scribblez.move_set_eval.evidence import build_evidence_inputs

    moves, obs = _synthetic_sobs(3)
    with pytest.raises(ValueError, match="does not fit"):
        build_evidence_inputs(moves, obs, 0, _first_pass(3), max_e=2)
    with pytest.raises(ValueError, match="does not fit"):
        build_evidence_inputs(moves[:0], obs[:0], 0, _first_pass(0), max_e=2)


def test_collated_builder_output_drives_the_model():
    """End to end: synthetic .sobs rows + the model's own first pass on the
    evidence candidates, collated across positions, condition a forward."""
    from scribblez.move_set_eval.evidence import build_evidence_inputs, collate_evidence

    counts = [3, 4]
    batch = _ragged_batch(counts)
    model = _model(batch)
    _randomize_fusion(model)

    per_position = []
    for i, k in enumerate([2, 1]):
        moves, obs = _synthetic_sobs(k)
        per_position.append(
            build_evidence_inputs(moves, obs, 15 * i, _first_pass(k), max_e=3, dtype=torch.float64)
        )
    evidence = collate_evidence(per_position)
    assert evidence.mask.shape == (2, 3)

    out = _forward(model, batch, evidence)
    assert out["wld"].shape == (sum(counts), 3)
    assert not torch.equal(out["wld"], _forward(model, batch)["wld"])
