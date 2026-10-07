#pragma once

// Which candidates block the replies other candidates' probes saw: a
// diagnostic for SupremeBot M1a (docs/plans/supreme_bot_m1a.md), testing
// whether a .sprobe's probes carry evidence about candidates they did not
// probe.
//
// Probe i of every candidate deals the opponent the same rack (the rollouts'
// common random numbers), so the reply candidate a's probe i saw is a play
// that rack could also make after candidate b, unless b's tiles took one of
// its squares or changed a cross-check it needed. Whether it could is decided
// by move generation on b's post-move board: a reply is blocked by b when no
// legal play for the rack after b places the same tiles on the same squares.

#include "data/probe_log.h"

#include <cstdint>

namespace scribblez {

class Board;
class Dictionary;

// For position `pos` of a .sprobe, whose pre-move board is `root`: out[r *
// stride + b] = 1 when record r's opponent reply, a play, is blocked by
// candidate b; 0 when it is not, when the reply is not a play, and for the
// record's own candidate. Records are the position's, in .sprobe order;
// stride >= the position's candidate count.
void reply_blocking(const Dictionary& dict, const Board& root, const ProbeReader::Position& pos,
                    int probes, int stride, uint8_t* out);

}  // namespace scribblez
