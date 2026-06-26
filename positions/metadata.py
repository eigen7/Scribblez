from dataclasses import dataclass

@dataclass
class Position:
    game_name: str
    lexicon: str
    urls: list[str]
    comment: str


positions = [
    Position('giant_E_hook_end_game', 'NWL23', [],
             """
Alice and Bob have just bingoed. Bob is about to refill his rack with the last 7 tiles from the
bag. He is currently trailing by 30. Normally, after refilling your rack to 7 tiles with the bag
empty, a 30 point deficit means you are likely to lose. In this position, however, whoever draws the
E has a 108 point move available at A15, which should win the game. From Bob's perspective, before
he draws, each player is equally likely to draw the E, so he has a 50% chance of winning.
             """),
]
