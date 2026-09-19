# Setup survey examples

Positions from HastyBot self-play where a high-value setup play ranked outside the
hasty top 10 simmed best against the top 10, strongest first. Each GCG ends on
the move the game actually played; `neural_rank_tool --gcg <file> --turn <turn>` opens
the decision point. Win% and spread (mean final score differential, mover's view) pool
both replicas' rollouts; the held-out gains are the setup pick of one replica valued on
the other, minus the same for the top-10 pick, one figure per replica assignment.

| file | turn | setup play (hasty rank) | win% | spread | best top-10 move | win% | spread | win gains | spread gains |
|---|---|---|---|---|---|---|---|---|---|
| 1788832066490927849-local-0-g132-turn5.gcg | 5 | 9F FORME (#14) | 53.6 | +6.2 | O1 OXES | 49.0 | -2.9 | +4.4, +5.7 | +8.2, +7.3 |
| 1788832066490927849-local-0-g937-turn7.gcg | 7 | G2 YEA (#48) | 78.0 | +62.2 | K9 YEAN | 73.2 | +52.9 | +4.2, +5.3 | +10.0, +8.6 |
| 1788832068067600565-local-0-g669-turn15.gcg | 15 | 7B OKE. (#81) | 79.7 | +53.5 | E11 J.B | 75.7 | +48.6 | +5.6, +3.0 | +4.9, +6.7 |
| 1788832066490927849-local-0-g24-turn13.gcg | 13 | 7I KI (#17) | 47.2 | -6.5 | L10 .OOKIE | 44.2 | -9.9 | +1.3, +4.6 | +1.9, +4.9 |
| 1788832066490927849-local-0-g940-turn6.gcg | 6 | E7 ..COT (#30) | 10.2 | -107.6 | 8A HOOT. | 8.3 | -116.6 | +1.7, +4.2 | +7.9, +15.4 |
| 1788832068067600565-local-0-g809-turn2.gcg | 2 | 9I HAG (#18) | 54.0 | +10.2 | 7E AGHA | 52.5 | +7.3 | +4.4, +1.2 | +7.1, +3.8 |
| 1788832165769829233-local-0-g829-turn4.gcg | 4 | 15E NAVE. (#15) | 44.0 | -13.3 | 15H U.VA | 42.9 | -16.8 | +4.2, -1.4 | +5.2, +0.1 |
| 1788832068067600565-local-0-g338-turn17.gcg | 17 | M6 EVE. (#175) | 95.0 | +45.1 | L8 V.RIX | 93.4 | +47.6 | +2.0, +0.1 | -3.1, -1.0 |
| 1788832165769829233-local-0-g253-turn19.gcg | 19 | 13H .OE (#21) | 52.7 | -0.5 | 14A .O.IE | 52.2 | -1.2 | +1.4, -0.3 | +0.8, +0.4 |
| 1788832068067600565-local-0-g125-turn9.gcg | 9 | 13C T.RO (#26) | 53.0 | +5.7 | 13B WR. | 52.5 | +2.9 | +0.7, +0.4 | +4.0, +0.8 |
| 1788832068067600565-local-0-g586-turn13.gcg | 13 | 13F .GO (#99) | 83.0 | +55.6 | 13C JOK.Y | 82.5 | +61.1 | +0.7, +0.4 | -5.3, -5.7 |
| 1788832068067600565-local-0-g570-turn19.gcg | 19 | M3 .ARE (#47) | 5.5 | -58.0 | J9 AX.E | 6.1 | -66.0 | +0.1, +0.7 | +8.0, +13.6 |
| 1788832068067600565-local-0-g201-turn11.gcg | 11 | 15H FEIST. (#21) | 6.9 | -97.7 | 2F FED | 6.5 | -101.6 | +0.6, +0.1 | +4.6, +3.1 |
| 1788832068067600565-local-0-g453-turn20.gcg | 20 | G13 DAM (#21) | 34.9 | -19.8 | 10A JA.M.N | 36.5 | -23.1 | +1.1, -0.5 | +7.0, +5.3 |
| 1788832066490927849-local-0-g259-turn15.gcg | 15 | 11M OI (#314) | 41.9 | -8.5 | K1 LOO. | 42.8 | -8.1 | +2.1, -1.5 | +1.6, -3.2 |
