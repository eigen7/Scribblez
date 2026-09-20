# Setup survey examples

Positions from HastyBot self-play where a high-value setup play ranked outside the
hasty top 10 simmed best against the top 10, strongest first. Each GCG ends on
the move the game actually played; `neural_rank_tool --gcg <file> --turn <turn>` opens
the decision point. Win% and spread (mean final score differential, mover's view) pool
both replicas' rollouts; the held-out gains are the setup pick of one replica valued on
the other, minus the same for the top-10 pick, one figure per replica assignment.

Regenerate this directory (games, sims and all; `--slog-dir` is scratch space) with:

```
./py/scripts/sim_candidate_survey.py --slog-dir target/setup-survey --generate-games 3000 --game-seed 1 --open-leaves --recipe setup --rollouts 1000 --max-positions 300 --review-dir positions/NWL23/setup-survey-examples
```

| file | turn | setup play (hasty rank) | win% | spread | best top-10 move | win% | spread | win gains | spread gains |
|---|---|---|---|---|---|---|---|---|---|
| hasty-seed1-2-g57-turn20.gcg | 20 | K5 LI (#65) | 61.1 | -38.2 | 5K R.J | 20.5 | -35.5 | +40.1, +43.9 | -3.7, -9.4 |
| hasty-seed1-0-g26-turn22.gcg | 22 | I13 .EE (#719) | 9.1 | -141.6 | 9K EX | 0.2 | -108.1 | +9.2, +8.6 | -33.3, -33.6 |
| hasty-seed1-0-g891-turn16.gcg | 16 | 15D VEIN. (#43) | 45.7 | -6.5 | C1 .INK | 39.2 | -13.2 | +7.3, +7.7 | +7.5, +7.9 |
| hasty-seed1-2-g419-turn16.gcg | 16 | 15H .MO (#366) | 77.8 | +52.4 | B11 J..ED | 72.5 | +41.3 | +5.1, +6.5 | +8.4, +13.4 |
| hasty-seed1-0-g952-turn4.gcg | 4 | I9 FE (#11) | 73.8 | +57.0 | 11E FAZ. | 68.5 | +41.3 | +8.0, +3.5 | +19.6, +10.4 |
| hasty-seed1-1-g722-turn16.gcg | 16 | 3J AL.C (#44) | 21.2 | -35.0 | M1 ECLAT | 17.7 | -52.7 | +4.0, +5.6 | +19.6, +14.4 |
| hasty-seed1-2-g390-turn19.gcg | 19 | 15D .AUN (#105) | 10.3 | -51.5 | 15D .UANO | 6.5 | -68.8 | +2.2, +5.4 | +15.8, +18.8 |
| hasty-seed1-0-g923-turn14.gcg | 14 | N11 URSAE (#37) | 78.2 | +42.5 | 8L ..XA | 75.5 | +42.6 | +3.2, +2.2 | +0.8, -1.1 |
| hasty-seed1-0-g941-turn9.gcg | 9 | 15G B.OOIE (#28) | 47.9 | -4.9 | 4D OBOE | 45.8 | -8.6 | +2.0, +3.4 | +5.0, +4.4 |
| hasty-seed1-0-g605-turn22.gcg | 22 | 12B AW (#235) | 10.9 | -28.5 | 14D PAW | 8.3 | -35.2 | +2.8, +2.4 | +7.3, +6.0 |
| hasty-seed1-2-g143-turn13.gcg | 13 | 3E LAP (#18) | 42.0 | -11.0 | 7L AJI | 40.3 | -15.0 | +2.9, +2.1 | +5.3, +2.0 |
| hasty-seed1-0-g955-turn13.gcg | 13 | C9 FA.O (#74) | 52.4 | +4.9 | 15A IF | 50.8 | -0.0 | +5.3, -0.7 | +5.5, -1.0 |
| hasty-seed1-0-g255-turn19.gcg | 19 | 13E .ERO (#17) | 4.2 | -94.1 | 12B JEO. | 2.5 | -103.7 | +2.8, +1.2 | +10.4, +2.5 |
| hasty-seed1-2-g75-turn16.gcg | 16 | 3F UH (#20) | 86.6 | +36.5 | 3C RALPH | 84.8 | +31.5 | +4.2, -0.6 | +7.0, +3.0 |
| hasty-seed1-2-g283-turn10.gcg | 10 | G11 .APE (#51) | 55.6 | +10.7 | J2 JOYPA. | 56.2 | +11.2 | -0.6, +3.7 | -1.3, +6.5 |
