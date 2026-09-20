# Sim survey examples

Positions from HastyBot self-play where a play ranked outside the hasty top 10 beat
the best top-10 move. A screening sim of every candidate singled out its few best
outside plays; a longer confirming sim on fresh rollouts, of those and the top moves, put
it at least 2 standard errors (of the paired win difference) above the best
of them. Strongest first. Each GCG ends on the move the game actually played;
`neural_rank_tool --gcg <file> --turn <turn>` opens the decision point. Win% and spread
(mean final score differential, mover's view) are the confirming sim's.

Regenerate this directory (games, sims and all; `--slog-dir` is scratch space) with:

```
./py/scripts/sim_candidate_survey.py --slog-dir target/all-plays-survey --generate-games 3000 --game-seed 1 --open-leaves --rollouts 1000 --confirm-rollouts 5000 --confirm-picks 5 --max-positions 300 --review-dir positions/NWL23/sim-survey-examples
```

| file | turn | outside play (hasty rank) | win% | spread | best top move | win% | spread | win gain | sigmas | spread gain |
|---|---|---|---|---|---|---|---|---|---|---|
| hasty-seed1-0-g391-turn20.gcg | 20 | H1 .A. (#127) | 75.5 | -1.4 | C12 .OE. | 0.0 | -34.7 | +75.5 | +124.2 | +33.3 |
| hasty-seed1-0-g391-turn20.gcg | 20 | I3 .O (#119) | 75.0 | -10.0 | C12 .OE. | 0.0 | -34.7 | +75.0 | +122.5 | +24.7 |
| hasty-seed1-0-g900-turn19.gcg | 19 | 10L ...S (#17) | 33.6 | -35.4 | J10 JONE. | 1.4 | -60.0 | +32.1 | +45.7 | +24.6 |
| hasty-seed1-0-g900-turn19.gcg | 19 | 4D .....S (#132) | 33.6 | -46.0 | J10 JONE. | 1.4 | -60.0 | +32.1 | +45.7 | +14.1 |
| hasty-seed1-0-g900-turn19.gcg | 19 | E4 ...S (#213) | 33.6 | -50.4 | J10 JONE. | 1.4 | -60.0 | +32.1 | +45.7 | +9.6 |
| hasty-seed1-0-g900-turn19.gcg | 19 | 8A ....S (#215) | 33.6 | -50.4 | J10 JONE. | 1.4 | -60.0 | +32.1 | +45.7 | +9.6 |
| hasty-seed1-0-g900-turn19.gcg | 19 | H4 .....S (#214) | 33.6 | -51.0 | J10 JONE. | 1.4 | -60.0 | +32.1 | +45.7 | +9.1 |
| hasty-seed1-1-g463-turn24.gcg | 24 | C9 .F (#50) | 75.2 | +0.5 | 8A OFT | 24.8 | -13.6 | +50.5 | +41.3 | +14.1 |
| hasty-seed1-0-g47-turn21.gcg | 21 | C1 U.FIT (#15) | 100.0 | +22.6 | G9 FU. | 83.6 | +35.2 | +16.4 | +31.3 | -12.6 |
| hasty-seed1-0-g47-turn21.gcg | 21 | C1 U.CI (#63) | 100.0 | +21.4 | G9 FU. | 83.6 | +35.2 | +16.4 | +31.3 | -13.9 |
| hasty-seed1-0-g47-turn21.gcg | 21 | C1 A.TIC (#89) | 100.0 | +21.5 | G9 FU. | 83.6 | +35.2 | +16.4 | +31.3 | -13.8 |
| hasty-seed1-0-g47-turn21.gcg | 21 | 3B TUI (#44) | 100.0 | +19.9 | G9 FU. | 83.6 | +35.2 | +16.4 | +31.3 | -15.3 |
| hasty-seed1-0-g47-turn21.gcg | 21 | 3A UTA (#78) | 100.0 | +15.1 | G9 FU. | 83.6 | +35.2 | +16.4 | +31.3 | -20.1 |
| hasty-seed1-0-g574-turn21.gcg | 21 | 2F VO. (#22) | 50.3 | -25.8 | F2 VRO. | 27.5 | -71.6 | +22.8 | +29.9 | +45.8 |
| hasty-seed1-1-g906-turn20.gcg | 20 | G13 P.G (#44) | 16.4 | -46.8 | H11 STI.G | 1.4 | -20.5 | +15.0 | +28.8 | -26.3 |
| hasty-seed1-0-g507-turn19.gcg | 19 | 7B ARO (#179) | 27.2 | -31.9 | L7 OI | 6.8 | -32.1 | +20.4 | +28.5 | +0.2 |
| hasty-seed1-0-g507-turn19.gcg | 19 | N11 I. (#43) | 23.7 | -32.3 | L7 OI | 6.8 | -32.1 | +16.9 | +25.0 | -0.2 |
| hasty-seed1-2-g94-turn25.gcg | 25 | B13 .T. (#101) | 9.9 | -50.4 | A8 UPDAT. | 0.0 | -63.3 | +9.9 | +23.4 | +12.8 |
| hasty-seed1-2-g94-turn25.gcg | 25 | N1 ..T (#111) | 9.9 | -51.5 | A8 UPDAT. | 0.0 | -63.3 | +9.9 | +23.4 | +11.8 |
| hasty-seed1-2-g94-turn25.gcg | 25 | 9L T. (#134) | 9.9 | -52.5 | A8 UPDAT. | 0.0 | -63.3 | +9.9 | +23.4 | +10.7 |
| hasty-seed1-2-g94-turn25.gcg | 25 | 15M T.. (#148) | 9.9 | -52.5 | A8 UPDAT. | 0.0 | -63.3 | +9.9 | +23.4 | +10.8 |
| hasty-seed1-2-g94-turn25.gcg | 25 | 9H .T (#147) | 9.9 | -53.4 | A8 UPDAT. | 0.0 | -63.3 | +9.9 | +23.4 | +9.8 |
| hasty-seed1-2-g244-turn17.gcg | 17 | 5L ..GS (#143) | 7.5 | -134.3 | 3C G.SH | 0.0 | -119.2 | +7.5 | +20.1 | -15.2 |
| hasty-seed1-0-g507-turn19.gcg | 19 | 7K .I (#28) | 19.8 | -37.1 | L7 OI | 6.8 | -32.1 | +13.0 | +19.9 | -5.0 |
| hasty-seed1-1-g613-turn20.gcg | 20 | N6 .ARKET (#200) | 100.0 | +54.8 | 10B P.RV | 92.8 | +67.4 | +7.2 | +19.9 | -12.6 |
| hasty-seed1-1-g613-turn20.gcg | 20 | N6 .ERK (#294) | 100.0 | +51.9 | 10B P.RV | 92.8 | +67.4 | +7.2 | +19.8 | -15.5 |
| hasty-seed1-0-g507-turn19.gcg | 19 | 6H .O (#41) | 19.0 | -33.9 | L7 OI | 6.8 | -32.1 | +12.2 | +19.5 | -1.8 |
| hasty-seed1-2-g244-turn17.gcg | 17 | 3L ..GS (#78) | 6.8 | -120.6 | 3C G.SH | 0.0 | -119.2 | +6.8 | +19.0 | -1.4 |
| hasty-seed1-1-g613-turn20.gcg | 20 | N6 .ARK (#267) | 99.6 | +49.3 | 10B P.RV | 92.8 | +67.4 | +6.9 | +18.9 | -18.0 |
| hasty-seed1-1-g851-turn21.gcg | 21 | I2 OW. (#87) | 82.8 | +40.8 | 3B LOW | 68.0 | +27.6 | +14.8 | +18.6 | +13.2 |
| hasty-seed1-0-g507-turn19.gcg | 19 | 10E O. (#61) | 18.2 | -37.6 | L7 OI | 6.8 | -32.1 | +11.4 | +18.4 | -5.5 |
| hasty-seed1-1-g613-turn20.gcg | 20 | N6 .ART (#367) | 99.4 | +48.9 | 10B P.RV | 92.8 | +67.4 | +6.6 | +17.7 | -18.5 |
| hasty-seed1-2-g93-turn22.gcg | 22 | 13B N..I (#69) | 100.0 | +45.5 | E11 IN | 97.2 | +64.7 | +2.8 | +17.3 | -19.2 |
| hasty-seed1-2-g93-turn22.gcg | 22 | E11 INS (#16) | 100.0 | +41.0 | E11 IN | 97.2 | +64.7 | +2.8 | +17.3 | -23.7 |
| hasty-seed1-2-g93-turn22.gcg | 22 | E11 ONS (#15) | 100.0 | +36.8 | E11 IN | 97.2 | +64.7 | +2.8 | +17.3 | -27.9 |
| hasty-seed1-2-g93-turn22.gcg | 22 | F9 .ORNU (#17) | 100.0 | +36.6 | E11 IN | 97.2 | +64.7 | +2.8 | +17.3 | -28.1 |
| hasty-seed1-2-g93-turn22.gcg | 22 | K8 .INOUS (#36) | 100.0 | +33.5 | E11 IN | 97.2 | +64.7 | +2.8 | +17.3 | -31.2 |
| hasty-seed1-2-g933-turn18.gcg | 18 | 6A F. (#64) | 26.4 | -46.8 | 13C H.G | 13.7 | -49.3 | +12.8 | +16.9 | +2.5 |
| hasty-seed1-2-g933-turn18.gcg | 18 | 7K .F (#161) | 26.4 | -50.8 | 13C H.G | 13.7 | -49.3 | +12.8 | +16.9 | -1.5 |
| hasty-seed1-1-g613-turn20.gcg | 20 | N6 .ET (#404) | 99.2 | +49.8 | 10B P.RV | 92.8 | +67.4 | +6.4 | +16.5 | -17.6 |
| hasty-seed1-2-g370-turn21.gcg | 21 | K11 SORI (#22) | 20.3 | -80.0 | K11 SOP | 10.2 | -78.2 | +10.1 | +16.1 | -1.9 |
| hasty-seed1-2-g244-turn17.gcg | 17 | 3L ..SE (#514) | 4.6 | -154.6 | 3C G.SH | 0.0 | -119.2 | +4.6 | +15.5 | -35.4 |
| hasty-seed1-1-g851-turn21.gcg | 21 | 3C OW (#32) | 80.0 | +36.4 | 3B LOW | 68.0 | +27.6 | +12.0 | +15.4 | +8.8 |
| hasty-seed1-1-g852-turn20.gcg | 20 | 14K T.ARS (#1364) | 6.5 | -90.5 | K11 ALIT | 1.2 | -92.6 | +5.4 | +15.1 | +2.2 |
| hasty-seed1-0-g839-turn23.gcg | 23 | F7 ST.. (#51) | 94.3 | +43.2 | 14M PU. | 86.1 | +65.2 | +8.3 | +14.4 | -22.0 |
| hasty-seed1-1-g956-turn22.gcg | 22 | 10H .ROMID (#27) | 9.7 | -105.3 | A1 ..R | 3.0 | -110.1 | +6.7 | +13.5 | +4.7 |
| hasty-seed1-1-g852-turn20.gcg | 20 | 14K L.ARS (#1344) | 5.7 | -94.3 | K11 ALIT | 1.2 | -92.6 | +4.5 | +13.0 | -1.7 |
| hasty-seed1-1-g852-turn20.gcg | 20 | 14K R.ALS (#1345) | 5.7 | -94.8 | K11 ALIT | 1.2 | -92.6 | +4.5 | +13.0 | -2.1 |
| hasty-seed1-1-g852-turn20.gcg | 20 | 12K L.TAI (#458) | 5.9 | -100.9 | K11 ALIT | 1.2 | -92.6 | +4.7 | +12.8 | -8.3 |
| hasty-seed1-2-g593-turn20.gcg | 20 | N11 D.RM (#17) | 77.2 | +26.2 | N11 P.RM | 69.0 | +17.1 | +8.2 | +12.6 | +9.1 |
| hasty-seed1-1-g956-turn22.gcg | 22 | A1 ..RMER (#106) | 8.7 | -120.4 | A1 ..R | 3.0 | -110.1 | +5.7 | +12.0 | -10.3 |
| hasty-seed1-2-g469-turn21.gcg | 21 | O3 SOUS (#24) | 81.8 | +8.9 | O3 SOUR | 79.0 | +15.6 | +2.8 | +11.9 | -6.7 |
| hasty-seed1-0-g113-turn22.gcg | 22 | 8D E. (#213) | 18.8 | -45.7 | 15B DUO | 11.4 | -41.7 | +7.5 | +11.7 | -3.9 |
| hasty-seed1-1-g956-turn22.gcg | 22 | B5 DIRE (#120) | 8.5 | -119.4 | A1 ..R | 3.0 | -110.1 | +5.5 | +11.7 | -9.3 |
| hasty-seed1-1-g956-turn22.gcg | 22 | A1 ..IRED (#81) | 8.5 | -121.5 | A1 ..R | 3.0 | -110.1 | +5.5 | +11.7 | -11.5 |
| hasty-seed1-1-g956-turn22.gcg | 22 | B5 RIDE (#121) | 8.5 | -125.2 | A1 ..R | 3.0 | -110.1 | +5.5 | +11.7 | -15.1 |
| hasty-seed1-0-g113-turn22.gcg | 22 | 15B DUE (#44) | 20.0 | -36.5 | 15B DUO | 11.4 | -41.7 | +8.6 | +11.5 | +5.3 |
| hasty-seed1-1-g852-turn20.gcg | 20 | 11L .IRT (#619) | 4.5 | -91.6 | K11 ALIT | 1.2 | -92.6 | +3.4 | +10.4 | +1.1 |
| hasty-seed1-0-g839-turn23.gcg | 23 | O5 PSI (#87) | 91.7 | +47.8 | 14M PU. | 86.1 | +65.2 | +5.7 | +10.1 | -17.4 |
| hasty-seed1-0-g839-turn23.gcg | 23 | 8F P... (#12) | 92.1 | +44.6 | 14M PU. | 86.1 | +65.2 | +6.0 | +10.0 | -20.7 |
| hasty-seed1-0-g113-turn22.gcg | 22 | 8C OO. (#14) | 17.4 | -37.4 | 15B DUO | 11.4 | -41.7 | +6.1 | +9.9 | +4.4 |
| hasty-seed1-1-g341-turn21.gcg | 21 | 10B .AN (#263) | 9.6 | -56.1 | O1 OI | 4.8 | -59.2 | +4.8 | +9.7 | +3.0 |
| hasty-seed1-0-g839-turn23.gcg | 23 | 8F T... (#29) | 92.0 | +40.6 | 14M PU. | 86.1 | +65.2 | +6.0 | +9.6 | -24.6 |
| hasty-seed1-1-g341-turn21.gcg | 21 | 10B .OT (#177) | 9.6 | -57.1 | O1 OI | 4.8 | -59.2 | +4.8 | +9.5 | +2.0 |
| hasty-seed1-0-g113-turn22.gcg | 22 | D6 .OO. (#18) | 17.1 | -38.1 | 15B DUO | 11.4 | -41.7 | +5.7 | +9.5 | +3.7 |
| hasty-seed1-1-g324-turn17.gcg | 17 | M7 R.SACEA (#69) | 57.6 | +10.3 | M11 OCA | 49.8 | +3.8 | +7.8 | +9.3 | +6.6 |
| hasty-seed1-2-g593-turn20.gcg | 20 | N11 D.RMA (#31) | 76.4 | +23.0 | N11 P.RM | 69.0 | +17.1 | +7.4 | +9.0 | +5.9 |
| hasty-seed1-1-g161-turn18.gcg | 18 | 12H ..T (#94) | 11.2 | -87.3 | 4I M.TT | 6.5 | -89.3 | +4.7 | +8.8 | +2.0 |
| hasty-seed1-1-g34-turn10.gcg | 10 | 15J NO.AU (#41) | 48.8 | -1.8 | 15J NO.AUX | 40.9 | -16.3 | +7.9 | +8.5 | +14.5 |
| hasty-seed1-1-g161-turn18.gcg | 18 | 6D T.. (#95) | 10.7 | -88.6 | 4I M.TT | 6.5 | -89.3 | +4.2 | +8.1 | +0.7 |
| hasty-seed1-0-g113-turn22.gcg | 22 | 13B .OUSE (#146) | 16.3 | -36.8 | 15B DUO | 11.4 | -41.7 | +5.0 | +8.0 | +5.0 |
| hasty-seed1-2-g318-turn18.gcg | 18 | H2 RI.T (#204) | 88.8 | +33.5 | H1 DIF. | 84.7 | +36.8 | +4.1 | +6.7 | -3.3 |
| hasty-seed1-2-g318-turn18.gcg | 18 | N2 AID (#37) | 88.8 | +40.0 | H1 DIF. | 84.7 | +36.8 | +4.2 | +6.7 | +3.2 |
| hasty-seed1-0-g887-turn18.gcg | 18 | N3 .LUGE (#11) | 82.9 | +58.2 | 13C LU.HERN | 78.5 | +50.4 | +4.3 | +6.0 | +7.8 |
| hasty-seed1-1-g324-turn17.gcg | 17 | N4 ORAC.ES (#370) | 54.9 | -0.3 | M11 OCA | 49.8 | +3.8 | +5.1 | +5.8 | -4.1 |
| hasty-seed1-1-g324-turn17.gcg | 17 | M10 CASE (#197) | 54.2 | +7.7 | M11 OCA | 49.8 | +3.8 | +4.4 | +5.6 | +3.9 |
| hasty-seed1-1-g792-turn21.gcg | 21 | 11H STRIVE (#35) | 100.0 | +93.4 | K7 RIVET | 99.4 | +96.3 | +0.6 | +5.6 | -2.8 |
| hasty-seed1-1-g792-turn21.gcg | 21 | 11H STIVER (#36) | 100.0 | +92.5 | K7 RIVET | 99.4 | +96.3 | +0.6 | +5.6 | -3.8 |
| hasty-seed1-1-g792-turn21.gcg | 21 | K7 STIVER (#21) | 100.0 | +89.1 | K7 RIVET | 99.4 | +96.3 | +0.6 | +5.6 | -7.1 |
| hasty-seed1-1-g792-turn21.gcg | 21 | K7 RIVETS (#22) | 100.0 | +86.5 | K7 RIVET | 99.4 | +96.3 | +0.6 | +5.6 | -9.8 |
| hasty-seed1-1-g792-turn21.gcg | 21 | 14A TRISTE (#17) | 100.0 | +80.7 | K7 RIVET | 99.4 | +96.3 | +0.6 | +5.6 | -15.6 |
| hasty-seed1-1-g608-turn12.gcg | 12 | 8F T... (#236) | 3.5 | -171.2 | G7 P.N | 1.8 | -164.8 | +1.7 | +5.5 | -6.5 |
| hasty-seed1-2-g933-turn18.gcg | 18 | 2A F.H (#169) | 17.3 | -66.0 | 13C H.G | 13.7 | -49.3 | +3.6 | +5.5 | -16.7 |
| hasty-seed1-0-g967-turn16.gcg | 16 | E3 PIN. (#26) | 99.6 | +105.5 | 12A WIZ | 98.6 | +93.1 | +1.0 | +5.4 | +12.4 |
| hasty-seed1-1-g161-turn18.gcg | 18 | F3 FL.. (#52) | 9.0 | -92.2 | 4I M.TT | 6.5 | -89.3 | +2.5 | +5.4 | -2.9 |
| hasty-seed1-2-g111-turn20.gcg | 20 | 10J .ONEY (#13) | 17.7 | -62.3 | 4J .ONEY | 15.5 | -62.2 | +2.2 | +5.4 | -0.0 |
| hasty-seed1-2-g933-turn18.gcg | 18 | L9 HA (#277) | 17.2 | -67.4 | 13C H.G | 13.7 | -49.3 | +3.5 | +5.3 | -18.1 |
| hasty-seed1-1-g608-turn12.gcg | 12 | J10 ..T (#218) | 3.4 | -170.4 | G7 P.N | 1.8 | -164.8 | +1.6 | +5.0 | -5.6 |
| hasty-seed1-2-g602-turn17.gcg | 17 | 2H .LOW (#12) | 69.2 | +15.8 | O9 WA.H | 65.0 | +12.5 | +4.2 | +4.9 | +3.4 |
| hasty-seed1-2-g111-turn20.gcg | 20 | 6J .OONEY (#36) | 18.2 | -68.9 | 4J .ONEY | 15.5 | -62.2 | +2.7 | +4.8 | -6.7 |
| hasty-seed1-1-g608-turn12.gcg | 12 | K9 T.. (#261) | 3.3 | -172.2 | G7 P.N | 1.8 | -164.8 | +1.4 | +4.7 | -7.4 |
| hasty-seed1-1-g608-turn12.gcg | 12 | B13 .T (#237) | 3.2 | -171.3 | G7 P.N | 1.8 | -164.8 | +1.4 | +4.6 | -6.5 |
| hasty-seed1-2-g840-turn21.gcg | 21 | C4 AMI (#110) | 27.4 | -12.4 | C7 AMIA | 23.2 | -7.1 | +4.1 | +4.4 | -5.2 |
| hasty-seed1-1-g847-turn17.gcg | 17 | B5 WIVE (#15) | 11.4 | -45.6 | K9 WIVE. | 9.0 | -47.0 | +2.3 | +4.3 | +1.4 |
| hasty-seed1-0-g368-turn20.gcg | 20 | H11 HE.K (#160) | 100.0 | +140.7 | H11 HO.KS | 99.7 | +134.4 | +0.3 | +4.0 | +6.2 |
| hasty-seed1-0-g562-turn10.gcg | 10 | 13M EGO (#18) | 2.8 | -129.5 | 5A nOOdGED | 1.6 | -133.2 | +1.1 | +3.9 | +3.7 |
| hasty-seed1-1-g469-turn17.gcg | 17 | 11A INDRI (#11) | 3.5 | -114.1 | 11C IRON | 2.2 | -117.3 | +1.3 | +3.8 | +3.1 |
| hasty-seed1-1-g324-turn17.gcg | 17 | M10 SOCA (#74) | 52.8 | +7.0 | M11 OCA | 49.8 | +3.8 | +3.0 | +3.8 | +3.2 |
| hasty-seed1-0-g646-turn16.gcg | 16 | 7N OI (#43) | 82.6 | +47.0 | 4D QI | 80.0 | +45.8 | +2.6 | +3.7 | +1.2 |
| hasty-seed1-1-g608-turn12.gcg | 12 | E13 .T (#292) | 2.9 | -173.3 | G7 P.N | 1.8 | -164.8 | +1.1 | +3.7 | -8.5 |
| hasty-seed1-0-g569-turn22.gcg | 22 | 9A ES (#112) | 94.7 | +42.0 | -EEEU | 93.0 | +56.5 | +1.7 | +3.7 | -14.4 |
| hasty-seed1-1-g122-turn17.gcg | 17 | O11 F.AT (#71) | 32.1 | -25.3 | 14D QATS | 29.0 | -24.7 | +3.0 | +3.5 | -0.6 |
| hasty-seed1-0-g368-turn20.gcg | 20 | H11 HO.K (#53) | 100.0 | +149.0 | H11 HO.KS | 99.7 | +134.4 | +0.3 | +3.5 | +14.6 |
| hasty-seed1-2-g318-turn18.gcg | 18 | 3J A.IF (#14) | 86.9 | +36.8 | H1 DIF. | 84.7 | +36.8 | +2.2 | +3.4 | -0.1 |
| hasty-seed1-2-g318-turn18.gcg | 18 | N3 ID (#41) | 86.9 | +35.6 | H1 DIF. | 84.7 | +36.8 | +2.2 | +3.3 | -1.3 |
| hasty-seed1-0-g569-turn22.gcg | 22 | 9A ET (#59) | 94.5 | +51.6 | -EEEU | 93.0 | +56.5 | +1.6 | +3.3 | -4.9 |
| hasty-seed1-2-g412-turn16.gcg | 16 | F7 P.SO (#39) | 92.3 | +58.0 | 14H OPE | 90.6 | +63.0 | +1.7 | +3.3 | -5.0 |
| hasty-seed1-1-g699-turn22.gcg | 22 | M1 YEA (#36) | 67.7 | +12.1 | O1 YA | 65.1 | +16.7 | +2.6 | +3.2 | -4.6 |
| hasty-seed1-2-g832-turn16.gcg | 16 | B1 MORAS (#56) | 6.3 | -82.3 | 12A SIMA | 4.9 | -77.3 | +1.4 | +3.2 | -5.0 |
| hasty-seed1-0-g368-turn20.gcg | 20 | 15K K.ETS (#28) | 99.9 | +120.0 | H11 HO.KS | 99.7 | +134.4 | +0.3 | +3.2 | -14.4 |
| hasty-seed1-0-g333-turn16.gcg | 16 | 12A TUNA (#12) | 5.6 | -84.5 | 12A NUTATE | 4.3 | -86.7 | +1.3 | +3.2 | +2.2 |
| hasty-seed1-2-g318-turn18.gcg | 18 | N1 ARID (#28) | 86.8 | +33.9 | H1 DIF. | 84.7 | +36.8 | +2.1 | +3.2 | -3.0 |
| hasty-seed1-1-g171-turn17.gcg | 17 | G11 A.VO (#30) | 99.8 | +123.3 | N5 OVA | 99.4 | +117.0 | +0.4 | +3.1 | +6.3 |
| hasty-seed1-1-g681-turn21.gcg | 21 | 12G SULTAN. (#203) | 67.0 | +31.7 | 10H UN | 64.0 | +18.2 | +2.9 | +3.1 | +13.5 |
| hasty-seed1-0-g527-turn17.gcg | 17 | I11 IC.. (#112) | 2.9 | -149.9 | L10 LIVI. | 2.0 | -134.2 | +0.9 | +3.0 | -15.6 |
| hasty-seed1-2-g942-turn16.gcg | 16 | J2 EVI.E (#12) | 2.5 | -66.0 | 7A FI.UE | 1.7 | -58.4 | +0.7 | +2.9 | -7.6 |
| hasty-seed1-0-g54-turn19.gcg | 19 | M10 MALLEI (#15) | 0.6 | -141.3 | O8 ....MEAL | 0.2 | -111.2 | +0.4 | +2.9 | -30.1 |
| hasty-seed1-0-g646-turn16.gcg | 16 | 3L .OO (#29) | 82.0 | +48.6 | 4D QI | 80.0 | +45.8 | +2.0 | +2.8 | +2.8 |
| hasty-seed1-1-g324-turn17.gcg | 17 | M11 ACE (#47) | 51.4 | +5.2 | M11 OCA | 49.8 | +3.8 | +1.6 | +2.8 | +1.5 |
| hasty-seed1-0-g887-turn18.gcg | 18 | N3 .UGEL (#15) | 80.4 | +53.7 | 13C LU.HERN | 78.5 | +50.4 | +1.9 | +2.6 | +3.3 |
| hasty-seed1-0-g368-turn20.gcg | 20 | 15K K.ET (#102) | 99.9 | +131.6 | H11 HO.KS | 99.7 | +134.4 | +0.2 | +2.6 | -2.8 |
| hasty-seed1-0-g646-turn16.gcg | 16 | 3K O.O (#15) | 81.7 | +48.8 | 4D QI | 80.0 | +45.8 | +1.8 | +2.6 | +3.0 |
| hasty-seed1-2-g840-turn21.gcg | 21 | 11B AM.A (#102) | 25.6 | -3.8 | C7 AMIA | 23.2 | -7.1 | +2.3 | +2.6 | +3.3 |
| hasty-seed1-2-g916-turn15.gcg | 15 | M10 EYAS (#22) | 97.0 | +97.5 | J8 EYE | 96.1 | +105.5 | +0.9 | +2.6 | -8.0 |
| hasty-seed1-0-g646-turn16.gcg | 16 | 3K O.E (#12) | 81.7 | +48.8 | 4D QI | 80.0 | +45.8 | +1.7 | +2.5 | +2.9 |
| hasty-seed1-2-g172-turn12.gcg | 12 | 15L .RIG (#39) | 94.0 | +83.2 | 5B GRAM | 92.8 | +85.4 | +1.2 | +2.5 | -2.2 |
| hasty-seed1-1-g957-turn15.gcg | 15 | K11 MOC (#16) | 4.4 | -98.8 | K8 ICEMAN | 3.5 | -97.1 | +0.9 | +2.4 | -1.7 |
| hasty-seed1-1-g256-turn19.gcg | 19 | N1 Z. (#121) | 6.0 | -115.9 | O11 .ERBY | 4.9 | -74.8 | +1.1 | +2.4 | -41.0 |
| hasty-seed1-0-g368-turn20.gcg | 20 | 15K T.TH (#205) | 99.9 | +130.8 | H11 HO.KS | 99.7 | +134.4 | +0.2 | +2.3 | -3.6 |
| hasty-seed1-0-g967-turn16.gcg | 16 | E3 PI (#191) | 99.1 | +94.3 | 12A WIZ | 98.6 | +93.1 | +0.4 | +2.2 | +1.2 |
| hasty-seed1-1-g966-turn16.gcg | 16 | O4 .IEVERS (#26) | 72.1 | +38.2 | O4 .AVER | 70.2 | +35.5 | +1.9 | +2.2 | +2.7 |
| hasty-seed1-2-g832-turn16.gcg | 16 | B1 OASIS (#395) | 5.9 | -92.5 | 12A SIMA | 4.9 | -77.3 | +0.9 | +2.2 | -15.2 |
| hasty-seed1-1-g361-turn12.gcg | 12 | 13E RONDO (#37) | 64.1 | +28.9 | 12C ROOD | 62.2 | +26.9 | +2.0 | +2.2 | +2.0 |
