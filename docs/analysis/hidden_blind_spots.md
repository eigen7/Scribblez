# Hidden-leaves blind spots: worked positions

A blind spot is a position where a move from outside HastyBot's ten best by
static equity out-sims all ten
([blind_spots.md](../blind_spots.md#hidden-leaves-blind-spots-on-the-gpu)).
These positions are the motivating examples for DeepBot
([plans/deep_bot.md](../plans/deep_bot.md)): each one shows something a sim
sees and static equity does not. This page works through a few of them by
hand, as guideposts: what the winning move is doing, and how much of the
sim's verdict to believe.

The positions come from the `hidden_blind_spots` tag `v1`: 1,005 kept
positions out of 68,000 screened, all confirmed at 5,000 rollouts under leaf
network `leaf-conv-join-32-4b` generation 730 and leave inference model
`tilted2` generation 1188. The tag lives on the mount, not in the repo
(`/workspace/mount/tags/hidden_blind_spots/v1/data/gcg/<id>.gcg`), so every
entry here carries its own board. With the tag present,
`./py/scripts/blind_spots_viewer.py --tag v1` shows any of them.

## Reading an entry

- The id is the position's `.gcg` file name: `<chunk>-g<game>-turn<N>`.
- "We" is the player to move. The score is ours first.
- Boards mark premium squares `=` triple word, `-` double word, `"` triple
  letter, `'` double letter. A lowercase letter is a blank.
- A move is written as in the reports: `8H WORD` runs across row 8 from
  column H, `H8 WORD` runs down column H from row 8, and tiles already on the
  board are in parentheses.
- Win % is the confirming sim's. Its standard error is under one point.
- Follow-up scores ("with the leave alone …") are Macondo's move generator on
  the board after the move, given only the tiles kept. The draw can only add
  to them.

## What the sim's opponent is

Every reply in these sims is HastyBot's: the best static-equity move for a
rack drawn from the leave inference model. That opponent does not see a
setup, so it blocks one only when its best-scoring play happens to land
there, and it does not pass while it has a play. A setup's win rate is
therefore measured against an opponent who is not looking for it, and a pass
is measured against an opponent who will not pass back. For most entries this
shades the number; for [one](#indigene-a-pass-that-works-only-against-hastybot)
it is the whole effect.

## Patterns

| Pattern | Entries |
|---|---|
| Set up a premium square only a tile in our rack can reach | [CLINKER](#clinker-an-s-hook-on-the-triple-word-square), [NIM](#nim-a-private-lane), [OIL](#oil-a-y-hook-on-the-triple-word-square), [AD](#ad-two-hook-squares-on-the-triple-word-row), [PITCH](#pitch-the-last-y), [ROUCHE](#rouche-both-ends-of-a-triple-triple) |
| Set up a spot for a heavy tile we hold | [GIVEN](#given-a-spot-for-the-x), [TORI](#tori-a-square-where-the-q-counts-twice), [VIG](#vig-the-same-q-setup-and-why-the-g-matters) |
| Far behind: give up points now for a larger swing later | [O(I)](#oi-fish-one-tile-to-bingo-out), [A(I)M](#aim-the-same-fish), [THAN](#than-declining-a-76-point-bingo) |
| One or two tiles in the bag: who empties it | [pass when ahead](#chord-a-pass-that-keeps-the-bag-closed), [Q(I)](#qi-play-one-tile-leave-one-in-the-bag), [pass when behind](#indigene-a-pass-that-works-only-against-hastybot) |
| Exchange at the last legal moment | [exchange seven](#exchange-seven-with-seven-in-the-bag) |
| Pass on the opening move | [ACDEINT](#acdeint-passing-the-opening-move) |

About half the set (487 of 1,005 positions) has three or fewer tiles in the
bag, and four in five have seven or fewer.

## Positions read by David

The reading of each position in this section is David Shin's, confirmed
against the lexicon and the move generator.

### CLINKER: an S hook on the triple-word square

`1792935124661189568-local-0-g125-turn14`

```
    A B C D E F G H I J K L M N O
  1 = . . ' . . . = . . . ' P U B
  2 . - . . . " . . . Z E T A S .
  3 . . - . . . ' . ' . . U P . .
  4 ' . . - . . . ' . . . F E . '
  5 . . . . - . . . . . B A R . .
  6 . " . . . " . . . T O . . " .
  7 . . ' . . . ' . ' H O . ' . .
  8 = . . ' . . . C L I N K . . =
  9 . . ' . . . ' . ' E . . ' . .
 10 . " . . . " . . . V . . . " .
 11 . . . . A R T E R i E S . . .
 12 ' . C O Y . . ' . N . - . . '
 13 . . - . . l I T I G A T E . .
 14 . - . . . " . . . " D I R L .
 15 = . . ' . . G U Y E D ' N I X
```

Rack **EIJLRST**, score 243–287, 28 in the bag. Unseen: AAAAADDEEEEFGHIIILMMNNNOOOOOQRSUVWW.

| Move | Equity rank | Score | Leave | Win % |
|---|---|---|---|---|
| 8H (CLINK)ER | #46 | 13 | IJLST | 55.0 |
| E10 J(AY) | #2 | 13 | EILRST | 37.2 |
| F6 JILTE(R) | #1 | 31 | RS | 35.4 |
| 8H (CLINK)ERS | #3 | 42 | IJLT | 34.4 |

CLINKER takes only an S, and that S lands on the triple-word square at O8. We
keep ours: with the leave alone, O4 JILTS scores 102 next turn, JILTS for 60
and CLINKERS for 42, both tripled. One other S is unseen, so the opponent can
hit the square first, but we are much the likelier to. Spending the S now on
CLINKERS, for 42, wins 34.4%.

### O(I): fish one tile to bingo out

`1792939424901615146-ssh-0-g166-turn22`

```
    A B C D E F G H I J K L M N O
  1 Q U A N T . . = . . . G I G A
  2 . - R . I M B U E D . R . - .
  3 . . C . . O E . X E N I A . .
  4 ' . . - . N A W . . . T . . '
  5 . . . . - D U H . . - . . . .
  6 . J . . . O . I . " . . . " .
  7 B A D . . . ' N ' . . . ' . .
  8 O Y E Z . . . E . . . ' . . =
  9 R . S . N A V Y ' L . . ' . .
 10 T " P L E W . . . I . . . " .
 11 . . I . E L U D E S - . . . .
 12 ' . T O . . . R . S . - . . '
 13 . . e H . . K I ' O . . - . .
 14 . - S O . " E F . M . . . - .
 15 = . . ' . . A T . E R V I L =
```

Rack **CEINOR?**, score 298–398, 1 in the bag. Unseen: AAEFGOPT.

| Move | Equity rank | Score | Leave | Win % |
|---|---|---|---|---|
| M14 O(I) | #114 | 2 | CEINR? | 19.8 |
| 14J (M)O | #44 | 6 | CEINR? | 10.5 |
| O9 CRONIEs | #1 | 101 | – | 4.6 |

Down 100 with one tile in the bag, a bingo is not enough: CRONIEs for 101
puts us one point ahead holding the last tile against the opponent's seven,
and wins 4.6%. The better shot is to dump one tile, draw the last tile
ourselves, and bingo out next turn, which also collects twice the opponent's
rack. After the dump we hold CEINR? and the drawn tile. If that tile is an A,
G, O, P or T (six of the eight unseen), column O from O9 takes a bingo for
101 to 113 that goes out.

### A(I)M: the same fish

`1792939480730359398-local-0-g4-turn18`

```
    A B C D E F G H I J K L M N O
  1 = . . ' . . . = . . . ' . S =
  2 . H A . . " . . . " . . . O .
  3 . E W . U . ' . ' . . . - L .
  4 ' R E I N . . F . . . - . V '
  5 . B E N C h T O P . - . . A .
  6 . Y . J O " . L O X . . . T I
  7 . . ' U . . ' I ' . . . ' E D
  8 Q U I R E . W O . . . ' . . O
  9 . . ' E . O E . ' . . . ' . C
 10 . M U D . H A . . " . . . " R
 11 . . . . - . V . A R I E T T A
 12 ' . B R A Z I N G . . - . . S
 13 . . - . . . N . ' . . . - . E
 14 . F L A K " G . . " . . . - .
 15 D A I R I E S = . . . ' . . =
```

Rack **AGMOPT?**, score 357–469, 3 in the bag. Unseen: DEEILNNSTY.

| Move | Equity rank | Score | Leave | Win % |
|---|---|---|---|---|
| K10 A(I)M | #594 | 5 | GOPT? | 29.9 |
| 13N M(E) | #487 | 4 | AGOPT? | 29.6 |
| A1 GAP | #3 | 32 | MOT? | 20.7 |
| L4 POsTGAM(E) | #1 | 78 | – | 0.0 |

The same fish, down 112 with three tiles in the bag. POsTGAME for 78 empties
the bag and leaves us 34 behind: 0.0%. A small play that keeps the blank is
the only shot. The five outside moves in the report are all plays of one or
two tiles that keep it, and all win 28 to 30%.

### NIM: a private lane

`1792939586653428488-local-0-g26-turn18`

```
    A B C D E F G H I J K L M N O
  1 = V . ' . . . = . . . ' . . T
  2 . A A . . " . . . " . . . H I
  3 . Q I . . . ' . D Y E . W I Z
  4 ' U N - . . . B A A E D . N E
  5 . I . . V . J U G . W E E N S
  6 . T R A I T O R S " . . . Y .
  7 H A E . L . ' P ' . . f ' . .
  8 O . C ' L . . E . . . I . . =
  9 O . O M . . A D I E U X ' . .
 10 T " D O . " . . . " . U . " .
 11 E . i F - . . . . . - R . . .
 12 R . N O . . . ' . . . E . . '
 13 . . G . . . ' . ' . . S - . .
 14 . - . . . " . . . " . . . - .
 15 = . . ' . . . = . . . ' . . =
```

Rack **AFIMNNS**, score 339–310, 10 in the bag. Unseen: ABCEEGIKLLOOPRRTT.

| Move | Equity rank | Score | Leave | Win % |
|---|---|---|---|---|
| K12 NIM | #124 | 9 | AFNS | 93.1 |
| 12K F(E)N | #10 | 6 | AIMNS | 87.5 |
| M12 FIN | #1 | 21 | AMNS | 85.2 |

NIM takes no letter in front and only an S behind, on K15. We hold the last
S, and no blank is unseen, so row 15 through K15 is a lane only we can play
in. It reaches the triple-word square at H15 on one side and the one at O15
on the other.

### GIVEN: a spot for the X

`1792939885730522277-local-0-g116-turn17`

```
    A B C D E F G H I J K L M N O
  1 = . . ' . . . = . . A L I Y A
  2 . - . . . " . . . H . O . U .
  3 . . - . . . ' . ' O P T - Z .
  4 ' . . - . . . ' . L I T . U '
  5 . . . . - . . . . O P A . . .
  6 R " . . . " . G . C . . . " .
  7 I . ' . . C U R F E W . ' . .
  8 N . . ' . . . O . n . ' . . H
  9 K . ' . . . ' W ' E . . F . I
 10 s " . . . B O L T " . . I " R
 11 I . . . Q I . . A U R O R A S
 12 D E V E I N . M O N E Y S . E
 13 E . - . . . ' A ' . . . T . L
 14 . - . . . " . B . " . . . - E
 15 = . . ' . . . E . . . ' . . D
```

Rack **AAGINVX**, score 365–366, 10 in the bag. Unseen: ADDEEEGJMNNORSSTT.

| Move | Equity rank | Score | Leave | Win % |
|---|---|---|---|---|
| 15E GIV(E)N | #50 | 9 | AAX | 74.0 |
| 15E GIV(E) | #55 | 8 | AANX | 66.3 |
| I14 AX | #4 | 22 | AGINV | 54.8 |
| C12 (V)AX | #1 | 26 | AGINV | 51.7 |

GIVEN sets up a spot for the X: with the leave alone, 14E AX scores 53 next
turn. The X lands on the triple-letter square at F14 and counts in both AX
and XI, and the A makes AG. We hold the last X, and a play on E14 needs an A
there, AG being the only two-letter word that ends in G.

### OIL: a Y hook on the triple-word square

`1792940041679790342-ssh-0-g55-turn16`

```
    A B C D E F G H I J K L M N O
  1 C U P I D . . = . . . ' . . =
  2 O - R . . " . S . " . . . - .
  3 N . O . . . ' O X . A Y E . .
  4 T . W E . . . P I G G E D . '
  5 E . A V O U C H . . A W . . .
  6 M " R E . " . I . " . . . " .
  7 N . ' . . . M E ' . . . ' . .
  8 S . . ' . . I S . . . ' . . =
  9 . . ' . . . R . ' . . . ' . .
 10 . " . Q U A R T I L E . . " .
 11 . . . . - . o . . . - . . . .
 12 ' . . - . T R O U B L E D . Z
 13 . . - . . . S . ' . . H O V E
 14 . - . . . " . . . " . . . - I
 15 = . . ' . . . = . . . ' . . N
```

Rack **AILLNTY**, score 348–370, 17 in the bag. Unseen: ?AAAABDEEEFFGIIJKNNOORTT.

| Move | Equity rank | Score | Leave | Win % |
|---|---|---|---|---|
| H12 (O)IL | #123 | 5 | ALNTY | 44.1 |
| 11A LAYIN | #1 | 31 | LT | 36.5 |
| H12 (O)ILY | #3 | 23 | ALNT | 35.2 |

OIL takes an S or a Y behind it, on the triple-word square at H15, and the T
of QUARTILE above the O rules out a front hook. No S or Y is unseen, so the
one unseen blank is the only tile that can hit the square before our Y does.
With the leave alone, 15F LAY scores 39. OILY now, for 23, wins 35.2%.

### AD: two hook squares on the triple-word row

`1792940041679790342-ssh-0-g81-turn17`

```
    A B C D E F G H I J K L M N O
  1 = . . ' . . . = . . . T . . =
  2 . - . . . G R I G " . R . - .
  3 . . - . . . ' T I . . I - . .
  4 ' . D J E M B E S . . L . . '
  5 . C O O N . . r . . - l . . .
  6 . " . . . S L A V E R I E S .
  7 . . ' . . . U N ' . . O ' . .
  8 = . . ' . . A T . . . N . . =
  9 . . ' . . . U . ' . . . ' . .
 10 . F O R U M S . . " . . . " .
 11 . . P E H . . . . . - . . . .
 12 ' . . L . . . V O C A B . . '
 13 F A Q I R . Z A ' . . . - . .
 14 . - . N E G A T I O N . . - .
 15 = . . K . . . = . . A X I L E
```

Rack **AADDHOY**, score 312–345, 11 in the bag. Unseen: DEEEEEIINOPRTTUWWY.

| Move | Equity rank | Score | Leave | Win % |
|---|---|---|---|---|
| 14A AD | #31 | 17 | ADHOY | 40.0 |
| 14A AH | #33 | 25 | ADDOY | 37.2 |
| 15A DHA(K) | #1 | 36 | ADOY | 33.6 |

AD narrows who can use the triple-word square at A15. A play on row 15 from
A15 now needs a letter under the A that extends FA and a letter under the D
that extends AD: D, O, S or Z. No S or Z is left in the game, and of the two
D's and two O's that are, we hold one of each. With the leave alone, 15A YO
or HO scores 46.

### PITCH: the last Y

`1792940041679790342-ssh-0-g111-turn16`

```
    A B C D E F G H I J K L M N O
  1 D . . ' . . . = . . D U C A L
  2 I - . . P O R t A G E . . - .
  3 T . E . . . ' . G . . . - . .
  4 Z I N E . . . Q I . . A . . '
  5 . . D . - . D U N L I N . . .
  6 . K O A L A . E . " . T . " .
  7 . . W . O B J E T . . I ' . .
  8 W A S H . Y O R E . . R . . =
  9 . . ' . V E E . ' . . I ' . .
 10 . " . . . " S P I V . O . " .
 11 . . . . - . . . . . - t . . .
 12 ' . . - . . . ' . . . - . . '
 13 . . - . . . ' . ' . . . - . .
 14 . - . . . " . . . " . . . - .
 15 = . . ' . . . = . . . ' . . =
```

Rack **ACHIMTY**, score 255–295, 22 in the bag. Unseen: AABEEEEFFGILMNNNOOORRRSSTTUUX.

| Move | Equity rank | Score | Leave | Win % |
|---|---|---|---|---|
| H10 (P)ITCH | #167 | 13 | AMY | 51.8 |
| H10 (P)ATCH | #309 | 13 | IMY | 49.3 |
| H10 (P)ATCHY | #2 | 51 | IM | 41.5 |
| H10 (P)ITCHY | #1 | 51 | AM | 40.2 |

PITCH takes only a Y, on the triple-word square at H15, and we hold the last
one: no Y and no blank is unseen. With the leave alone, 15F MAY or 15H YAM
scores 72. PITCHY now, for 51, wins 40.2%: the sim prefers 13 points and a
private hook to 51 points.

## Further positions

These readings were worked out by an agent from the sim reports, the lexicon
and Macondo's move generator. No expert has checked them.

### ROUCHE: both ends of a triple-triple

`1792939424901615146-ssh-0-g51-turn16`

```
    A B C D E F G H I J K L M N O
  1 = R . ' . . . = . . . ' . . =
  2 . E . . . " . . . " . F . - .
  3 . L - . . . R E A L T O R S .
  4 ' O . - . . . ' . . . Z . . '
  5 . A G . - . . J U N K Y . . .
  6 . D I N . " A E . " . . . " .
  7 . . F E R . M U ' . . . ' . .
  8 = . . M E . . X I . . ' . . =
  9 . . ' A L . ' . S . . . ' . .
 10 . " . T I " . . A D . . . " .
 11 . . . O E . . . T A - . . . .
 12 ' . . D R A W L I N g - . . '
 13 . . - E . . ' . N . . . - . .
 14 . - . . . " Q . E " . . . - .
 15 = . . B O V I D S . . ' . . =
```

Rack **CEHOOSU**, score 266–312, 22 in the bag. Unseen: ?AABCEEEGGHIIIINNOOPPRTTTUVWY.

| Move | Equity rank | Score | Leave | Win % |
|---|---|---|---|---|
| 1B (R)OUCHE | #83 | 12 | OS | 47.9 |
| 1B (R)OUCHES | #3 | 39 | O | 26.1 |
| 2L (F)OU | #1 | 18 | CEHOS | 22.6 |

ROUCHE takes only an S behind it, and ROUCHES takes a C or a G in front. Row 1
runs from the triple-word square at A1 to the one at H1, so ROUCHE without
the S leaves both ends open to a single play: a C on A1 and our S on H1 make
CROUCHES across both squares for 135 (GROUCHES for 126).

No S is unseen, so the opponent needs the blank to touch the row at all. We
need one of the C, the two G's or the blank, four of the 29 unseen tiles, in
five draws: about 55%. ROUCHES now, for 39, spends the S and hands the
opponent the C or G hook on A1.

### THAN: declining a 76-point bingo

`1792942025771220660-local-0-g37-turn12`

```
    A B C D E F G H I J K L M N O
  1 = . . ' . . . = . . . ' . . =
  2 . - . . . " . . . " . . . - .
  3 . . - . . . ' . ' . . . B . .
  4 ' . . - . . . ' . . . - U . '
  5 . . . . - . . . . . - . R . N
  6 . " . . . " . . . " . U N A U
  7 . . ' . . . R O M . . . I . R
  8 = . . ' . . . I O D I D E . L
  9 . . ' . . D U L C E . E ' . .
 10 . " . . . " . . . W I L T " .
 11 . . . . - . . . . . - A . . .
 12 ' . . - . . . ' . . . I . . '
 13 . . - . . . ' . ' . . N - . .
 14 B R E V E T . . . " . E . - .
 15 Y E W ' D I A R I S T S . . =
```

Rack **AHINT??**, score 143–230, 35 in the bag. Unseen: AAAAACEEEEEFFGGGHIJKLMNNOOOOOOPPQRSSTTVXYZ.

| Move | Equity rank | Score | Leave | Win % |
|---|---|---|---|---|
| L1 THAN | #32 | 24 | I?? | 38.4 |
| 13H THeA(N)INe | #1 | 76 | – | 26.4 |
| 13B AH | #9 | 31 | INT?? | 26.1 |
| M13 AH | #68 | 19 | INT?? | 18.3 |

Down 87 with both blanks. THeANINe for 76 still leaves us 11 behind, with a
fresh rack and both blanks spent: 26.4%. THAN puts a T on L1, between the
triple-word squares at H1 and O1, and keeps I??. An eight-letter word from H1
to O1 through that T is tripled twice, and with two blanks almost any four
tiles make one. Four hand-picked draws gave AERaTIOn for 104 (drawing AEOR),
EGesTION for 113 (EGNO), KIlOTOnS for 140 (OOKS) and FIrETrAP for 149 (AEFP).

The comparison with AH shows that the lane is the point: 13B AH keeps INT??,
a stronger leave than I??, and wins only as often as the bingo; M13 AH, the
same leave for fewer points, wins 18.3%. The opponent holds no blank, so they
rarely bingo on row 1 themselves, but any ordinary play there spoils the
lane. The rollout opponent does that only by accident.

### TORI: a square where the Q counts twice

`1792942635811056326-local-0-g133-turn17`

```
    A B C D E F G H I J K L M N O
  1 = . . ' . . . N . . . ' . . =
  2 . - . . . " . I N F O L D S .
  3 . . - . . . ' P ' . . . - . .
  4 ' . . - . . . A B . C L E W '
  5 . . . . - . . . R O E . . . .
  6 . " . . . " . K A F S . . " .
  7 . J ' . . G ' O I . U . ' . .
  8 S I M O N Y . I N . R E A V E
  9 . R ' V . o P S I M A T H . X
 10 . D . E . Z . . N " E . . " p
 11 . . . R - A . . G . - . . . L
 12 ' . U H . . . ' . . . - . . O
 13 . . L A . . ' . ' . . . - . D
 14 . - U T . " . . . " . . . - E
 15 = . . E . . . = . . . ' . . R
```

Rack **CIIOQRT**, score 311–330, 14 in the bag. Unseen: AAABDEEEEGIINOOTTTUWY.

| Move | Equity rank | Score | Leave | Win % |
|---|---|---|---|---|
| G11 TORI | #31 | 7 | CIQ | 47.7 |
| 15D (E)ROTIC | #1 | 24 | IQ | 32.5 |
| exchange IQ | #5 | 0 | CIORT | 30.1 |
| G11 TORIC | #10 | 10 | IQ | 21.5 |

The Q has no play worth making: the best Q move by equity is the exchange of
IQ. TORI's last letter, the I on G14, changes that. F14 is a triple-letter
square, and QI played down from F14 also makes QI across with TORI's I: 62
points for two tiles, both in the CIQ we keep. The Q is ours, so the opponent
can block the square but not use it.

TORI also takes a C or an I on G15, beside the triple-word square at H15, and
we hold one of each. TORIC, which spends the C now, wins 21.5%.

### VIG: the same Q setup, and why the G matters

`1792939424901615146-ssh-0-g44-turn15`

```
    A B C D E F G H I J K L M N O
  1 = . . ' . . . = . . . ' . Z A
  2 . - . . . " . . . " . . . E L
  3 . . - . . . ' . ' . . . - N A
  4 ' . . - . . . ' . . . - V . M
  5 . . . . - . . . . . - . I . O
  6 . " . . . " . . . " . D E B S
  7 . . ' . . . ' . ' . . E W E .
  8 = . . ' . . . A W A I T . R =
  9 . . ' . . . ' U ' P . A ' R .
 10 . " . . . J U T . E . I . Y .
 11 V . . . M O N O . R - N . . .
 12 A . . K O B . l . I . E . . '
 13 R U T I N S ' Y ' E . R - . .
 14 I - . . . " . S . N . . . - .
 15 X . . ' . . . E . T O G G L E
```

Rack **ADGIQT?**, score 278–298, 19 in the bag. Unseen: AACCDDEEEFFHHIILLNOOOPRSTU.

| Move | Equity rank | Score | Leave | Win % |
|---|---|---|---|---|
| 11A (V)IG | #12 | 7 | ADQT? | 64.2 |
| 11A (V)ID | #72 | 7 | AGQT? | 48.5 |
| B12 G(U)T | #1 | 15 | ADIQ? | 48.3 |
| exchange GQ | #3 | 0 | ADIT? | 40.9 |

VIG sets up 10B QAT for 66: the Q on the triple-letter square at B10 counts
in QAT and in QI with VIG's I, and the A makes AG. The leave ADQT? holds all
three tiles.

VID sets up the same QAT, for the same score, and keeps a G instead of a D,
yet wins 16 points less often. The difference is the square above the last
letter: over a D it takes A, E, I or O (AD, ED, ID, OD), over a G only an A.
The opponent has fewer plays that land there. Over 150 random opponent racks,
Macondo's top static reply left QAT playable 102 times after VIG and 51 times
after VID. A setup's value depends on how hard it is to stumble into.

### CHORD: a pass that keeps the bag closed

`1792941188977491622-local-0-g53-turn21`

```
    A B C D E F G H I J K L M N O
  1 A . . ' . P I X . . . I K A T
  2 J A W . Q I . I N R U N . - .
  3 E . O P I N E . ' . . . - . .
  4 E . . A . . . C . . . - . . '
  5 . T I N Y . . U . . - . . . .
  6 . " . F A V E R . " . . . " .
  7 . . B U M . ' T ' . . . ' . .
  8 F E E L . . . A . . . ' . . =
  9 A . ' . D . ' T ' . . . ' . .
 10 I " G L E G N e S S . . . " .
 11 R . . O W . . . . . - . . . .
 12 I . . V . . . B L O O D . . '
 13 E . Z E U G M A ' . . . - . .
 14 S - . L . " . . . " . . . - .
 15 T . H Y E N O I D . . ' . . =
```

Rack **DHOORRS**, score 380–325, 1 in the bag. Unseen: ?ACEEORT.

| Move | Equity rank | Score | Leave | Win % |
|---|---|---|---|---|
| pass | – | 0 | DHOORRS | 75.8 |
| 13K HOODS | #42 | 26 | RR | 12.6 |
| 4H (C)HORD | #1 | 22 | ORS | 0.0 |
| 11K HOOD | #2 | 29 | RRS | 0.0 |

We lead by 55, one tile is in the bag, and the opponent holds seven of
?ACEEORT. Whatever we play, we draw the last tile and the opponent bingos
out: for every rack they can hold, Macondo finds a bingo for 74 to 86 on row
14 or down column O, and twice our leftover rack on top of it beats the 77 we
would lead by after CHORD. All ten top moves win 0.0%.

A pass keeps the tile in the bag. The opponent's bingo then no longer ends
the game: they draw the last tile, we answer with 31 to 46 points from a full
rack, and they are left with one tile. This pass does not lean on the
rollout opponent's habits: passing back loses, because six scoreless turns
end the game with us ahead.

### Q(I): play one tile, leave one in the bag

`1792943897352376103-local-0-g167-turn23`

```
    A B C D E F G H I J K L M N O
  1 = J . ' . . . C . . . K E P I
  2 . U . . . " . H A N D A X E S
  3 . R - . . . V I G . . . - . .
  4 W A M E . R . V . . . Y . . '
  5 . . E N T I R E S . B I . . .
  6 . " . . . N O . . Y A P . " .
  7 . . ' . . D O W ' O N . ' . D
  8 = . . ' . . F U N D I ' . L A
  9 . . ' . . . ' Z A S . . ' O F
 10 . " . . . " . . G " . . . U T
 11 . . . . - . . L A T T I N S .
 12 ' . . - . . . O . . . - . I '
 13 . . - . . . ' U ' . . . B E T
 14 . - . . . E R G O M E t E R .
 15 = . . ' . . . H . . . ' . . =
```

Rack **AEIILOQ**, score 424–358, 2 in the bag. Unseen: ?ACEELORT.

| Move | Equity rank | Score | Leave | Win % |
|---|---|---|---|---|
| L10 Q(I) | #75 | 11 | AEIILO | 97.5 |
| 15H (H)I | #53 | 7 | AEILOQ | 91.2 |
| 1B (J)O | #32 | 9 | AEIILQ | 58.4 |
| 3A O(R)E | #3 | 21 | AIILQ | 23.9 |
| 1A A(J)I | #1 | 30 | EILOQ | 20.5 |

The same danger with two tiles in the bag: we lead by 66 and the opponent
holds seven of ?ACEELORT. A play of two or more tiles empties the bag and
lets a bingo end the game; a play of one tile does not. After O(R)E, which
draws both tiles, the opponent bingos out at once in 19 of the 36 equally
likely deals (CLEAROUT and others, for 68 to 82), and in the rest scores 33
to 54, mostly at A1, while we hold the Q.

Every one-tile play in the report wins 58 to 98%, and every longer play wins
at most 24%. Of the one-tile plays, QI is best because it also unloads the
Q.

### INDIGENE: a pass that works only against HastyBot

`1792940705497667449-ssh-0-g186-turn28`

```
    A B C D E F G H I J K L M N O
  1 = . . ' . . . = . . . ' . . =
  2 . - . . . " . . . " . . . - .
  3 . . - . . . ' . ' . . . - . .
  4 ' . . - . . . ' . . . - . . '
  5 . . . . - . . . . J U C O . .
  6 . " . . . V A T . " . O N E .
  7 . . ' . . . B O X . W O O F s
  8 = . . ' . . . P U L I K . . H
  9 . . ' . . . M E ' . M I ' T A
 10 . " . . . F A . . " P E T E R
 11 . . . . V I R G E R S . . S I
 12 Z E E - . G I N . . . - . T A
 13 . . W U D . A A ' . Q U B I T
 14 O Y . . O Y . R E " . . A E .
 15 H A N D R a I L S . . C A R T
```

Rack **DEGIINN**, score 345–408, 1 in the bag. Unseen: DEELLNOS.

| Move | Equity rank | Score | Leave | Win % |
|---|---|---|---|---|
| pass | – | 0 | DEGIINN | 100.0 |
| K13 (Q)I | #2 | 11 | DEGINN | 21.9 |
| B5 INDIGEN(E) | #1 | 64 | – | 0.0 |

Down 63 with INDIGENE for 64 in hand and one tile in the bag. Playing it
puts us one point ahead holding only the tile we draw, against a full rack:
0.0%. The pass wins 100%, and the reason is the rollout opponent. For each
rack it can hold, its best static-equity move is a play for 5 to 11 points
(LEV, LAB, JO, LOVAT) that keeps its better tiles and draws the last tile.
INDIGENE then goes out, collects twice the opponent's rack, and wins by 6 to
12.

An opponent that looks one move ahead passes back. Six scoreless turns end
the game with its 63-point lead intact, so we must bingo into the open bag
after all. The mirror image, [the pass when ahead](#chord-a-pass-that-keeps-the-bag-closed),
has no such refutation.

This is the entry to remember when reading the set's passes. The pass is the
best move in 25 kept positions, and in 22 of them the passer is behind; two
more with a bingo in hand are `1792945961871666323-ssh-0-g28-turn22`
(BIRCHEN) and `1792942687894377692-local-0-g13-turn18` (MEDIATE). A sim whose
opponent never passes and does not play to the score overvalues these.

### Exchange seven, with seven in the bag

`1792940249407138500-local-0-g161-turn17`

```
    A B C D E F G H I J K L M N O
  1 = . . ' . . W I N D I L Y . =
  2 J A d E I T E . . E . . . - .
  3 . . - . . . ' . ' N . . P . .
  4 Z E T A . . . R . A . - O . '
  5 E . . S E A B O O T S . L . .
  6 U " . . . Y O U . U . . E " .
  7 G . ' . . . ' E ' R . . W . .
  8 M . . R . . . N . E N H A L O
  9 A . K I D d I S H . . . R . .
 10 . G I F . " . . O X . . D " .
 11 . . . E - . C U M I N . . . .
 12 ' . . R . . . V . . . - . . '
 13 . . - . . . ' E ' . . . - . .
 14 . - . . . " . A . " . . . - .
 15 = . . ' . . . L . . . ' . . =
```

Rack **ADFGQTV**, score 383–395, 7 in the bag. Unseen: ABCEIINOOPRSTT.

| Move | Equity rank | Score | Leave | Win % |
|---|---|---|---|---|
| exchange ADFGQTV | #20 | 0 | – | 35.2 |
| exchange AFGQTV | #19 | 0 | D | 30.3 |
| 13G V(E)G | #7 | 13 | ADFQT | 8.8 |
| 11C D(E)V | #1 | 22 | AFGQT | 7.2 |

Down 12 with a Q and no good way to play it: no U is on our rack or unseen,
and the Q's only plays are 14G Q(A)T for 12 and E1 Q(I) for 11, which
Macondo's static equity ranks 52nd and 132nd. Every top move wins under 9%.

Seven tiles in the bag is the last moment an exchange is legal. Exchanging
all seven takes the whole bag, seven of ABCEIINOOPRSTT, and puts ADFGQTV in
its place, so every tile the opponent draws from here on is one we threw
back, the Q among them.

The same move from ahead: `1792940883968980450-local-0-g66-turn19` (seven in
the bag, exchange ENNOOQ keeping the Z, 79.9% against 65.6% for OZONES) and
`1792944003221424672-local-0-g130-turn19` (nine in the bag, exchange CFHMTV
keeping the X, 79.1% against 63.3%).

### ACDEINT: passing the opening move

`1792943088404665925-local-0-g129-turn1`

```
    A B C D E F G H I J K L M N O
  1 = . . ' . . . = . . . ' . . =
  2 . - . . . " . . . " . . . - .
  3 . . - . . . ' . ' . . . - . .
  4 ' . . - . . . ' . . . - . . '
  5 . . . . - . . . . . - . . . .
  6 . " . . . " . . . " . . . " .
  7 . . ' . . . ' . ' . . . ' . .
  8 = . . ' . . . - . . . ' . . =
  9 . . ' . . . ' . ' . . . ' . .
 10 . " . . . " . . . " . . . " .
 11 . . . . - . . . . . - . . . .
 12 ' . . - . . . ' . . . - . . '
 13 . . - . . . ' . ' . . . - . .
 14 . - . . . " . . . " . . . - .
 15 = . . ' . . . = . . . ' . . =
```

Rack **ACDEINT**, score 0–0, 86 in the bag. Unseen: ??AAAAAAAABBCDDDEEEEEEEEEEEFFGGGHHIIIIIIIIJKLLLLMMNNNNNOOOOOOOOPPQRRRRRRSSSSTTTTTUUUUVVWWXYYZ.

| Move | Equity rank | Score | Leave | Win % |
|---|---|---|---|---|
| pass | – | 0 | ACDEINT | 58.3 |
| exchange D | #41 | 0 | ACEINT | 55.2 |
| H4 CNIDA | #1 | 22 | ET | 53.2 |
| 8H ANTIC | #3 | 20 | DE | 53.0 |

ACDEINT makes no seven-letter word, but it makes an eight through eleven
different letters: C, I, K, M, N, O, P, R, S, T and U (ACCIDENT, INDICATE,
ANTICKED, MEDICANT, INCANTED, CATENOID, PEDANTIC, DICENTRA, DISTANCE,
NICTATED, INCUDATE). The pass keeps the rack whole and waits for the
opponent's opening play to supply the eighth letter. It beats exchanging the
D as well as every play.

This is the least certain entry. The gain is five points, the sim this early
is two replies and then the leaf network, and the rollout opponent always
opens the board. `1792944955544870588-local-0-g194-turn1` is the same pass
with ADINNOT (55.2% against 49.2% for DONNA).

## More of the same patterns

Positions that repeat a pattern above, each checked the same way but not
written up:

- `1792943844829405245-ssh-0-g10-turn19`: 8J DU(OM)O, 8 points, keeps AFSW.
  DUOMO takes only an S, on the triple-word square at O8, and no S or blank is
  unseen. 73.8% against 55.3% for 15H WO.
- `1792939586653428488-local-0-g185-turn15`: 14H GRACE, 19 points, keeps FZ.
  The E on L14 opens row 15 from L15 to the triple-word square at O15; FAZE
  there scores 69 if we draw the vowels. 55.6% against 39.0% for 14I FAG.
