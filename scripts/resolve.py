# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                                                             #
#   Ethereal is a UCI chess playing engine authored by Andrew Grant.          #
#   <https://github.com/AndyGrant/Ethereal>     <andrew@grantnet.us>          #
#                                                                             #
#   Ethereal is free software: you can redistribute it and/or modify          #
#   it under the terms of the GNU General Public License as published by      #
#   the Free Software Foundation, either version 3 of the License, or         #
#   (at your option) any later version.                                       #
#                                                                             #
#   Ethereal is distributed in the hope that it will be useful,               #
#   but WITHOUT ANY WARRANTY; without even the implied warranty of            #
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the             #
#   GNU General Public License for more details.                              #
#                                                                             #
#   You should have received a copy of the GNU General Public License         #
#   along with this program.  If not, see <http://www.gnu.org/licenses/>.     #
#                                                                             #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import sys
import chess

from processpgn import *

try:
    for pgn in iter(yield_pgns(sys.argv[1])):

        result, data = process_pgn(pgn, EvalCollector)

        for fen, move, score in data:

            # Normalize the Score if possible
            try: cp = [float(score), -float(score)][' b ' in fen]
            except: break

            # Resolve the line of play
            board = chess.Board(fen, chess960=False)
            board.push(chess.Move.from_uci(move))

            # Save the final [Fen, Result, Eval]
            print ('{} [{}] {}'.format(board.fen(), result, int(cp * 100)))

except StopIteration:
    pass

