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

import io
import chess
import chess.pgn

class FenCollector(chess.pgn.BaseVisitor):

    def begin_game(self):
        self.fens = []

    def visit_move(self, board, move):
        self.fens.append(board.fen())

    def result(self):
        return (self.fens)

class EvalCollector(chess.pgn.BaseVisitor):

    def begin_game(self):
        self.fens  = []
        self.moves = []
        self.evals = []

    def visit_move(self, board, move):
        self.fens.append(board.fen())
        self.moves.append(str(move))

    def visit_comment(self, comment):
        self.evals.append(comment.split('/')[0])

    def result(self):
        return list(zip(self.fens, self.moves, self.evals))

def process_pgn(pgn, visitor):

    # Read a string into a PGN, otherwise read an actual File
    if type(pgn) == str: game = chess.pgn.read_game(io.StringIO(pgn))
    else: game = chess.pgn.read_game(pgn)
    if game == None: return None

    # Normalize Results to [0.0, 0.5, 1.0]
    result = game.headers['Result']
    if result.startswith('1-0') : result = '1.0'
    if result.startswith('1/2') : result = '0.5'
    if result.startswith('0-1') : result = '0.0'

    # Return Result and requested collection
    return result, game.accept(visitor())

def yield_pgns(filename):
    with open(filename) as pgn:
        while True:
            game = chess.pgn.read_game(pgn)
            if game == None: break
            yield str(game)
