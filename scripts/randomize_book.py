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

import chess
import argparse

from random import choice, randrange
from subprocess import Popen, PIPE
from multiprocessing import Process, Queue

class Engine():

    def __init__(self, ename, fischer):
        self.engine = Popen([ename], stdin=PIPE, stdout=PIPE, universal_newlines=True, shell=True)
        self.uci_ready()
        if fischer: self.uci_set_frc()

    def write_line(self, line):
        self.engine.stdin.write(line)
        self.engine.stdin.flush()

    def read_line(self):
        return self.engine.stdout.readline().rstrip()

    def uci_ready(self):
        self.write_line('isready\n')
        while self.read_line() != 'readyok': pass

    def uci_set_frc(self):
        self.write_line('setoption name UCI_Chess960 value true\n')
        self.uci_ready()

    def uci_search(self, fen, depth):
        self.uci_ready()
        self.write_line('position fen %s\ngo depth %d\n' % (fen, depth))
        return list(self.uci_bestmove())

    def uci_bestmove(self):
        while True:
            line = self.read_line()
            if line.startswith('bestmove'): break
            yield line

    def uci_depthn_eval(self, fen, depth):
        output = self.uci_search(fen, depth)[-1]
        if ' cp ' not in output: return None
        relative = int(output.split(' cp ')[1].split()[0])
        return [relative, -relative][' b ' in fen]

def random_position(arguments):

    if arguments.fischer:
        pos = chess.Board.from_chess960_pos(randrange(960))

    elif arguments.dfischer:

        # As a proxy for randomization of starting position
        p1 = chess.Board.from_chess960_pos(randrange(960)).shredder_fen()
        p2 = chess.Board.from_chess960_pos(randrange(960)).shredder_fen()

        # Grab each sides random arrangment of pieces
        white_pieces = p1.split()[0].split('/')[7]
        black_pieces = p2.split()[0].split('/')[0]

        # Grab the file markers for the castling rights
        white_castle = p1.split()[2][0:2]
        black_castle = p2.split()[2][2:4]

        # Merge the two FENs
        fstring = '%s/pppppppp/8/8/8/8/PPPPPPPP/%s w %s%s - 0 1'
        fen = fstring % (black_pieces, white_pieces, white_castle, black_castle)
        pos = chess.Board(fen, chess960=True)

    else:
        pos = chess.Board('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1')

    for ii in range(int(arguments.plies)):
        moves = list(pos.legal_moves)
        if not moves: return random_position(arguments)
        pos.push(choice(moves))

    if pos.legal_moves:
        if arguments.fischer or arguments.dfischer:
            return pos.shredder_fen()
        return pos.fen()

    return random_position(arguments)

def thread_build_book(outqueue, arguments):

    engine = Engine(arguments.engine, arguments.fischer or arguments.dfischer)

    while True:
        fen = random_position(arguments)
        seval = engine.uci_depthn_eval(fen, int(arguments.depth))
        if seval != None and abs(seval) < int(arguments.cutoff):
            outqueue.put(fen)

if __name__ == '__main__':

    p = argparse.ArgumentParser()
    p.add_argument('--engine'  , help='Binary File'   , required=True)
    p.add_argument('--threads' , help='Worker Threads', required=True)
    p.add_argument('--samples' , help='Openings Count', required=True)
    p.add_argument('--depth'   , help='Search Depth'  , default=10)
    p.add_argument('--plies'   , help='Random Length' , default=10)
    p.add_argument('--cutoff'  , help='Search Margin' , default=600)
    p.add_argument('--fischer' , help='FRC/960'       , action='store_true')
    p.add_argument('--dfischer', help='DFRC/D960'     , action='store_true')
    arguments = p.parse_args()

    outqueue = Queue()

    workers = [
        Process(
            target=thread_build_book,
            args=(outqueue, arguments,),
            daemon=True,
        ) for ii in range(int(arguments.threads))
    ]

    for worker in workers:
        worker.start()

    for ii in range(int(arguments.samples)):
        print(outqueue.get())

    for worker in workers:
        worker.terminate()
        worker.join()