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

    if arguments.fischer: pos = chess.Board.from_chess960_pos(randrange(960))
    else: pos = chess.Board('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1')

    for ii in range(int(arguments.plies)):
        moves = list(pos.legal_moves)
        if not moves: return random_position()
        pos.push(choice(moves))

    if pos.legal_moves:
        if arguments.fischer: return pos.shredder_fen()
        else: return pos.fen()

    return random_position(arguments)

def thread_build_book(outqueue, arguments):

    engine = Engine(arguments.engine, arguments.fischer)

    while True:
        fen = random_position(arguments)
        seval = engine.uci_depthn_eval(fen, int(arguments.depth))
        if seval != None and abs(seval) < int(arguments.cutoff):
            outqueue.put(fen)

if __name__ == '__main__':

    p = argparse.ArgumentParser()
    p.add_argument('--engine' , help='Binary File'   , required=True)
    p.add_argument('--depth'  , help='Search Depth'  , required=True)
    p.add_argument('--cutoff' , help='Search Margin' , required=True)
    p.add_argument('--plies'  , help='Random Length' , required=True)
    p.add_argument('--threads', help='Worker Threads', required=True)
    p.add_argument('--fischer', help='FRC/960', action='store_true')
    arguments = p.parse_args()

    outqueue = Queue()

    workers = [
        Process(target=thread_build_book, args=(outqueue, arguments,))
        for ii in range(int(arguments.threads))]
    for worker in workers: worker.start()

    while True:
        print(outqueue.get())