import chess
import random

from subprocess import Popen, PIPE
from multiprocessing import Process, Queue

ENGINE  = './Ethereal'
RANDOMS = 10
DEPTH   = 12
CUTOFF  = 1000
THREADS = 32

class Engine():

    def __init__(self, ename):
        self.engine = Popen([ename], stdin=PIPE, stdout=PIPE, universal_newlines=True, shell=True)
        self.uci_ready()
        self.uci_set_frc()

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

with open('frcpositions.epd') as fin:
    FENS = [x.strip() for x in fin.readlines()]

def random_position():
    pos = chess.Board(random.choice(FENS), chess960=True)
    for ii in range(RANDOMS):
        moves = list(pos.legal_moves)
        if not moves: return random_position()
        pos.push(random.choice(moves))
    return pos.fen()

def thread_build_book(outqueue):

    engine = Engine(ENGINE)

    while True:
        fen = random_position()
        seval = engine.uci_depthn_eval(fen, DEPTH)
        if seval != None and abs(seval) < CUTOFF:
            outqueue.put(fen)

if __name__ == '__main__':

    outqueue = Queue()

    workers = [
        Process(target=thread_build_book, args=(outqueue,))
        for ii in range(THREADS)]
    for worker in workers: worker.start()

    while True:
        print(outqueue.get())