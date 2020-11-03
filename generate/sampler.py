import time, sys, os, io, random, chess, chess.pgn
from subprocess import Popen, PIPE
from multiprocessing import Process, Queue

THREADS  = 32
BATCH    = 16384
FENS_PER = 10

SOURCE = 'Fishtest.pgn'
OUTPUT = 'Fishtest.fens'

class Visitor(chess.pgn.BaseVisitor):

    def begin_game(self):
        self.fens = []

    def visit_move(self, board, move):
        self.fens.append(board.fen())

    def result(self):
        return self.fens

def parse_source_pgns(fname):

    with open(fname) as fin:
        builder = ''
        for line in fin:
            if line.startswith('[Result'):
                if builder: yield builder
                builder = ''
            builder += line

def thread_process_data(inqueue, outqueue):

    ResultDict = { "1-0" : "[1.0]", "0-1" : "[0.0]", "1/2-1/2" : "[0.5]" }

    while True:

        pgn = inqueue.get()
        if pgn == None: return

        try:
            game     = chess.pgn.read_game(io.StringIO(pgn))
            fens     = game.accept(Visitor())
            sampled  = random.sample(fens, min(len(fens), FENS_PER))
            result   = ResultDict[game.headers["Result"]]
            finished = [fen + " " + result for fen in sampled]
            outqueue.put(finished)
        except:
            outqueue.put([])

def enqueue_elements(generator, elements, queue):

    for ii in range(elements):
        try: queue.put(next(generator))
        except StopIteration: return ii

    return elements

if __name__ == '__main__':

    inqueue = Queue(); outqueue = Queue()

    workers = [
        Process(target=thread_process_data, args=(inqueue, outqueue,))
        for ii in range(THREADS)]
    for worker in workers: worker.start()

    processed = 0
    generator = parse_source_pgns(SOURCE)

    with open(OUTPUT, 'w') as fout:

        while True:
            placed = enqueue_elements(generator, BATCH, inqueue)
            for ii in range(placed):
                for fen in outqueue.get():
                    fout.write(fen + '\n')

            processed += placed
            print('Processed {} Games'.format(processed))
            if placed != BATCH: break

