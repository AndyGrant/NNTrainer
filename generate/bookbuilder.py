
import io
import chess
import chess.pgn
import multiprocessing

THREADS   = 16
DATAFILE  = 'frcgames.pgn'
BATCHSIZE = 8192

class Visitor(chess.pgn.BaseVisitor):

    def begin_game(self):
        self.fens = []
        self.comments = []

    def visit_move(self, board, move):
        self.fens.append(board.fen() + " " + str(move))

    def visit_comment(self, comment):
        self.comments.append(comment)

    def result(self):
        return (self.fens, self.comments)

def parse_each_pgn():
    with open(DATAFILE) as pgn:
        lines = []
        for line in pgn:
            if line.startswith('[Event') and lines:
                yield ''.join(lines)
                lines = []
            lines.append(line)

def process_data(inqueue, outqueue):

    while True:

        pgn = inqueue.get()
        if pgn == None: return

        game = chess.pgn.read_game(io.StringIO(pgn))
        fens, comments = game.accept(Visitor())
        samples = []

        result = game.headers['Result']
        if result.startswith('1-0') : result = '1.0'
        if result.startswith('1/2') : result = '0.5'
        if result.startswith('0-1') : result = '0.0'

        for fen, comment in zip(fens[:-1], comments[:-1]):

            ev = comment.split('/')[0]
            if 'M' in ev.upper(): break

            ev = int(float(ev) * 100.0)
            samples.append('{} [{}] {}'.format(fen, result, ev))

        outqueue.put(samples)

def enqueue_elements(generator, elements, queue):
    for ii in range(elements):
        try: queue.put(next(generator))
        except StopIteration: return ii
    return elements

def build_book():

    inqueue = multiprocessing.Queue()
    outqueue = multiprocessing.Queue()

    workers = [
        multiprocessing.Process(target=process_data, args=(inqueue, outqueue,))
        for ii in range(THREADS)]
    for worker in workers: worker.start()

    generator = parse_each_pgn()

    while True:
        placed = enqueue_elements(generator, BATCHSIZE, inqueue)

        for ii in range(placed):
            for sample in outqueue.get():
                print(sample)

        if placed != BATCHSIZE:
            break

    for ii in range(THREADS):
        inqueue.put(None)

    for worker in workers:
        worker.join()

if __name__ == '__main__':
    build_book()