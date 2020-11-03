import platform, time, sys, os
from subprocess import Popen, PIPE, call
from multiprocessing import Process, Queue, active_children

DEPTH   = 12
THREADS = 32
BATCH   = 8192
ENGINE  = './Ethereal'
SOURCE  = 'NeuralBoth.nnbook'

def parse_source_fens(fname):
    with open(fname) as fin:
        for line in fin:
            yield process_source_fen(line)

def process_source_fen(line):
    fen    = line.split('[')[0].rstrip()
    result = line.split('[')[1].split(']')[0]
    return (fen, result)

class Engine():

    def __init__(self, ename):
        self.engine = Popen([ename], stdin=PIPE, stdout=PIPE, universal_newlines=True, shell=True)
        self.uci_ready()

    def write_line(self, line):
        self.engine.stdin.write(line)
        self.engine.stdin.flush()

    def read_line(self):
        return self.engine.stdout.readline().rstrip()

    def uci_ready(self):
        self.write_line('isready\n')
        while self.read_line() != 'readyok': pass

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

def thread_process_data(inqueue, outqueue):

    engine = Engine(ENGINE)

    while True:
        fen, result = inqueue.get()
        if fen == None or result == None:
            return

        seval = engine.uci_depthn_eval(fen, DEPTH)
        if seval != None: outqueue.put('%s [%s] %d' % (fen, result, seval))
        else: outqueue.put('')

def enqueue_elements(generator, elements, queue):
    for ii in range(elements):
        try: queue.put(next(generator))
        except StopIteration: return ii
    return elements

if __name__ == '__main__':

    inqueue = Queue(); outqueue = Queue()

    workers = [
        Process(target=thread_process_data, args=(inqueue, outqueue,), daemon=True)
        for ii in range(THREADS)]
    for worker in workers: worker.start()

    generator = parse_source_fens(SOURCE)

    try:
        while True:
            placed = enqueue_elements(generator, BATCH, inqueue)
            for ii in range(placed):
                out = outqueue.get()
                if out != '': print (out)
            if placed != BATCH: break
    except:
        for worker in workers:
            worker.join()