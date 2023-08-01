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

# Strips a PGN all the way down to only two headers, and minimal data to convey
# the evaluation and moves made at each turn. Below is an example, with added
# newlines. The output is no longer a PGN, and cannot be used as such. Extra
# headers could be included without causing issues in Ethereal's processor
#
#   [Result "0-1"]
#   [FEN "rnbqkbnr/1p1p2pp/5p2/p1p1p3/1P4P1/4PP2/P1PP1K1P/RNBQ1BNR w kq - 0 1"]
#
#   bxc5 -0.42 Bxc5 0.10 d4 -0.13 Bb6 -0.11 c4 0.14 d5 -0.14 c5 -0.09 Bc7 -0.02
#   Nc3 -0.09 Ne7 0.22 h4 -0.15 O-O 0.20 Nb5 -0.28 b6 0.25 Ba3 -0.44 bxc5 0.33
#   Bxc5 -0.36 Bb6 0.69 Rc1 -0.84 Nbc6 0.87 Ne2 -0.78 f5 0.65 gxf5 -0.62 Bxc5
#   0.96 Rxc5 -0.48 Qb6 0.67 Nec3 -0.60 exd4 0.71 exd4 -0.82 Bxf5 1.22 Qd2
#   -0.98 Rad8 1.52 h5 -1.20 h6 1.19 Na4 -1.42 Qb8 1.37 Rc3 -1.85 Bh7 2.98 Rc5
#   -2.78 Rxf3+ 4.76 Kxf3 -5.61 Be4+ 1.93 Kg4 -6.63 Bxh1 4.06 Bd3 -7.33 Rf8
#   6.40 Nac3 -7.73 Rf3 8.97 Ne2 -3.29 Qc8+ M5 Bf5 -M4 0-1

import sys

INPUT   = sys.argv[1]
HEADERS = ['FEN', 'Result', 'White', 'Black']

def parseGames(fin):

    segment = []; headers = True

    while True:

        line = fin.readline()
        if not line: break

        if line.strip() == "":
            if not headers:
                yield segment
                segment = []
            else:
                segment.append(line)
            headers = not headers

        else:
            segment.append(line)

def stripGame(game):

    headers = []

    for line in game:
        for header in HEADERS:
            if line.startswith("[{0} ".format(header)):
                headers.append(line)

    moves = "".join(game).split("\n\n")[1]
    moves = moves.replace("\n", " ")
    moves = stripComments(moves)
    moves = [x for x in moves.split() if x[-1] != '.']
    moves = [x.lstrip('+') for x in moves]
    moves = ' '.join(moves)

    return "".join(headers) + "\n" + moves + "\n\n"

def stripComments(line):

    rebuilt = ''; ii = 0

    while ii < len(line):

        if line[ii] == '{':

            ii = ii + 1

            while line[ii] != '/':
                rebuilt += line[ii];
                ii = ii + 1

            while True:
                if line[ii] == '}':
                    break
                ii = ii + 1
        else:
            rebuilt += line[ii]

        ii = ii + 1

    return rebuilt.replace("  ", " ")

with open(INPUT, 'r') as fin:
    for game in parseGames(fin):
        print(stripGame(game), end='')
