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

import argparse

p = argparse.ArgumentParser()
p.add_argument('-B', '--book',   help='Book',   required=True)
p.add_argument('-O', '--output', help='Output', required=True)
p.add_argument('-T', '--chunks', help='Chunks', required=True)
arguments = p.parse_args()

pgnfile = str(arguments.book)
output  = str(arguments.output)
chunks  = int(arguments.chunks)

files = [open("{}.{}".format(output, f), "w") for f in range(chunks)]

with open(pgnfile, "r") as fin:

    lines = []
    cycle = count = 0

    while True:

        line = fin.readline()
        if not line: break

        lines.append(line)
        if line.strip() == "": count += 1
        if count != 2: continue

        for line in lines:
            files[cycle].write(line)

        lines = []; count = 0
        cycle = (cycle + 1) % chunks

for file in files: file.close()
