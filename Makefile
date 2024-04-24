#  Ethereal is a UCI chess playing engine authored by Andrew Grant.
#  <https://github.com/AndyGrant/Ethereal>     <andrew@grantnet.us>
#
#  Ethereal is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  Ethereal is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.

CC	 = gcc
SRC	 = *.c
LIBS = -fopenmp -lm -lpthread

WFLAGS = -Wall -Wextra -Wshadow -std=gnu11 -Wno-unused-result
CFLAGS = -O3 -flto -ffast-math -mfma -march=native -DNDEBUG

NETDIR := $(shell mkdir -p Networks)

halfkp:
	$(CC) $(SRC) archs/halfkp.c    $(WFLAGS) $(CFLAGS) $(LIBS) -DMAX_INPUTS=43850 -DNN_TYPE=HALFKP

psqbb:
	$(CC) $(SRC) archs/psqbb.c     $(WFLAGS) $(CFLAGS) $(LIBS) -DMAX_INPUTS=768   -DNN_TYPE=PSQBB

mirrorhkp:
	$(CC) $(SRC) archs/mirrorhkp.c $(WFLAGS) $(CFLAGS) $(LIBS) -DMAX_INPUTS=21120 -DNN_TYPE=MIRRORHKP

mirrorhka:
	$(CC) $(SRC) archs/mirrorhka.c $(WFLAGS) $(CFLAGS) $(LIBS) -DMAX_INPUTS=25344 -DNN_TYPE=MIRRORHKA