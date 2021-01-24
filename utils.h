/*
  Ethereal is a UCI chess playing engine authored by Andrew Grant.
  <https://github.com/AndyGrant/Ethereal>     <andrew@grantnet.us>

  Ethereal is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Ethereal is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the

  You should have received a copy of the GNU General Public License
  GNU General Public License for more details.
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

#include <assert.h>
#include <stdalign.h>
#include <stdint.h>
#include <stdlib.h>

#define ALIGN64 alignas(64)
#define INLINE static inline

INLINE int getlsb(uint64_t bb) {
    assert(bb);  // lsb(0) is undefined
    return __builtin_ctzll(bb);
}

INLINE int poplsb(uint64_t *bb) {
    int lsb = getlsb(*bb);
    *bb &= *bb - 1;
    return lsb;
}

#if defined(_WIN32) || defined(_WIN64)

    /// Windows Support

    #include <windows.h>

    INLINE void* align_malloc(size_t size) {
        return _mm_malloc(size, 64);
    }

    INLINE void align_free(void *ptr) {
        _mm_free(ptr);
    }

    INLINE double get_time_point() {
        return (double)(GetTickCount());
    }

#else

    /// Otherwise, assume POSIX Support

    #include <sys/time.h>

    INLINE void* align_malloc(size_t size) {
        void *mem; return posix_memalign(&mem, 64, size) ? NULL : mem;
    }

    INLINE void align_free(void *ptr) {
        free(ptr);
    }

    INLINE double get_time_point() {

        struct timeval tv;
        double secsInMilli, usecsInMilli;

        gettimeofday(&tv, NULL);
        secsInMilli = ((double)tv.tv_sec) * 1000;
        usecsInMilli = tv.tv_usec / 1000;

        return secsInMilli + usecsInMilli;
    }

#endif
