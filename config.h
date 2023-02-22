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
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

#include <stdbool.h>
#include <stdint.h>

#include "activate.h"
#include "types.h"

#define BATCHSIZE 16384
#define LEARNRATE 0.001

#define BETA_1 0.9
#define BETA_2 0.999

static const uint64_t NSAMPLES    = 1024 * 1024 * 1ULL; // 128ULL;
static const char     DATAFILE[]  = "s44.training.nndata.%d";
static const int      NDATAFILES  = 1;

static const uint64_t NVALIDATE   = 1024 * 1024 * 1ULL; // 32ULL;
static const char     VALIDFILE[] = "s44.validation.nndata";

static const int      START_EPOCH = 0;

static const bool     USE_WEIGHTS = false;
static const char     NNWEIGHTS[] = "";

static const bool     USE_STATE   = false;
static const char     NNSTATE[]   = "";

static const float    SIGM_COEFF  = 2.315 / 400.00;

// Choose a Loss, LossProp, and NN Architecture

#define LOSS_FUNC     l2_one_neuron_loss
#define LOSSPROP_FUNC l2_one_neuron_lossprop
