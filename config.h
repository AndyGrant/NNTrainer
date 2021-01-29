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

#include "activate.h"
#include "types.h"

static const int   NSAMPLES    = 1024 * 1024 * 16;
static const char  DATAFILE[]  = "training.nndata";

static const int   NVALIDATE   = 1024 * 1024 * 16;
static const char  VALIDFILE[] = "validation.nndata";

static const int   MAX_INPUTS  = 43850;
static const int   BATCHSIZE   = 1024;
static const float LEARNRATE   = 0.001;

static const bool  USE_WEIGHTS = false;
static const char  NNWEIGHTS[] = "";

static const float SIGM_COEFF = 2.42 / 400.00;

static const Layer ARCHITECTURE[] = {
    {43850, 128, &activate_relu, &backprop_relu },
    {  256,  32, &activate_relu, &backprop_relu },
    {   32,  32, &activate_relu, &backprop_relu },
    {   32,   1, &activate_null, &backprop_null },
};

// Choose a Loss, LossProp, and NN Architecture

#define LOSS_FUNC     l2_one_neuron_loss
#define LOSSPROP_FUNC l2_one_neuron_lossprob
#define NN_TYPE       HALFKP
