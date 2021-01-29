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

#include "types.h"

/// Activation functions. These functions are all matching
/// >> typedef void (*Activation) (const Vector*, Vector*);

void activate_relu(const Vector *input, Vector *output);
void activate_clipped_relu(const Vector *input, Vector *output);
void activate_sigmoid(const Vector *input, Vector *output);
void activate_null(const Vector *input, Vector *output);

/// BackProp functions. These functions are all matching
/// >> typedef void (*BackProp) (float *dlossdz, const Vector *vector);

void backprop_relu(float *dlossdz, const Vector *vector);
void backprop_clipped_relu(float *dlossdz, const Vector *vector);
void backprop_sigmoid(float *dlossdz, const Vector *vector);
void backprop_null(float *dlossdz, const Vector *vector);

/// Loss and LossProp functions. These functions are all matching
/// >> typedef float (*Loss)     (const Sample*, const Vector *outputs);
/// >> typedef void  (*LossProp) (const Sample*, const Vector *outputs, float *dlossdz);

float l2_one_neuron_loss(const Sample *sample, const Vector *outputs);
void l2_one_neuron_lossprob(const Sample *sample, const Vector *outputs, float *dlossdz);

float l2_loss_phased(const Sample *sample, const Vector *outputs);
void l2_loss_phased_lossprop(const Sample *sample, const Vector *outputs, float *dlossdz);
