# data-poisoning-linear-systems

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Code for the paper _Analysis and Detectability of Offline Data Poisoning Attacks on Linear Systems_.

*Author*: Alessio Russo (alessior@kth.se)

*License*: MIT

## Input poisoning attack

To run the example, run the file `main.py` in the folder `example_input_attack`. Feel free to change the standard deviation of the process noise `std_w`, or of the input signal `std_u`.
<p align="center">
  <img src="./example_input_attack/figures/input_poisoning_0.1.png" width="50%">
</p>

## Residuals maximization attack

To run the example, run the file `main.py` in the folder `example_residuals_maximization`. Use `plot_data.py` to plot the results.

To use the data of the paper, first extract the data from the file `example_residuals_maximization\data\data.7z`.
<p align="center">
  <img src="./example_residuals_maximization/figures/example_residuals_variance.png" width="50%">
</p>

## Stealthy attack

