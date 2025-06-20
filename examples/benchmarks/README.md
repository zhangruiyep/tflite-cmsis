# TFLite for Microcontrollers Benchmarks

These benchmarks are for measuring the performance of key models and workloads.
They are meant to be used as part of the model optimization process for a given
platform.

## Keyword benchmark

The keyword benchmark contains a model for keyword detection with scrambled
weights and biases.  This model is meant to test performance on a platform only.
Since the weights are scrambled, the output is meaningless. In order to validate
the accuracy of optimized kernels, please run the kernel tests.

## Person detection benchmark

The keyword benchmark provides a way to evaluate the performance of the 250KB
visual wakewords model. This is the default benchmark application for this project.

## Run on sf32lb52-lcd_n16r8
enter project folder

* cd examples/benchmarks/project

then use command to compile.

* scons --board=sf32lb52-lcd_n16r8

in build folder, issue following command to download to board.

* .\build_sf32lb52-lcd_n16r8_hcpu\uart_download.bat

