The aim of this application is to provide a quick way to test different
networks.

It contains one testcase and a default network model (network_model.h), default
input data (input_data.h) and default expected output data
(expected_output_data.h). The header files were created using the `xxd` command.

The default model is a single int8 DepthwiseConv2D operator, with an input shape
of {1, 8, 8, 16}, {1, 2, 2, 16} and {16} and an output shape of {1, 4, 4, 16}.

## Run the tests on a sf32lb52x board

```
* cd examples/network_tester/project
* scons --board=sf32lb52-lcd_n16r8
* .\build_sf32lb52-lcd_n16r8_hcpu\uart_download.bat
```
