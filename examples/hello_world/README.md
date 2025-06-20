<!-- mdformat off(b/169948621#comment2) -->

# Hello World Example

This example is designed to demonstrate the absolute basics of using [TensorFlow
Lite for Microcontrollers](https://www.tensorflow.org/lite/microcontrollers).
It includes the full end-to-end workflow of training a model, converting it for
use with TensorFlow Lite for Microcontrollers for running inference on a
microcontroller. The project in this repo contain only running inference on SF32LB52X board.


## Run the tests on a sf32lb52-lcd_n16r8 board

```
* cd examples/hellow_world/project
* scons --board=sf32lb52-lcd_n16r8
* .\build_sf32lb52-lcd_n16r8_hcpu\uart_download.bat
```

The source for the test is [hello_world_test.cc](hello_world_test.cc).
It's a fairly small amount of code that creates an interpreter, gets a handle to
a model that's been compiled into the program, and then invokes the interpreter
with the model and sample inputs.


