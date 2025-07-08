<!-- mdformat off(b/169948621#comment2) -->

# Micro Speech Example

This example shows how to run inference using TensorFlow Lite Micro (TFLM)
on two models for wake-word recognition.
The first model is an audio preprocessor that generates spectrogram data
from raw audio samples.
The second is the Micro Speech model, a less than 20 kB model
that can recognize 2 keywords, "yes" and "no", from speech data.
The Micro Speech model takes the spectrogram data as input and produces
category probabilities.


## Run the C++ tests on a sf32lb52x board

The following commands show how to compile and run test on a sf32lb52x board :
```bash
* cd examples/micro_speech/project
* scons --board=sf32lb52-lcd_n16r8
* .\build_sf32lb52-lcd_n16r8_hcpu\uart_download.bat
```
