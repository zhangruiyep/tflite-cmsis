# DTLN example
The DTLN example is a demonstration of DTLN network for Noise suppression in speech.
It uses feature_data as input and provides noise suppressed speech as output.
It is based on the paper(https://github.com/breizhn/DTLN).
While paper presents 2 parts, one for noise suppression and the other for speech enhancement, 
the example presented here follows the noise suppression part only.
The model was re-trained by Cadence using the DNS challenge data (https://github.com/microsoft/DNS-Challenge) 
and the noise suppression part was 8-bit quantized. 
This example is not to be used to evaluate the network quality or quality of noise suppression, but only as a demonstration as stated above.

## Run the tests on a development machine

```
* cd examples/dtln/project
* scons --board=sf32lb52-lcd_n16r8
* .\build_sf32lb52-lcd_n16r8_hcpu\uart_download.bat
```

You should see a series of files get compiled, followed by some logging output
from a test, which should conclude with `~~~ALL TESTS PASSED~~~`. If you see
this, it means that a small program has been built and run that loads a trained
TensorFlow model, runs with features data, and got the expected
outputs. This particular test runs with a feature data as input,
and validate the output with golden reference output.
