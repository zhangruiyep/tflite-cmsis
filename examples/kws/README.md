<!-- mdformat off(b/169948621#comment2) -->

 Keyword Spotting Example

## Introduction

This is a sample code showing keyword spotting using google TFlite-Micro C++ API. The compiled application can take

* an 16bit 16K sample rate audio PCM data

as input and produce

* recognised keyword in the audio file

as output. The application could works with the [fully quantised DS CNN Large model](https://github.com/ARM-software/ML-zoo/raw/68b5fbc77ed28e67b2efc915997ea4477c1d9d5b/models/keyword_spotting/ds_cnn_large/tflite_clustered_int8/) which is trained to recongize 12 keywords, including an unknown word.

The model used in this example is an ds_cnn_medium model pretrained from 
https://github.com/Arm-Examples/ML-zoo/tree/master/models/keyword_spotting/ds_cnn_medium/model_package_tf/model_archive/TFLite/tflite_int8

## Run the C++ tests on a sf32lb52x board

The following commands show how to compile and run test on a sf32lb52x board :
```bash
* cd examples/kws/project
* scons --board=sf32lb52-lcd_n16r8
* .\build_sf32lb52-lcd_n16r8_hcpu\uart_download.bat
```

If microphone on board is used, please Add 
CONFIG_KWS_MIC_SUPPORT=y
to the end of 
examples/kws/project/proj.conf
