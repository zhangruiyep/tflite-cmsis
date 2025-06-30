# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
"""
Wake-word model evaluation, with audio preprocessing using MicroInterpreter

Run:
bazel build tensorflow/lite/micro/examples/micro_speech:evaluate
bazel-bin/tensorflow/lite/micro/examples/micro_speech/evaluate
  --sample_path="path to 1 second audio sample in WAV format"
"""

from absl import app
from absl import flags
import numpy as np
from pathlib import Path

import tensorflow as tf
from tflite_micro.python.tflite_micro import runtime

_SAMPLE_PATH = flags.DEFINE_string(
    name='sample_path',
    default='sample.wav',
    help='path for the audio sample to generate feature.',
)

_MODEL_PATH = flags.DEFINE_string(
    name='model_path',
    default='audio_preprocessor_int8.tflite',
    help='path for audio preprocessor folder',
)

_FREQUENCY = flags.DEFINE_integer(
    name='frequency',
    default=16000,
    help='Audio sample frequency',
)

_WINDOW_MS = flags.DEFINE_integer(
    name='window_ms',
    default=30,
    help='Audio Windows size in ms',
)

_STRIDE_MS = flags.DEFINE_integer(
    name='stride_ms',
    default=20,
    help='Audio Stride size in ms',
)

_FEATURE_SIZE = flags.DEFINE_integer(
    name='feature_size',
    default=40,
    help='Feature size for each small audio window',
)

def _main(_):
  file_data = tf.io.read_file(str(_SAMPLE_PATH.value))
  samples: tf.Tensor
  samples, sample_rate = tf.audio.decode_wav(file_data, desired_channels=1)
  max_value = tf.dtypes.int16.max
  min_value = tf.dtypes.int16.min
  samples = ((samples * max_value) + (-min_value + 0.5)) + min_value
  samples = tf.cast(samples, tf.int16)  # type: ignore
    
  tflm_interpreter = runtime.Interpreter.from_file(_MODEL_PATH.value)
  
  window_size = int(_WINDOW_MS.value * _FREQUENCY.value / 1000)
  window_stride = int(_STRIDE_MS.value * _FREQUENCY.value / 1000)
 
  # 计算可切割的片段数量
  num_segments = tf.math.floordiv(tf.size(samples)-window_size, window_stride) + 1

  # 切割音频
  segments = tf.TensorArray(
    dtype=tf.int16,
    size=num_segments,
    dynamic_size=True
  )
  print(segments)
  for i in tf.range(num_segments):    
    start = i*window_stride
    end =  start+ window_size
    if end>len(samples):
      break
    segment = tf.strided_slice(samples, [start], [end])
    segments = segments.write(i, segment)
  segments = segments.stack()
  print(segments)

  # 切割音频
  segments_out = tf.TensorArray(
    dtype=tf.int8,
    size=num_segments,
    dynamic_size=True
  )   
  i=0
  for segment in segments:
    # 设置输入张量
    segment= tf.transpose(segment)
    tflm_interpreter.set_input(segment,0)
    tflm_interpreter.invoke()
    segment_out=tflm_interpreter.get_output(0)
    segments_out=segments_out.write(i, segment_out)
    i=i+1
  segments_out = segments_out.stack()
  print(segments_out)  # 例如: [40, 400]
    
if __name__ == '__main__':
  app.run(_main)
