# Copyright (C) 2017 Beijing Didi Infinity Technology and Development Co.,Ltd.
# All rights reserved.
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
# ==============================================================================
"""The model tests WriteWav OP."""

import os
from pathlib import Path
import tensorflow as tf
from tensorflow.python.framework.ops import disable_eager_execution
from transform.feats.read_wav import ReadWav
from transform.feats.write_wav import WriteWav

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

class WriteWavTest(tf.test.TestCase):
    """
    WriteWav OP test.
    """
    def test_write_wav(self):

        wav_path = str(
            Path(os.environ['MAIN_ROOT']).joinpath('examples/sm1_cln.wav'))
        with self.cached_session() as sess:
            speed = 0.9
            read_wav = ReadWav.params().instantiate()
            input_data, sample_rate = read_wav(wav_path, speed)
            write_wav = WriteWav.params().instantiate()
            new_path = wav_path[:-4] + '_speed.wav'
            writewav_op = write_wav(new_path, input_data / 32768.0, sample_rate)
            sess.run(writewav_op)
            del writewav_op


if __name__ == '__main__':
    is_eager = True
    if not is_eager:
        disable_eager_execution()
    else:
        if tf.__version__ < '2.0.0':
            tf.compat.v1.enable_eager_execution()
    tf.test.main()
