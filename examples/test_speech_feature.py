import librosa
import numpy as np
import tensorflow as tf
from transform import AudioFeaturizer


class AudioFeaturizerTest(tf.test.TestCase):
  def setUp(self):
    self.audio_file = 'examples/sm1_cln.wav'

  # 1. Test extract spectrum and fbank from audio file
  def test_feature(self):
    names = [{'type': 'Spectrum'}, {'type': 'Fbank'}]
    for name in names:
      ext = AudioFeaturizer(name)
      print('feature dim is ', ext.dim)
      print('feat channel is ', ext.num_channels)
      self.assertEqual(ext.num_channels, 1)
      feat = ext(self.audio_file)
      print('feat shape is ', feat.shape)
      print(feat)

  # 2, Test extract fbank from audio data and delta_delta is true
  def test_fbank(self):
    config = {'type': 'Fbank', 'delta_delta': True}
    ext = AudioFeaturizer(config)
    audio_data, sr = librosa.load(self.audio_file, 16000)
    feat = ext(audio_data, sr)
    print('Fbank feature dim is ', ext.dim)
    print(feat)
    print('feat shape is ', feat.shape)
    self.assertEqual(ext.num_channels, 3)
    print('feat channel is ', ext.num_channels)

  # 3. Test tensorflow tensor
  def test_tensor(self):
    audio_file = tf.convert_to_tensor('examples/sm1_cln.wav', dtype=tf.string)
    ext = AudioFeaturizer()
    feat = ext(audio_file)
    print(feat)
    print('feat shape is ', feat.shape)
    print('feat channel is ', ext.num_channels)
    self.assertEqual(ext.num_channels, 1)

  # 4 Test global cmvn
  def test_cmvn(self):
    dim = 40
    config = {'type': 'CMVN', 'global_mean': np.zeros(dim).tolist(),
              'global_variance': np.ones(dim).tolist()}
    cmvn = AudioFeaturizer(config)
    audio_feature = tf.compat.v1.random_uniform(shape=[10, 40], dtype=tf.float32, maxval=1.0)
    print('cmvn : ', cmvn(audio_feature))

  # 5 Test fbank with global cmvn
  def test_fbank_cmvn(self):
    dim = 23
    config = {'type': 'Fbank',
              'delta_delta': False,
              'global_mean': np.zeros(dim).tolist(),
              'global_variance': np.ones(dim).tolist(),
              'local_cmvn': False}
    ext = AudioFeaturizer(config)
    feat = ext(self.audio_file)
    print('fbank with cmvn', feat)


if __name__ == '__main__':
  if tf.__version__ < '2.0.0':
    tf.compat.v1.enable_eager_execution()
  tf.test.main()
