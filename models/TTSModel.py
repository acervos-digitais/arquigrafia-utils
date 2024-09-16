import librosa
import numpy as np
import pydub

from transformers import pipeline
from warnings import simplefilter

simplefilter(action="ignore")

class TTSModel:
  TARGET_SR = 11025
  TARGET_BITRATE = "32k"

  @classmethod
  def tts(cls, txt):
    output = cls.pipeline(txt)
    samples = librosa.resample(output["audio"], orig_sr=output["sampling_rate"], target_sr=TTSModel.TARGET_SR)
    mp3 = pydub.AudioSegment(np.int16(samples * 2 ** 15).tobytes(), frame_rate=TTSModel.TARGET_SR, sample_width=2, channels=1)
    return mp3
