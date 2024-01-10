from espnet2.bin.tts_inference import Text2Speech
from espnet2.utils.types import str_or_none
from scipy.io.wavfile import write
import torch
import numpy as np

class ESPnetTextToByte:
    def __init__(self):
        self.text2speech = None
    def build(self, model, config,vocoder_tag):
        try:
            self.text2speech = Text2Speech.from_pretrained(
                model_file=str_or_none(model),
                train_config=str_or_none(config),
                vocoder_tag=str_or_none(vocoder_tag),
                device="cpu"
            )
        except Exception as e:
            raise e
            
    def remove_new_line(self, text_file):
        with open(text_file, 'r', encoding='utf-8') as file:
            text = file.read().replace('\n', '')
        return text
            
    def get_wav_data(self, text_file):
        text = self.remove_new_line(text_file)
        with torch.no_grad():
            wav = self.text2speech(text)["wav"]
        wavdata = wav.view(-1).cpu().numpy()
        return wavdata
    
    
    def get_audio(self, text_file):
        wavdata = self.get_wav_data(text_file)
        samplerate = self.text2speech.fs
        write("audio.wav", samplerate, wavdata)
        
    def get_byte_data(self, text_file, output_path="audio_byte_file.raw"):
        wavdata = self.get_wav_data(text_file)
        with open(output_path, 'wb') as file:
            file.write(wavdata.tobytes())
    
    # for test pupose
    def recreate_audio_from_bytes(self, byte_file, output_path="recreated_audio.wav"):
        with open(byte_file, 'rb') as file:
            byte_data = file.read()
        wavdata = np.frombuffer(byte_data, dtype=np.float32)
        write(output_path, self.text2speech.fs, wavdata)

espnet = ESPnetTextToByte()
model_path = "model/train.total_count.ave_10best.pth"
config_file_path = "model/config.yaml"
text_file = "text.txt"
vocoder_tag = "none"

espnet.build(model_path, config_file_path, vocoder_tag)
espnet.get_audio(text_file)
byte_data = espnet.get_byte_data(text_file)
espnet.recreate_audio_from_bytes("audio_byte_file.raw")

