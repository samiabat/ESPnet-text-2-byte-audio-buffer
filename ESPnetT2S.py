from espnet2.bin.tts_inference import Text2Speech
from espnet2.utils.types import str_or_none
import torch

class ESPnetTextToByte:
    '''
    input: 
        1, espnet model
        2, config file
        3, audio file
    output:
        1, audio file as a byte data
    '''
    def __init__(self):
        self.text2speech = None
    def build(self, model, config,vocoder_tag, device="cpu"):
        # This function will initialize the text2speech instance
        try:
            self.text2speech = Text2Speech.from_pretrained(
                model_file=str_or_none(model),
                train_config=str_or_none(config),
                vocoder_tag=str_or_none(vocoder_tag),
                device=device
            )
        except Exception as e:
            raise e
            
    def remove_new_line(self, text_file_path):
        # This function will remove new lines from the text
        with open(text_file_path, 'r', encoding='utf-8') as file:
            text = file.read().replace('\n', '')
        return text
            
    def get_wav_data(self, text_file_path):
        # This function will give us a wav data
        text = self.remove_new_line(text_file_path)
        with torch.no_grad():
            wav = self.text2speech(text)["wav"]
        wavdata = wav.view(-1).cpu().numpy()
        return wavdata
    
    def get_byte_data(self, text_file, output_path="audio_byte_file.raw"):
        # we will get bytefile audio_byte_file.raw
        wavdata = self.get_wav_data(text_file)
        with open(output_path, 'wb') as file:
            file.write(wavdata.tobytes())


if __name__ == "__main__":
    
    # Initialize model parameters
    model_path = "model/train.total_count.ave_10best.pth"
    config_file_path = "model/config.yaml"
    text_file_path = "text.txt"
    vocoder_tag = "parallel_wavegan/vctk_parallel_wavegan.v1.long"

    # Initialize the espnet model
    espnet = ESPnetTextToByte()
    espnet.build(model_path, config_file_path, vocoder_tag)

    # Initialize output file path and and getbyte data
    output_path = "audio_byte_file.raw"
    espnet.get_byte_data(text_file_path, output_path)


