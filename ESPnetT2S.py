from espnet2.bin.tts_inference import Text2Speech
from espnet2.utils.types import str_or_none
import torch
from scipy.io.wavfile import write

class ESPnetTextToByte:
    """
    Processes an audio file using the provided ESPnet model and configuration.

    Parameters:
    - espnet_model (object): An instance of the ESPnet model for audio processing.
    - config_file (str): The path to the configuration file associated with the ESPnet model.
    - audio_file (str): The path to the input audio file for processing.

    Returns:
    - bytes: The processed audio data as a byte object.

    Raises:
    - Any exceptions raised during the audio processing with the ESPnet model.

    Note:
    - This function takes an ESPnet model, a configuration file, and an text file as input.
    - It returns the processed audio data as a byte object.
    """
    def __init__(self):
        self.text2speech = None
    def build(self, model, config, vocoder_tag, device="cpu"):
        """
        Builds the text-to-speech (TTS) model for audio processing.

        Parameters:
        - model (str): Path to the TTS model file.
        - config (str): Path to the TTS model configuration file.
        - vocoder_tag (str): Tag associated with the vocoder.
        - device (str): Device on which the TTS model will be loaded (default is "cpu").

        Raises:
        - Exception: If an error occurs during the initialization of the TTS model.

        Note:
        - This method initializes the text2speech attribute using the provided model, config, vocoder_tag, and device.
        """
        try:
            # Initialize the Text2Speech instance from ESPnet
            self.text2speech = Text2Speech.from_pretrained(
                model_file=str_or_none(model),
                train_config=str_or_none(config),
                vocoder_tag=str_or_none(vocoder_tag),
                device=device
            )
        except Exception as e:
            # Raise an exception if an error occurs during TTS model initialization
            raise e
            
    def remove_new_line(self, text_file_path):
        """
        Removes new lines from the text in the specified file.

        Parameters:
        - text_file_path (str): Path to the text file.

        Returns:
        - str: Text content without new lines.

        Note:
        - This function reads the content of the text file at the specified path and removes new line characters.
        """
        # Open the text file, read its content, and replace new lines with an empty string
        with open(text_file_path, 'r', encoding='utf-8') as file:
            text = file.read().replace('\n', '')
        return text
            
    def get_wav_data(self, text_file_path):
        """
        Generates WAV data from the text content of the specified file.

        Parameters:
        - text_file_path (str): Path to the text file.

        Returns:
        - numpy.ndarray: WAV data as a one-dimensional NumPy array.

        Note:
        - This function utilizes the `remove_new_line` method to preprocess the text and then generates WAV data
          using the text-to-speech (TTS) model.
        """
        # Remove new lines from the text content
        text = self.remove_new_line(text_file_path)

        # Generate WAV data using the text-to-speech model
        with torch.no_grad():
            wav = self.text2speech(text)["wav"]

        # Convert WAV data to a one-dimensional NumPy array
        wavdata = wav.view(-1).cpu().numpy()
        return wavdata
    
    def get_byte_data(self, text_file, output_path="audio_byte_file.raw"):
        """
        Converts WAV data generated from the text content into a byte file.

        Parameters:
        - text_file (str): Path to the text file.
        - output_path (str): Path to the output byte file (default is "audio_byte_file.raw").

        Note:
        - This function utilizes the `get_wav_data` method to generate WAV data from the text content.
        - The WAV data is then written to a binary file in raw byte format.
        """
        # Generate WAV data from the text content
        wavdata = self.get_wav_data(text_file)

        # Write the WAV data to a binary file in raw byte format
        with open(output_path, 'wb') as file:
            file.write(wavdata.tobytes())
            
    def get_audio(self, text_file):
        """
        Generates an audio file from the text content.

        Parameters:
        - text_file (str): Path to the text file.

        Note:
        - This function is for just test purpose.
        - This function utilizes the `get_wav_data` method to generate WAV data from the text content.
        - The WAV data is then used to create an audio file in WAV format.
        """
        # Generate WAV data from the text content
        wavdata = self.get_wav_data(text_file)

        # Get the sample rate from the text-to-speech model
        samplerate = self.text2speech.fs

        # Write the WAV data to an audio file in WAV format
        write("audio.wav", samplerate, wavdata)


if __name__ == "__main__":
    
    # Initialize model parameters
    model_path = "model/train.total_count.ave_10best.pth"
    config_file_path = "model/config.yaml"
    text_file_path = "text.txt"
    vocoder_tag = "none"
    device ="cpu"
    

    # Initialize the espnet model
    espnet = ESPnetTextToByte()
    espnet.build(model_path, config_file_path, vocoder_tag, device)

    # Initialize output file path and and getbyte data
    output_path = "audio_byte_file.raw"
    espnet.get_byte_data(text_file_path, output_path)
    
    # Generate audio from text file
    espnet.get_audio(text_file_path)


