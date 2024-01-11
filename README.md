## ESPnet Text to Byte Array

This project leverages ESPnet2 for converting text to a byte array and subsequently reconstructing the audio from the byte data.

## Setup

Follow the steps below to set up and run the project:

## Step 1: Clone the Repository
```bash
git clone https://github.com/samiabat/ESPnet-text-2-byte-audio-buffer.git
```

## Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

## Step 3: Organize Model Files
Create a folder named 'model' in the project root directory.
 Place the 'config.yaml' and 'train.total_count.ave_10best.pth' files inside the 'model' folder.

## Step 4: Prepare Text File
Create a file named 'text.txt' in the project root directory and add the text you want to synthesize.

## Step 5: Run the Code
```bash
python3 ESPnetT2S.py
```

## Additional Notes
- The `ESPnetTextToByte` class in `ESPnetT2S.py` handles the text-to-speech conversion and byte array creation.
- The `get_byte_data` method in the class writes the byte data to a file named 'audio_byte_file.raw'.
- The `recreate_audio_from_bytes` method reads the byte file and recreates the audio, saving it as 'recreated_audio.wav'.
- Adjust the file paths as needed, and feel free to customize the code to suit your requirements.
