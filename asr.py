from transformers import pipeline

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")


transcriber = pipeline(
  "automatic-speech-recognition",
  model="BELLE-2/Belle-whisper-large-v3-zh-punct",
)

transcriber.model.config.forced_decoder_ids = (
  transcriber.tokenizer.get_decoder_prompt_ids(
    language="zh",
    task="transcribe"
  )
)



#transcription = transcriber("./train/000_001.wav")
#print(transcription['text'])


import os

input_dir = './train'
input_dir = '/home/ubuntu/tts/tmp/output'


for root, dirs, files in os.walk(input_dir):
    for file in files:
        if file.endswith(".wav"):
            input_path = os.path.join(root, file)
            text=transcriber(input_path)['text']

            output_filename = os.path.splitext(file)[0] + '.normalized.txt'
            output_path = os.path.join(root, output_filename)

            with open(output_path, 'w') as f:
                f.write(text)

            print(f'转换完成: {file} -> {output_path}')
