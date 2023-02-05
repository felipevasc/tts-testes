from model_loaders import *
from text_utils import normalize_text
import numpy as np
import noisereduce as nr


class TTS_controller():
    def __init__(self, config_path: str) -> None:
        self.update_model(config_path)

    def update_model(self,config_path: str) -> None:
        self.general_config = load_general_config(config_path)
        self.model, self.vocoder,self.processor = load_synth_model(self.general_config)

    def inference(self,raw_text_input: str, speech_settings: dict = None) -> np.ndarray:
        synth_text = normalize_text(raw_text_input)
        input_ids = self.processor.text_to_sequence(synth_text)
        print(f'Texto para falar: {synth_text}')
        if self.general_config['text2mel_model'] == 'tacotron2':
            _, mel_outputs, _, _= self.model.inference(
                    tf.expand_dims(tf.convert_to_tensor(input_ids, dtype=tf.int32), 0),
                    tf.convert_to_tensor([len(input_ids)], tf.int32),
                    tf.convert_to_tensor([0], dtype=tf.int32)
            )
        elif self.general_config['text2mel_model'] == 'fastspeech2':
            speed_ratios = 2 - float(speech_settings["speed"])# 2 -x para inverter (maior valor, maior velocidade)
            f0_ratios = float(speech_settings["pitch"])
            energy_ratios = float(speech_settings["energy"])
            print(f'Speed_ratios: {speed_ratios}, f0_ratios: {f0_ratios}, Energy ratios: {energy_ratios}')
            _, mel_outputs, _, _, _ = self.model.inference(
                input_ids=tf.expand_dims(tf.convert_to_tensor(input_ids, dtype=tf.int32), 0),
                speaker_ids=tf.convert_to_tensor([0], dtype=tf.int32),
                speed_ratios=tf.convert_to_tensor([speed_ratios], dtype=tf.float32),
                f0_ratios =tf.convert_to_tensor([f0_ratios], dtype=tf.float32),
                energy_ratios =tf.convert_to_tensor([energy_ratios], dtype=tf.float32)
            )
        mel_outputs = tf.reshape(mel_outputs, [-1, 80]).numpy()
        if self.general_config['vocoder'] == "parallel-wavegan":
            audio = self.vocoder(tf.expand_dims(mel_outputs,axis=0))[0, :, 0]
        elif self.general_config['vocoder'] == 'mb_melgan':
            mb_melgan,pqmf = self.vocoder
            generated_subbands = mb_melgan(mel_outputs[None,...])
            audio = pqmf.synthesis(generated_subbands)[0,:,0]

        audio_denoised = nr.reduce_noise(y=audio.numpy(), sr=self.general_config['sample_rate'],prop_decrease=self.general_config['denoiser_alpha'])
        return audio_denoised