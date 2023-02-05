from tensorflow_tts.inference import AutoConfig
from tensorflow_tts.inference import TFAutoModel
from tensorflow_tts.inference import AutoProcessor
from tensorflow_tts.models import TFParallelWaveGANGenerator,TFMelGANGenerator, TFPQMF
from tensorflow_tts.configs import ParallelWaveGANGeneratorConfig, MultiBandMelGANGeneratorConfig
import torch
from parallel_wavegan.models import ParallelWaveGANGenerator
import json
import yaml
import tensorflow as tf
import numpy as np

def load_general_config(cfg_path):
    with open(cfg_path,'r') as cfg_file:
        general_config = json.load(cfg_file)
    return general_config

def load_tacotron2_model(tacotron2_path,tacotron2_cfg_path):
    cfg_tacotron2 = AutoConfig.from_pretrained(tacotron2_cfg_path)
    tacotron2 = TFAutoModel.from_pretrained(tacotron2_path,cfg_tacotron2)
    tacotron2.setup_window(win_front=6, win_back=6)
    tacotron2.setup_maximum_iterations(3000)
    return tacotron2


def load_fastspeech2_model(fastspeech2_path,fastspeech2_cfg_path):
    cfg = AutoConfig.from_pretrained(fastspeech2_cfg_path)
    fastspeech2 = TFAutoModel.from_pretrained(fastspeech2_path,cfg)
    return fastspeech2

def convert_weights_pytorch_to_tensorflow(weights_pytorch):
    """
    Convert pytorch Conv1d weight variable to tensorflow Conv2D weights.
    1D: Pytorch (f_output, f_input, kernel_size) -> TF (kernel_size, f_input, 1, f_output)
    2D: Pytorch (f_output, f_input, kernel_size_h, kernel_size_w) -> TF (kernel_size_w, kernel_size_h, f_input, 1, f_output)
    """
    if len(weights_pytorch.shape) == 3: # conv1d-kernel
        weights_tensorflow = np.transpose(weights_pytorch, (0,2,1))  # [f_output, kernel_size, f_input]
        weights_tensorflow = np.transpose(weights_tensorflow, (1,0,2))  # [kernel-size, f_output, f_input]
        weights_tensorflow = np.transpose(weights_tensorflow, (0,2,1))  # [kernel-size, f_input, f_output]
        return weights_tensorflow
    elif len(weights_pytorch.shape) == 1: # conv1d-bias
        return weights_pytorch
    elif len(weights_pytorch.shape) == 4: # conv2d-kernel
        weights_tensorflow = np.transpose(weights_pytorch, (0,2,1,3))  # [f_output, kernel_size_h, f_input, kernel_size_w]
        weights_tensorflow = np.transpose(weights_tensorflow, (1,0,2,3))  # [kernel-size_h, f_output, f_input, kernel-size-w]
        weights_tensorflow = np.transpose(weights_tensorflow, (0,2,1,3))  # [kernel_size_h, f_input, f_output, kernel-size-w]
        weights_tensorflow = np.transpose(weights_tensorflow, (0,1,3,2))  # [kernel_size_h, f_input, kernel-size-w, f_output]
        weights_tensorflow = np.transpose(weights_tensorflow, (0,2,1,3))  # [kernel_size_h, kernel-size-w, f_input, f_output]
        weights_tensorflow = np.transpose(weights_tensorflow, (1,0,2,3))  # [kernel-size_w, kernel_size_h, f_input, f_output]
        return weights_tensorflow

def load_parallel_wavegan_model(parallel_wavegan_path,p_wavegan_tf_path, p_wavegan_pytorch_path):
    cfg_p_wavegan = AutoConfig.from_pretrained(p_wavegan_tf_path)
    tf_model = TFParallelWaveGANGenerator(config=cfg_p_wavegan, name="parallel_wavegan_generator")
    tf_model._build()
    torch_checkpoints = torch.load(parallel_wavegan_path, map_location=torch.device('cpu'))
    torch_generator_weights = torch_checkpoints["model"]["generator"]

    with open(p_wavegan_pytorch_path) as f:
        cfg_pytorch = yaml.load(f, Loader=yaml.Loader)
    torch_model = ParallelWaveGANGenerator(**cfg_pytorch['generator_params'])
    torch_model.load_state_dict(torch_checkpoints["model"]["generator"])
    torch_model.remove_weight_norm()
    torch_weights = []
    all_keys = list(torch_model.state_dict().keys())
    all_values = list(torch_model.state_dict().values())
    tf_var = tf_model.trainable_variables
    
    idx_already_append = []

    for i in range(len(all_keys) -1):
        if i not in idx_already_append:
            if all_keys[i].split(".")[0:-1] == all_keys[i + 1].split(".")[0:-1]:
                if all_keys[i].split(".")[-1] == "bias" and all_keys[i + 1].split(".")[-1] == "weight":
                    torch_weights.append(convert_weights_pytorch_to_tensorflow(all_values[i + 1].cpu().detach().numpy()))
                    torch_weights.append(convert_weights_pytorch_to_tensorflow(all_values[i].cpu().detach().numpy()))
                    idx_already_append.append(i)
                    idx_already_append.append(i + 1)
            else:
                if i not in idx_already_append:
                    torch_weights.append(convert_weights_pytorch_to_tensorflow(all_values[i].cpu().detach().numpy()))
                    idx_already_append.append(i)
    for i, var in enumerate(tf_var):
        tf.keras.backend.set_value(var, torch_weights[i])
    tf_var = tf_model.trainable_variables
    for i, var in enumerate(tf_var):
        tf.keras.backend.set_value(var, torch_weights[i])
    parallel_wavegan_model = tf_model
    return parallel_wavegan_model

def load_multiband_melgan_model(mb_melgan_path,mb_melgan_cfg_path):
    with open(mb_melgan_cfg_path) as f:
        config = yaml.load(f, Loader=yaml.Loader)
    mb_melgan = TFMelGANGenerator(
            config=MultiBandMelGANGeneratorConfig(**config["multiband_melgan_generator_params"]),
            name="multiband_melgan_generator",
        )
    mb_melgan._build()
    mb_melgan.load_weights(mb_melgan_path)
    pqmf = TFPQMF(
            config=MultiBandMelGANGeneratorConfig(**config["multiband_melgan_generator_params"]), name="pqmf"
        )
    return mb_melgan,pqmf

def load_synth_model(general_config):
    text2mel_cfg_path,preprocess_path, stats_path, vocoder_cfg_path  = general_config['cfg_paths']
    preprocess_config = yaml.load(open(preprocess_path), Loader=yaml.Loader)
    processor = AutoProcessor.from_pretrained("models/ljspeech_mapper.json")
    

    if general_config['text2mel_model'] == 'tacotron2':
        text2mel_model = load_tacotron2_model(general_config['text2mel_path'],text2mel_cfg_path)
    elif general_config['text2mel_model'] == 'fastspeech2':
        text2mel_model = load_fastspeech2_model(general_config['text2mel_path'],text2mel_cfg_path)
    else:
        raise Exception(f"Text2Mel model not supported:{general_config['text2mel_model']}")
    if general_config['vocoder'] == "parallel-wavegan":
        p_wavegan_tf_path, p_wavegan_pytorch_path = vocoder_cfg_path
        vocoder_model = load_parallel_wavegan_model(general_config['vocoder_path'],p_wavegan_tf_path,p_wavegan_pytorch_path)
    elif general_config['vocoder'] == 'mb_melgan':
        vocoder_model = load_multiband_melgan_model(general_config['vocoder_path'],vocoder_cfg_path)
    else: 
        raise Exception(f"Vocoder not supported:{general_config['vocoder']}")
    
    return text2mel_model, vocoder_model,processor