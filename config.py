from pathlib import Path

def get_config():
    return {
        "batch_size": 64,
        "num_epochs": 100,
        "lr": 10**-4,
        "seq_len": 350,
        "d_model": 512,
        "datasource": 'opus-100',
        "lang_src": "en",
        "lang_tgt": "fa",
        "model_folder": "weights",
        "model_basename": "transformer_model",
        "preload": "latest",
        "tokenizer_file": "tokenizer_{}.json",
        "experiment_name": "runs/transformer_model"
    }

def get_weights_file_path(config, epoch: str):
    model_folder = f'{config['datasource']}_{config['model_folder']}'
    model_filename = f"{config['model_basename']}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)

def latest_weights_file_path(config):
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])


