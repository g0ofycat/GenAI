import json
import os

def load_config():
    config_dir = os.path.dirname(__file__)
    project_root = os.path.dirname(os.path.dirname(config_dir))
    
    with open(os.path.join(config_dir, 'config.json'), 'r') as f:
        config = json.load(f)

    config['tokenizer']['vocab_path'] = os.path.join(project_root, 'files', config['tokenizer']['vocab_path'])
    config['tokenizer']['merges_path'] = os.path.join(project_root, 'files', config['tokenizer']['merges_path'])
    
    return config

config = load_config()