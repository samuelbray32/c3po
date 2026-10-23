from pathlib import Path
from c3po.analysis.utils_io import read_config


config_files = Path(__file__).parent.glob("*.json")
configs = {config_file.stem: read_config(config_file) for config_file in config_files}
globals().update(configs)
