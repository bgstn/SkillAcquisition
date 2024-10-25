import json
import argparse
import configparser

def parse_config():
    config_path = "config.ini"
    config = configparser.ConfigParser()
    config.read(config_path)
    
    parser = argparse.ArgumentParser(description="Configuration Overrides")
    
    # Simulator Section
    parser.add_argument('--ntrajs', type=int, default=config.getint('simulator', 'ntrajs'), help='Override ntrajs in simulator')
    
    # Network Section
    parser.add_argument('--model_type', type=str, default=config.get('network', 'model_type'), help='Override model_type in Network')
    parser.add_argument('--n_skill', type=int, default=config.getint('network', 'n_skill'), help='Override seq_len in Network')
    
    # message 
    parser.add_argument("-m", '--message', type=str, required=True, help="experiment info")
    parser.add_argument("-t", '--etag', type=str, required=True, help="experiment tag")
    # parse args
    cmd_args = parser.parse_args()
    
    # update the arguments
    for section in config.sections():
        for key in config[section]:
            if getattr(cmd_args, key, None) is not None:
                config[section][key] = str(getattr(cmd_args, key))
    
    # print configurations
    configurations = {}
    for section in config.sections():
        options = {}
        for key, value in config[section].items():
            options[key] = value
        configurations[section] = options
    return config, json.dumps(configurations, indent=4)

if __name__ == "__main__":
    import json
    res = parse_config()
    print(res[1])