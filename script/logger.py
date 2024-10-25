import os
import sys
import shutil
import time
import logging
import time
import string

BASE62 = string.digits + string.ascii_letters

def encode_base62(num):
    if num == 0:
        return BASE62[0]
    arr = []
    base = len(BASE62)
    while num:
        num, rem = divmod(num, base)
        arr.append(BASE62[rem])
    arr.reverse()
    return ''.join(arr)

# # Example usage:
# timestamp = int(time.time())  # Current Unix timestamp in seconds
# base62_timestamp = encode_base62(timestamp)
# print("Original timestamp:", timestamp)
# print("Base62 encoded timestamp:", base62_timestamp)


FILE_LIST=["train.py", "simulator.py", "inference.py", 
           "data_loader.py", "embedders.py", "config.py",
           "logger.py"]

class logger:
    def __init__(self, cfg) -> None:
        debugging = cfg["logger"].getboolean("debugging")
        etag = cfg["experiment"].get("etag")
        self.ttag = "{}_{}".format(etag, encode_base62(int(time.time())))
        self.home_dir = "result"
        self._prep(debugging)
        
        
    def _prep(self, debugging):
        # file dir
        ttag = self.ttag if not debugging else "test"
        if not os.path.exists(self.home_dir):
            os.mkdir(self.home_dir)
        self.log_dir = "result/{}".format(ttag)
        self.pic_dir = "{}/pics/train".format(self.log_dir)
        self.model_dir = "{}/model".format(self.log_dir)
        self.control_dir = "{}/control".format(self.log_dir)
        self.text_dir = "{}/log/".format(self.log_dir)
        self.code_dir = "{}/code".format(self.log_dir)
        self.config_dir = "{}/config".format(self.log_dir)
        
        if debugging and os.path.exists(self.log_dir):
            shutil.rmtree(self.log_dir)
        
        for tmp_dir in [self.pic_dir, self.model_dir, self.control_dir, self.text_dir, 
                        self.code_dir, self.config_dir]:
            if not os.path.exists(tmp_dir):
                os.makedirs(tmp_dir, exist_ok=False)
        
        # logging
        logging.basicConfig(filename="{}/log.txt".format(self.text_dir), 
                            level=logging.DEBUG, 
                            format='%(asctime)s %(levelname)-8s %(message)s',
                            datefmt='%Y%m%d %H:%M:%S')
        self.logging = logging
        
                
    def log_file(self):
        os.system("cp {} {}".format(" ".join(FILE_LIST), self.code_dir))
        os.system("chmod -w {}/*".format(self.code_dir))
        
    def info(self, *args):
        self.logging.info(*args)
    
    def error(self, *args):
        self.logging.error(*args)
        
    def log_config(self, config):
        with open("{}/config.ini".format(self.config_dir), "w") as f:
            config.write(f)
            
