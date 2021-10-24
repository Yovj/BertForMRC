import os
import torch
import logging
from logging import handlers
from transformers import BertTokenizer

class Config(object):
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.n_gpu = torch.cuda.device_count()
        self.seed = 18

        self.model_name = "bert-base-uncased"
        self.max_seq_length = 512
        self.doc_stride = 128
        self.max_query_length = 64
        self.Tokenizer = BertTokenizer
        self.num_type = 2

        self.train_path = "data/train"
        self.train_file = "train-v2.0.json"
        self.dev_path = "data/dev"
        self.dev_file = "dev-v2.0.json"
        self.check_point_path = "" # 是否继续训练

        self.model_dir = "model"
        self.logging_path = "all.log"
        self.output_dir = "model/checkpoints"
        self.data_output_dir = ""


        self.batch_size = 10
        self.learning_rate = 1e-4
        # self.optimizer = 'Adam'
        self.adam_epsilon = 1e-8
        self.nums_epochs = 3
        self.max_steps = 1000000000
        self.save_steps = 10000
        self.gradient_accumulation_steps = 50
        self.logging_steps = 1000
        self.warmup_steps = 0


        self.hidden_size = 768
        self.num_class = 10

        self.n_best_size = 20 # "The total number of n-best predictions to generate in the nbest_predictions.json output file."
        self.max_answer_length = 30 # "The maximum length of an answer that can be generated. This is needed because the start and end predictions are not conditioned on one another."
        self.do_lower_case = True
        self.verbose_logging = True
        self.version_2_with_negative = True
        self.null_score_diff_threshold = 0.0 # "If null_score - best_non_null is greater than the threshold predict null."

class Log(object):
    level_relations = {
        'debug':logging.DEBUG,
        'info':logging.INFO,
        'warning':logging.WARNING,
        'error':logging.ERROR,
        'crit':logging.CRITICAL
    }#日志级别关系映射

    def __init__(self,filename,level='info',when='D',backCount=3,fmt='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'):
        self.logger = logging.getLogger(filename)
        format_str = logging.Formatter(fmt)#设置日志格式
        self.logger.setLevel(self.level_relations.get(level))#设置日志级别
        sh = logging.StreamHandler()#往屏幕上输出
        sh.setFormatter(format_str) #设置屏幕上显示的格式
        th = handlers.TimedRotatingFileHandler(filename=filename,when=when,interval=3,backupCount=backCount,encoding='utf-8')#往文件里写入#指定间隔时间自动生成文件的处理器
        #实例化TimedRotatingFileHandler
        #interval是时间间隔，backupCount是备份文件的个数，如果超过这个个数，就会自动删除，when是间隔的时间单位，单位有以下几种：
        # S 秒
        # M 分
        # H 小时、
        # D 天、
        # W 每星期（interval==0时代表星期一）
        # midnight 每天凌晨
        th.setFormatter(format_str)#设置文件里写入的格式
        # self.logger.addHandler(sh) #把对象加到logger里
        self.logger.addHandler(th)


log = Log('all.log',level='debug')