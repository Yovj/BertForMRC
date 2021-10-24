
import random
import numpy as np
import torch
from config import Config
import os
from config import log
from model.MRC_model import BertForQA
from train_eval.train_eval import train

from data.process.dataset import Dataset
logger = log.logger

from transformers import BertTokenizer




def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if config.n_gpu > 0:
        torch.cuda.manual_seed_all(seed)



if __name__ == '__main__':
    config = Config()
    set_seed(config.seed)
    tokenizer = BertTokenizer.from_pretrained(os.path.join(config.model_dir,config.model_name))

    train_Dataset = Dataset(tokenizer=tokenizer,data_dir=config.train_path,filename=config.train_file,is_training=True,config=config,cached_features_file=os.path.join(config.train_path,"cache_" + config.train_file.replace("json","data")))
    train_features,train_dataset = train_Dataset.features,train_Dataset.dataset

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

    model = BertForQA(config)
    model.to(config.device)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", config.nums_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", config.batch_size)
    # logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
    #                args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    # logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    # logger.info("  Total optimization steps = %d", t_total)
    train(config,model,train_loader,tokenizer)








    print("done")




