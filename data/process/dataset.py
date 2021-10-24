import os
import torch
from config import log
from transformers.data.processors.squad import squad_convert_examples_to_features,SquadV2Processor


logger = log.logger

class Dataset:
    def __init__(self,tokenizer,data_dir,filename,is_training,config,cached_features_file):
        self.tokenizer = tokenizer
        self.data_dir = data_dir
        self.filename = filename
        self.is_training = is_training
        self.config = config
        self.cached_features_file = cached_features_file
        if is_training:
            self.dataset,self.features = self.load_and_cache_dataset()
        else:
            self.dataset,self.examples,self.features = self.load_and_cache_dataset()

    def load_and_cache_dataset(self):
        if self.is_training:
            if os.path.exists(self.cached_features_file):
                logger.info("Loading features from cached file %s", self.cached_features_file)
                features_and_dataset = torch.load(self.cached_features_file)
                features, dataset = features_and_dataset["features"], features_and_dataset["dataset"]
            else:
                processor = SquadV2Processor()
                examples = processor.get_train_examples(self.data_dir,self.filename)

                features, dataset = squad_convert_examples_to_features(
                        examples=examples,
                        tokenizer=self.tokenizer,
                        max_seq_length=self.config.max_seq_length,
                        doc_stride=self.config.doc_stride,
                        max_query_length=self.config.max_query_length,
                        is_training=self.is_training,
                        return_dataset='pt'
                    )
                logger.info("Saving features into cached file %s",  self.cached_features_file)
                torch.save({"features": features, "dataset": dataset},  self.cached_features_file)
            return dataset,features
        else:
            if os.path.exists(self.cached_features_file):
                logger.info("Loading features from cached file %s", self.cached_features_file)
                features_and_dataset = torch.load(self.cached_features_file)
                features, dataset,examples = features_and_dataset["features"], features_and_dataset["dataset"], features_and_dataset["examples"]
            else:
                processor = SquadV2Processor()
                examples = processor.get_dev_examples(self.data_dir,self.filename)

                features, dataset = squad_convert_examples_to_features(
                        examples=examples,
                        tokenizer=self.tokenizer,
                        max_seq_length=self.config.max_seq_length,
                        doc_stride=self.config.doc_stride,
                        max_query_length=self.config.max_query_length,
                        is_training=self.is_training,
                        return_dataset='pt'
                    )

                logger.info("Saving features into cached file %s", self.cached_features_file)
                torch.save({"features": features, "dataset": dataset,"examples":examples}, self.cached_features_file)

            return dataset,examples,features





