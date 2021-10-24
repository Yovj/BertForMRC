import torch
import time
import torch.nn.functional as F
from sklearn import  metrics
import numpy as np
from config import log
from tqdm import tqdm, trange
import os
from data.process.dataset import Dataset
import timeit
from transformers.data.metrics.squad_metrics import compute_predictions_logits, compute_predictions_log_probs, squad_evaluate
from transformers.data.processors.squad import SquadResult,SquadFeatures,SquadExample
from train_eval.evaluate_official2 import eval_squad
from torch.utils.tensorboard import SummaryWriter
from transformers import get_linear_schedule_with_warmup


logger = log.logger

def to_list(tensor):
    return tensor.detach().cpu().tolist() if not tensor==torch.Size([0]) else None

def train(config,model,train_loader,tokenizer):
    global_step = 0
    best_f1 = -1
    tr_loss, logging_loss = 0.0, 0.0
    t_total = len(train_loader) // config.gradient_accumulation_steps * config.nums_epochs

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, eps=config.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=config.warmup_steps, num_training_steps=t_total)

    tb_writer = SummaryWriter()

    model.zero_grad()

    for train_iterator in range(int(config.nums_epochs)):
        epoch_iterator = tqdm(train_loader, desc="Epoch:{} Iteration".format(train_iterator), position=0)
        for step,batch in enumerate(epoch_iterator):
            global_step += 1
            model.train()

            batch = tuple(t.to(config.device) for t in batch)

            inputs = {
                'input_ids':       batch[0],
                'attention_mask':  batch[1],
                'token_type_ids' : batch[2],
                'start_positions': batch[3],
                'end_positions':   batch[4]
            }

            span_loss,start_logits,end_logits,type_prob = model(inputs)

            loss = span_loss

            tr_loss += loss.item()
            loss.backward()

            if (step + 1) % config.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                # tqdm.write(str(loss))

            # Log metrics
            if config.logging_steps > 0 and global_step % config.logging_steps == 0:
                results = evaluate(config, model, tokenizer)
                for key, value in results.items():
                    tqdm.write('{}:{}'.format(key,value))
                    tb_writer.add_scalar('eval_{}'.format(key), value, global_step)
                tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                tb_writer.add_scalar('loss', (tr_loss - logging_loss)/config.logging_steps, global_step)
                logging_loss = tr_loss
                f1 = results['f1']
                if f1 > best_f1:
                    output_path = os.path.join(config.output_dir, 'best_model-{}.pkl'.format(global_step))
                    if not os.path.exists(config.output_dir):
                        os.makedirs(config.output_dir)
                    torch.save(model.state_dict(),output_path)
                    torch.save(config, os.path.join(config.output_dir, 'best_training_args.bin'))
                    logger.info("Saving best model to %s", output_path)
                    best_f1 = f1


            # 保存 checkpoints
            if config.save_steps > 0 and global_step % config.save_steps == 0:
                output_path = os.path.join(config.output_dir, 'checkpoint-{}.pkl'.format(global_step))
                if not os.path.exists(config.output_dir):
                    os.makedirs(config.output_dir)

                torch.save(model.state_dict(),output_path)
                torch.save(config, os.path.join(config.output_dir, 'training_args.bin'))
                logger.info("Saving model checkpoint to %s", output_path)

            if config.max_steps > 0 and global_step > config.max_steps:
                train_iterator.close()
                break

        if config.max_steps > 0 and global_step > config.max_steps:
            train_iterator.close()
            break











def test(config,model,test_iter):
    pass


def evaluate(config, model, tokenizer, prefix=""):
    train_Dataset = Dataset(tokenizer=tokenizer,data_dir=config.dev_path,filename=config.dev_file,is_training=False,config=config,cached_features_file=os.path.join(config.dev_path,"cache_" + config.dev_file.replace("json","data")))
    dataset,examples,features = train_Dataset.dataset,train_Dataset.examples,train_Dataset.features
    eval_dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, shuffle=True)


    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", config.batch_size)

    all_results = []
    start_time = timeit.default_timer()

    for batch in tqdm(eval_dataloader, desc="Evaluating", position=0):
        model.eval()
        batch = tuple(t.to(config.device) for t in batch)

        with torch.no_grad():
            inputs = {
                'input_ids':      batch[0],
                'attention_mask': batch[1]
            }

            example_indices = batch[3]


            outputs = model(inputs)
        output = [to_list(k) for k in outputs]

        for i, example_index in enumerate(example_indices):
            eval_feature = features[example_index.item()]
            unique_id = int(eval_feature.unique_id)


            start_logits, end_logits = output[1][i],output[2][i]
            result = SquadResult(
                unique_id, start_logits, end_logits
            )

            all_results.append(result)

    evalTime = timeit.default_timer() - start_time
    logger.info("  Evaluation done in total %f secs (%f sec per example)", evalTime, evalTime / len(dataset))

    # Compute predictions
    output_prediction_file = os.path.join(config.output_dir, "predictions_{}.json".format(prefix))
    output_nbest_file = os.path.join(config.output_dir, "nbest_predictions_{}.json".format(prefix))

    output_null_log_odds_file = os.path.join(config.output_dir, "null_odds_{}.json".format(prefix))

    predictions = compute_predictions_logits(examples, features, all_results, config.n_best_size,
                    config.max_answer_length, config.do_lower_case, output_prediction_file,
                    output_nbest_file, output_null_log_odds_file, config.verbose_logging,
                    config.version_2_with_negative, config.null_score_diff_threshold,tokenizer)

    # Compute the F1 and exact scores.
    # results = squad_evaluate(examples, predictions)
    #SQuAD 2.0
    results = eval_squad(os.path.join(config.dev_path, config.dev_file), output_prediction_file, output_null_log_odds_file,
                            config.null_score_diff_threshold)
    return results