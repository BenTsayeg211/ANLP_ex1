n_seeds = 3
n_training = -1
n_validation = -1
n_predictions = -1
import torch
from datasets import load_dataset
from transformers import AutoConfig, AutoModelForSequenceClassification, TrainingArguments, Trainer, AutoTokenizer
from datasets import load_dataset, Dataset
import numpy as np
import evaluate
from time import time

sst2 = load_dataset("sst2")
if n_training != -1:
    sst2['train'] = sst2['train'].select(range(0, n_training))
if n_validation != -1:
    sst2['validation'] = sst2['validation'].select(range(0, n_validation))
if n_predictions != -1:
    sst2['test'] = sst2['test'].select(range(0, n_predictions))

metric_func = evaluate.load("accuracy")


# helper function- get the prediction and compare to the true value
def compute_metrics(eval_pred):
    logs, true_value = eval_pred
    pred = np.argmax(logs, axis=-1)
    return metric_func.compute(predictions=pred, references=true_value)


# language models
BERT = 'bert-base-cased'
ROBERTA = 'roberta-base'
ELECTRA = 'google/electra-base-generator'
models_list = [BERT, ROBERTA, ELECTRA]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def fine_tuning(lang_model):
  pretrained_config = AutoConfig.from_pretrained(lang_model)
  tokenizer = AutoTokenizer.from_pretrained(lang_model)
  def tokenize(x):
      tokenized = tokenizer(x['sentence'],padding=True, truncation=True,return_tensors='pt')
      #I added padding and truncation because of the requirements of the exercise
      return tokenized

  train_set_after_tokenize = sst2['train'].map(tokenize, batched=True)
  validation_set_after_tokenize = sst2['validation'].map(tokenize, batched=True)
  accs_list=np.array([])
  for i in range(n_seeds):
    # wandb.login(key='33647ccd7f59e77f3f5b4ac6efc326285e9c5e31')
    # run = wandb.init(project="ANLP_EX1", name=lang_model + str(i))
    pre_trained_model = AutoModelForSequenceClassification.from_pretrained(best_model, config=pretrained_config).to(device)
    training_args = TrainingArguments(output_dir="saved_args\ "+str(i)+"\ "+lang_model,seed=i, evaluation_strategy="epoch")
    trainer = Trainer(
        model=pre_trained_model,
        args=training_args,
        train_dataset=train_set_after_tokenize,
        eval_dataset=validation_set_after_tokenize,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    model_after_train = trainer.train()
    cur_acc=trainer.evaluate(validation_set_after_tokenize)['eval_accuracy']
    #check time
    print(f'For seed = {i} the accuracy is {cur_acc}')
    accs_list=np.append(accs_list,cur_acc)
    # run.finish()
  print(accs_list.mean())
  print(accs_list.std())
  return accs_list.mean(),accs_list.std()


# I checked and the best language model is :
# and his best seed is :
best_model = BERT
best_seed = 0


def fine_tuning_best():
    pretrained_config = AutoConfig.from_pretrained(best_model)
    tokenizer = AutoTokenizer.from_pretrained(best_model)

    def tokenize(x):
        tokenized = tokenizer(x['sentence'], padding=True, truncation=True)
        # I added padding and truncation because of the requirements of the exercise
        return tokenized

    train_set_after_tokenize = sst2['train'].map(tokenize, batched=True)
    validation_set_after_tokenize = sst2['validation'].map(tokenize, batched=True)
    pre_trained_model = AutoModelForSequenceClassification.from_config(pretrained_config).to(device)
    training_args = TrainingArguments(output_dir="saved_args\best", seed=best_seed, save_strategy="epoch",
                                      save_total_limit=1)

    trainer = Trainer(
        model=pre_trained_model,
        args=training_args,
        train_dataset=train_set_after_tokenize,
        eval_dataset=validation_set_after_tokenize,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    model_after_train = trainer.train()
    return tokenizer, pre_trained_model, trainer

# import wandb
# import os
# wandb.login(key='33647ccd7f59e77f3f5b4ac6efc326285e9c5e31')
# os.environ["WANDB_PROJECT"] = "ANLP_EX1"
# os.environ["WANDB_LOG_MODEL"] = "false"
# os.environ["WANDB_WATCH"] = "false"

#write res.txt
with open("res.txt",'w') as results, open("predictions.txt",'w') as p:
  start_time=time()
  for model in models_list:
    mean,std = fine_tuning(model)
    results.write(f"{model},{mean} +- {std}\n")
  total_time=time()-start_time
  results.write("----\n")
  results.write(f"train time,{total_time}\n")
  #write pred.txt
  tokenizer,pre_trained_model,trainer=fine_tuning_best()
  pre_trained_model.eval()
  start_time_pred=time()
  for sample in sst2['test']:
    tokenized = tokenizer(sample['sentence'], truncation=True, return_tensors='pt').to(device)
    preds = trainer.prediction_step(pre_trained_model,tokenized, prediction_loss_only=False)
    pred = torch.argmax(preds[1]).item()
    p.write(f"{sample['sentence']}###{pred}\n")
  total_time_pred=time()-start_time_pred
  results.write(f"predict time,{total_time_pred}")
# wandb.finish()

# I checked and the best language model is :
# and his best seed is :
best_model = BERT
best_seed = 0
