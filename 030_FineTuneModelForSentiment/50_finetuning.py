#%% packages
from transformers import AutoModelForSequenceClassification, DistilBertTokenizer, Trainer, TrainingArguments
from datasets import DatasetDict

import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torch.nn.functional import cross_entropy
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.dummy import DummyClassifier
# %% YELP Dataset
# source: https://huggingface.co/datasets/yelp_review_full/viewer/yelp_review_full/train?f%5blabel%5d%5bvalue%5d=0

#%%  dataset
yelp_hidden_states = joblib.load('.models/yelp_hidden_states.joblib')

#%% Model and Tokenizer
model_name = 'distilbert-base-uncased'

device = 'mps' if torch.backends.mps.is_available() else "cpu"  # for Apple Silicon Macs
print(f"Using device: {device}")

num_labels = 5
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels).to(device)
device = 'mps' if torch.backends.mps.is_available() else "cpu"  # for Apple Silicon Macs
tokenizer = DistilBertTokenizer.from_pretrained(model_name)

#%% Dataset
train_ds = yelp_hidden_states.select(range(0, 800))
eval_ds = yelp_hidden_states.select(range(800, 1000))
print(train_ds[0]['input_ids'].shape)
print(eval_ds[0]['input_ids'].shape)
print(yelp_hidden_states[800]['input_ids'].shape)

#%% DatasetDict
yelp_ds_dict = DatasetDict({'train': train_ds, 'test':eval_ds})
MAX_EPOCHS = 5
#%% Trainer Arguments
batch_size = 8  # adapt BS to fit into memory
training_args = TrainingArguments(
    output_dir='./results',          # output directory
    learning_rate=2e-5,              # learning rate
    num_train_epochs=MAX_EPOCHS,              # total number of training epochs
    per_device_train_batch_size=batch_size,  # batch size per device during training
    per_device_eval_batch_size=batch_size,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    disable_tqdm=False,
    push_to_hub=False,
    save_strategy='epoch',
    log_level='error',
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    
)
#%% Trainer
trainer = Trainer(model=model, 
                  args=training_args, 
                  train_dataset=yelp_ds_dict['train'], eval_dataset=yelp_ds_dict['test'])
trainer.train()

# %% get losses
trainer.evaluate()

# %% calculate predictions
preds = trainer.predict(yelp_ds_dict['test'])

# %%
preds.metrics
# %%
np.argmax(preds.predictions, axis=1)
s = np.argmax(preds.predictions, axis=1)
v = [p[inx] for inx, p in zip([int(i) for i in s], preds.predictions)]

# %%

# %% confusion matrix
true_classes = yelp_ds_dict['test']['label']
preds_classes = np.argmax(preds.predictions, axis=1)
conf_mat = confusion_matrix(true_classes, preds_classes)
conf_mat
sns.heatmap(conf_mat, annot=True)
#%% calculat F1 score
from sklearn.metrics import f1_score, precision_recall_fscore_support
f1 = f1_score(true_classes, preds_classes, average='macro')

print(f"F1 Score: {f1}")

# Compute precision, recall, and F1 scores for each class
precision, recall, f1, support = precision_recall_fscore_support(true_classes, preds_classes, average=None)

metrics_df = pd.DataFrame({
    'Precision': precision,
    'Recall': recall,
    'F1 Score': f1,
    'NumOfActualSamples': support
}, index=[0, 1, 2, 3, 4])
metrics_df
#%% calculate Precision and Recall
from sklearn.metrics import precision_score, recall_score
precision = precision_score(true_classes, preds_classes, average='macro')
recall = recall_score(true_classes, preds_classes, average='macro')
print(f"Precision: {precision}, Recall: {recall}")

# %% accuracy
accuracy_score(true_classes, preds_classes)
# %% baseline classifier training

dummy_clf = DummyClassifier(strategy="most_frequent")
dummy_clf.fit(yelp_ds_dict['train']['label'], yelp_ds_dict['train']['label'])

# %% baseline classifier accuracy
dummy_clf.score(yelp_ds_dict['test']['label'], yelp_ds_dict['test']['label'])

# %% inspect individual reviews
model_cpu = model.to('cpu')

test_data = yelp_ds_dict['test'][:]
input_ids = torch.tensor(test_data['input_ids'], dtype=torch.long)
attention_mask = torch.tensor(test_data['attention_mask'], dtype=torch.long)

print(type(input_ids))  # Should be <class 'torch.Tensor'>
print(input_ids.shape)  # Should be (batch_size, sequence_length)
print(type(attention_mask))  # Should be <class 'torch.Tensor'>
print(attention_mask.shape)  # Should be (batch_size, sequence_length)
#%% Inference
with torch.no_grad():
    outputs = model_cpu(input_ids=input_ids, attention_mask=attention_mask)
  
#%% Loss calculation
import torch
import torch.nn.functional as F

# Convert label column to tensor
labels = torch.tensor(test_data['label'], dtype=torch.long)

pred_labels = torch.argmax(outputs.logits, dim=1)
loss = F.cross_entropy(outputs.logits, labels, reduction='none')

# %%

df_individual_reviews = yelp_ds_dict['test'].to_pandas()[['text','label']]
df_individual_reviews['pred_label'] = pred_labels.numpy()
df_individual_reviews['loss'] = loss.numpy()
df_individual_reviews = df_individual_reviews.sort_values('loss', ascending=False).reset_index(drop=True)

# %%
df_individual_reviews
# %%
sns.lineplot(data=df_individual_reviews, x='label', y='loss')
# %% save the model
# login via Terminal: huggingface-cli login
# create Token in HuggingFace Account


#%%
# %% Push the model to HuggingFace Hub
trainer.create_model_card(model_name = 'boar-distilbert-base-uncased-yelp')
trainer.push_to_hub(commit_message='Yelp review classification')

# %% load model from HuggingFace Hub
# name was changed online to distilbert-base-uncased-yelp
from transformers import pipeline
model_id = "dexter-chan/boar-distilbert-base-uncased-yelp"
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
classifier = pipeline('sentiment-analysis', model=model_id, tokenizer=tokenizer)
# %%
# %% visualise scores
res = classifier('it is not so great', return_all_scores=True)[0]
df_res = pd.DataFrame(res)
sns.barplot(data=df_res, x='label', y='score')
# %%
