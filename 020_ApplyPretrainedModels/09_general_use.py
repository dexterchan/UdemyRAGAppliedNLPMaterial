#%% packages
from transformers import pipeline

# %% only provide task
pipe = pipeline(task="text-classification", model=".model/text-classification")
#pipe.save_pretrained(".model/text-classification")
# %% run pipe
pipe("I like it very much.")
# %% provide model
# pipe = pipeline(task="text-classification", 
#                 model="nlptown/bert-base-multilingual-uncased-sentiment")
pipe = pipeline(task="text-classification", 
                model=".model/bert-sentiment")
pipe.save_pretrained(".model/bert-sentiment")
# %% run pipe
# consume just a string
pipe("I like it very much.")

# %% consume a list
pipe(["I like it very much.", 
      "I hate it."])


# %%
