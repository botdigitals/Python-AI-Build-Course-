# Tiny GPT (character level)

A GPT with 118,353 parameters, trained from scratch in the free course "Build Your Own AI and Automations Using Python" on botdigitals.com. It writes text in the style of three old books. It is a teaching model, not a product.

## Files

| File | What it is |
|---|---|
| `model.pt` | the learned weights (a PyTorch state_dict) |
| `config.json` | the sizes used to build the model |
| `vocab.json` | the 81 characters the model knows (its tokenizer) |
| `gpt_model.py` | the model code, plus `load_model` and `generate_text` |
| `MODEL_CARD.md` | what the model is, its data and its limits |

## Use it in Google Colab

```python
import urllib.request
import torch

repo = "botdigitals/Python-AI-Build-Course-"
base = "https://raw.githubusercontent.com/" + repo + "/main/model/"
for name in ["gpt_model.py", "model.pt", "config.json", "vocab.json"]:
    urllib.request.urlretrieve(base + name, name)

from gpt_model import load_model, generate_text

device = "cuda" if torch.cuda.is_available() else "cpu"
model, encode, decode, config = load_model(".", device)
print(generate_text(model, encode, decode, config, "The whale ", 200, temperature=0.8, seed=3))
```

A prompt can only use the 81 characters in `vocab.json`. Any other character raises a `KeyError`.

## More

Read `MODEL_CARD.md` before you use the model. The full course is at botdigitals.com.
