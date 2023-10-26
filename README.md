
# Finetune llama2 7B model

Here is step-by-step to run this repo


### Step 1: Clone this repository
```bash
git clone https://github.com/devrunner09/llama2_train.git

cd llama-recipes/
```

### Step 2: Install requirements
```bash
pip install -r requirements.txt 
```

### Step 3: Change config

Go to ```llama-recipes/src/llama_recipes/configs/datasets.py```. Change ```data_path``` into your dataset's location.

Go to ```llama-recipes/src/llama_recipes/configs/training.py```. Change ```output_dir``` into location where you want to save PEFT model.

### Step 4: Run train script

```bash
cd src/

python -m llama_recipes.finetuning  --use_peft --peft_method lora --quantization
```
