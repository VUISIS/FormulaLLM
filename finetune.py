from xturing.datasets.instruction_dataset import InstructionDataset
from xturing.models import BaseModel
from pytorch_lightning.loggers import WandbLogger

import os

# Initializes WandB integration 
wandb_logger = WandbLogger()

prompt = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{text}\n\n### Response:"

instruction_dataset = InstructionDataset(os.path.abspath("./alpaca_data"), promt_template=prompt)
# Initializes the model
model = BaseModel.create("opt_lora")
#model.finetuning_config().batch_size = 2
#model.finetuning_config().max_length = 512

model.finetune(dataset=instruction_dataset, logger=wandb_logger)

model.save(os.path.abspath("./saved"))
