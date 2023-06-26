from xturing.datasets.instruction_dataset import InstructionDataset
from xturing.models import GenericLoraModel

instruction_dataset = InstructionDataset("./alpaca_data")

model = GenericLoraModel('mosaicml/mpt-7b-instruct', target_modules=["q_proj", "v_proj"])

#model.finetune(dataset=instruction_dataset)

output = model.generate(texts=["Why LLM models are becoming so important?"])

print("Generated output by the model: {}".format(output))
model.save("./generic_lora_weights")