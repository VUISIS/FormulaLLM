from xturing.datasets.instruction_dataset import InstructionDataset
from xturing.models import BaseModel

import os

prompt = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{text}\n\n### Response:"

data = {
    "instruction": ["Explain why this model is solvable", "Explain why this model is not solvable"],
    "text": ["domain Simple\n{\n  Ape ::= new (id: Integer).\n  Bat ::= (id: Integer).\n  Cat ::= (id: Integer).\n\n  Bat(x) :- Ape(x), x > -10.\n  Cat(x) :- Ape(x), x < 10.\n\n  batModel :- Bat(x).\n  catModel :- Cat(x).\n\n conforms batModel, catModel.\n}\n\npartial model pm of Simple\n{\n  Ape(x).\n}\n, Ape(0) Bat(0) Cat(0)",
             "domain Simple\n{\n  Super ::= new (id: Integer).\n  Node ::= (id : Integer).\n  Code ::= (id : Integer).\n\n  Node(y) :- Super(x), x > 20, y = x - 1.\n  Code(y) :- Super(x), x < 20, y = x - 1.\n  nodeModel :- Node(x).\n  codeModel :- Code(x).\n\n  conforms nodeModel, codeModel.\n}\n\npartial model pm of Simple\n{\n  Super(x).\n}\n, Node(x - 1) Code(x - 1)"],
    "target": ["", ""]
}

instruction_dataset = InstructionDataset(data, promt_template=prompt)

model = BaseModel.load_from_local(os.path.abspath("./saved_model"))

output = model.generate(dataset=instruction_dataset)

for i,o in enumerate(output):
    print("Input used by LLM:")
    print()
    print(data["text"][i])
    print()
    print("Generated output by the model:")
    print()
    print(output[i])