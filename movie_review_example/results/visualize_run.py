import pandas as pd
import json
import pathlib
import matplotlib.pyplot as plt

FILEPATH=pathlib.Path(__file__).parent.absolute()

path_results = FILEPATH/"gpt2-xl/accuracies.json"
results = json.load(open(path_results))
print(results)
results_table = pd.DataFrame.from_dict(results)
results_table['epoch'] = results_table.index + 1

plt.figure()
ax=results_table.plot("epoch", "train_acc", marker='o')
results_table.plot("epoch", "val_acc", ax=ax, marker='x')
plt.title("Fine-tuning GPT-2 1.5B parameter model on aclImdb reviews")
plt.xlabel("epoch"); plt.ylabel("accuracy")
plt.xlim(1,4)
plt.xticks(range(1,4), range(1,4))
plt.savefig(FILEPATH/"test.png")

# df = pd.read_json()
print(results_table.head())