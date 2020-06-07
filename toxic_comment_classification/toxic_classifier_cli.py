import json
import requests
import argparse
from pathlib import Path
from pprint import pprint
parser = argparse.ArgumentParser("Toxic comment classification client")

parser.add_argument("--address", type=str, help="address (of format <host>:<port>) of the model server", required=True)
parser.add_argument("--input", type=Path, help="path to the file with texts to classify. Each line should contain a single text.")
parser.add_argument("--output", type=Path, default=Path("out.txt"), help="path to the file with outputs.")

args = parser.parse_args()

host = args.address
input_file = args.input
output_file = args.output

address = f"{host}/v1/models/ToxicClassifier:predict"

with input_file.open("r") as f:
    texts = f.readlines()
    

query = {
    "signature_name": "serving_default",
    "instances": texts
}

data = json.dumps(query)

response = requests.post(
    address,
    data=data
)

predictions = json.loads(response.content)["predictions"]

snt_to_pred = {
    snt: pred[0]
    for (snt, pred) in zip(texts, predictions)
}

pprint(snt_to_pred)

with output_file.open("w") as f:
    for p in predictions:
        digit = 0 if p[0] < 0.5 else 1
        f.write(f"{digit}\n")


print("Predictions written to", output_file)