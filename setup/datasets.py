import tensorflow_datasets as tfds
import pandas as pd
import os
import gzip
import json
from datetime import datetime

# Print available configurations for the ted_hrlr_translate dataset
builder = tfds.builder('ted_hrlr_translate')
print(f"Available configurations for ted_hrlr_translate: {builder.builder_configs.keys()}")

# Dataset names
datasets = ["wmt14_translate/fr-en", "wmt14_translate/de-en"]

# Create the datasets folder if it doesn't exist
os.makedirs("datasets", exist_ok=True)

metadata = {}

for dataset_name in datasets:
    # Download and prepare the dataset
    dataset = tfds.load(dataset_name, as_supervised=True)

    # Print the available splits
    print(f"Available splits for {dataset_name}: {dataset.keys()}")

    # Extract train, validation, and test sets
    train_set, validation_set, test_set = dataset['train'], dataset['validation'], dataset['test']

    # Check if the dataset is not empty
    if not any(tfds.as_numpy(train_set.take(1))):
        print(f"Train set for {dataset_name} is empty.")
        continue
    if not any(tfds.as_numpy(validation_set.take(1))):
        print(f"Validation set for {dataset_name} is empty.")
        continue
    if not any(tfds.as_numpy(test_set.take(1))):
        print(f"Test set for {dataset_name} is empty.")
        continue

    # Convert a small part of the dataset to CSV
    def to_csv(dataset, filename, num_samples=100):
        data = []
        for example in tfds.as_numpy(dataset.take(num_samples)):
            data.append({
                'input': example[0].decode('utf-8'),
                'target': example[1].decode('utf-8')
            })
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)

    # Save a small part of the datasets to CSV
    to_csv(train_set, f"datasets/{dataset_name.split('/')[-1]}_train.csv")
    to_csv(validation_set, f"datasets/{dataset_name.split('/')[-1]}_validation.csv")
    to_csv(test_set, f"datasets/{dataset_name.split('/')[-1]}_test.csv")

    # Save the complete datasets in compressed format
    def save_compressed(dataset, filename):
        with gzip.open(filename, 'wb') as f:
            for example in tfds.as_numpy(dataset):
                f.write(f"{example[0].decode('utf-8')}\t{example[1].decode('utf-8')}\n".encode('utf-8'))

    save_compressed(train_set, f"datasets/{dataset_name.split('/')[-1]}_train.txt.gz")
    save_compressed(validation_set, f"datasets/{dataset_name.split('/')[-1]}_validation.txt.gz")
    save_compressed(test_set, f"datasets/{dataset_name.split('/')[-1]}_test.txt.gz")

    # Collect metadata
    metadata[dataset_name] = {
        'dataset_name': dataset_name,
        'train_size': len(list(tfds.as_numpy(train_set))),
        'validation_size': len(list(tfds.as_numpy(validation_set))),
        'test_size': len(list(tfds.as_numpy(test_set))),
        'num_features': len(tfds.as_numpy(train_set.take(1))[0]),
        'feature_names': ['input', 'target'],
        'data_types': [str(type(tfds.as_numpy(train_set.take(1))[0][0])), str(type(tfds.as_numpy(train_set.take(1))[0][1]))],
        'dataset_size_bytes': sum([len(example[0]) + len(example[1]) for example in tfds.as_numpy(train_set)]),
        'creation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'source_url': f"https://www.tensorflow.org/datasets/catalog/{dataset_name.replace('/', '_')}",
        'description': f"{dataset_name} dataset"
    }

# Save metadata to JSON file
with open('datasets/metadata.json', 'w') as f:
    json.dump(metadata, f, indent=4)