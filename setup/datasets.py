import tensorflow_datasets as tfds
import pandas as pd
import os
import gzip
import shutil

# Dataset names
datasets = ["ted_hrlr_translate/pt_to_en", "ted_hrlr_translate/fr_to_en"]

# Create the datasets folder if it doesn't exist
os.makedirs("datasets", exist_ok=True)

for dataset_name in datasets:
    # Download and prepare the dataset
    dataset = tfds.load(dataset_name, as_supervised=True, split=['train', 'test'])

    # Extract train and test sets
    train_set, test_set = dataset['train'], dataset['test']

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
    to_csv(test_set, f"datasets/{dataset_name.split('/')[-1]}_test.csv")

    # Save the complete datasets in compressed format
    def save_compressed(dataset, filename):
        with gzip.open(filename, 'wb') as f:
            for example in tfds.as_numpy(dataset):
                f.write(f"{example[0].decode('utf-8')}\t{example[1].decode('utf-8')}\n".encode('utf-8'))

    save_compressed(train_set, f"datasets/{dataset_name.split('/')[-1]}_train.txt.gz")
    save_compressed(test_set, f"datasets/{dataset_name.split('/')[-1]}_test.txt.gz")