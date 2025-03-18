import tensorflow_datasets as tfds
import pandas as pd
import os
import gzip
import shutil

# Print available configurations for the ted_hrlr_translate dataset
builder = tfds.builder('ted_hrlr_translate')
print(f"Available configurations for ted_hrlr_translate: {builder.builder_configs.keys()}")

# Dataset names
datasets = ["wmt14_translate/fr-en", "wmt14_translate/de-en"]

# Create the datasets folder if it doesn't exist
os.makedirs("datasets", exist_ok=True)

for dataset_name in datasets:
    # Download and prepare the dataset
    dataset = tfds.load(dataset_name, as_supervised=True)

    # Print the available splits
    print(f"Available splits for {dataset_name}: {dataset.keys()}")

    # Extract train and validation sets
    train_set, validation_set = dataset['train'], dataset['validation']

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

    # Save the complete datasets in compressed format
    def save_compressed(dataset, filename, chunk_size=1000):
        with gzip.open(filename, 'wb') as f:
            for i in range(0, len(dataset), chunk_size):
                chunk = dataset.skip(i).take(chunk_size)
                for example in tfds.as_numpy(chunk):
                    f.write(f"{example[0].decode('utf-8')}\t{example[1].decode('utf-8')}\n".encode('utf-8'))

    save_compressed(train_set, f"datasets/{dataset_name.split('/')[-1]}_train.txt.gz")
    save_compressed(validation_set, f"datasets/{dataset_name.split('/')[-1]}_validation.txt.gz")