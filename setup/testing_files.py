import gzip

def read_gz_file(filepath):
    with gzip.open(filepath, 'rt', encoding='utf-8') as f:
        for line in f:
            print(line.strip())

# Example usage
read_gz_file('datasets/fr-en_train.txt.gz')