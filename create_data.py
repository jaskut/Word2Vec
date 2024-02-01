import tensorflow_datasets as tfds
import tqdm

ds = tfds.load('huggingface:wikipedia/20220301.de', split='train')

st = b''
for item in tqdm.tqdm(ds.take(1000)):
  st += item['text'].numpy()

with open('dewiki.txt', 'wb') as f:
  f.write(st)