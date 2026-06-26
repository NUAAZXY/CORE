from datasets import load_dataset

ds = load_dataset('THUDM/LongBench','repobench-p')['test']
print(ds)
# 指定一个本地目录，比如 "./data/LongBench/narrativeqa_test"
ds.save_to_disk("datasets/LongBench/repobench-p")
