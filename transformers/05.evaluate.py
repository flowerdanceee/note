import evaluate

# 拿到一个新metric，看 key 以及 Value 的数据类型（string / int / sequence）
# 再写compute
metric = evaluate.load("accuracy")
print(metric.features)

# 全局计算
results = metric.compute(references=[0, 1, 2, 0, 1, 2], predictions=[0, 1, 1, 2, 1, 0])
print(results)
# 有些metric会保存以往的记录，因此需要reset。
# metric.reset()

# batch计算
for refs, preds in zip([[1, 0], [0, 1]], [[1, 0], [0, 1]]):
    metric.add_batch(references=refs, predictions=preds)
results = metric.compute()
print(results)
# metric.reset()

# 多个指标并用：
clf_metrics = evaluate.combine(["accuracy", "f1", "recall", "precision"])
print(clf_metrics.compute(predictions=[0, 1, 0], references=[0, 1, 1]))

# 从d中筛选keep_keys后组成新字典
# {k: v for k, v in d.items() if k in keep_keys}
# # 合并字典
# d.update(other)
