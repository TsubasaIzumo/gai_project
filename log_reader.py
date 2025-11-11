from tensorboard.backend.event_processing import event_accumulator

# 加载 event 文件
ea = event_accumulator.EventAccumulator("lightning_logs/2025-11-07-02-18-14_formal_run_with_early_stopping/version_0/events.out.tfevents.1762453096.LAPTOP-GUVFPBMU.26892.0")
ea.Reload()  # 加载数据

# 查看记录了哪些 scalar 值
print("Available scalar tags:", ea.Tags()['scalars'])

# 例如获取 'train/loss' 的数据
loss_events = ea.Scalars('train/loss')

# 输出 loss 值
for event in loss_events:
    print(f"Step {event.step}, Loss = {event.value}")