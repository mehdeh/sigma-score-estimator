# بهینه‌سازی‌های Performance برای GPU های High-End (A100, H100, etc.)

## خلاصه تغییرات

این فایل توضیح می‌دهد که چه بهینه‌سازی‌هایی برای بهبود سرعت training روی GPU های قدرتمند مثل RTX A100 اعمال شده است.

## مشکلات اصلی که برطرف شدند

### 1. ❌ **cuDNN Benchmark = False** (بزرگترین مشکل)
**قبل:**
```python
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
```

**بعد:**
```python
torch.backends.cudnn.benchmark = True  # ✅ بهبود قابل توجه performance
torch.backends.cudnn.deterministic = False
```

**تاثیر:** 20-50% سرعت بیشتر در convolution operations
- این تنظیم به cuDNN اجازه می‌دهد بهترین الگوریتم convolution را برای hardware شما پیدا کند
- برای GPU های قدرتمند مثل A100 که الگوریتم‌های optimized زیادی دارند، تاثیر بسیار زیادی دارد

---

### 2. ✅ **Automatic Mixed Precision (AMP)** اضافه شد

**قابلیت جدید در trainer:**
- استفاده از `torch.cuda.amp.autocast()` برای forward pass
- استفاده از `GradScaler` برای backward pass
- به صورت خودکار از FP16 برای محاسبات استفاده می‌کند (جایی که امن است)

**تاثیر:** 2-3x سرعت بیشتر روی A100
- A100 و GPU های Ampere/Hopper برای FP16/TF32 compute بهینه شده‌اند
- مصرف memory کمتر → می‌توان batch size بزرگتری استفاده کرد

**فعال‌سازی در config:**
```yaml
training:
  use_amp: true  # پیش‌فرض: true
```

---

### 3. ⚡ **DataLoader Optimizations**

**تغییرات:**
```python
# قبل
num_workers=4

# بعد
num_workers=8
persistent_workers=True  # workers بین epoch ها زنده می‌مانند
prefetch_factor=2  # 2 batch از قبل load می‌شود
```

**تاثیر:** کاهش data loading bottleneck
- `persistent_workers` overhead راه‌اندازی worker ها را بین epoch ها حذف می‌کند
- `prefetch_factor` اطمینان می‌دهد GPU همیشه data برای process کردن دارد

---

### 4. 🚀 **Non-blocking Data Transfer**

**قبل:**
```python
images = images.to(self.device)
```

**بعد:**
```python
images = images.to(self.device, non_blocking=True)
```

**تاثیر:** GPU و CPU می‌توانند به صورت asynchronous کار کنند
- Data transfer به GPU دیگر باعث block شدن CPU نمی‌شود
- پیپ لاین training روان‌تر می‌شود

---

### 5. 📦 **Batch Size افزایش یافت**

**قبل:** `batch_size: 128`  
**بعد:** `batch_size: 256`

**توصیه برای GPU های مختلف:**
- **RTX 2080 Ti (11GB):** 128-192
- **RTX 3090 (24GB):** 256-384
- **RTX A100 (40GB):** 256-512
- **A100 (80GB):** 512-1024

**تاثیر:** GPU utilization بهتر و throughput بیشتر

---

### 6. ⚡ **Optimizer Zero Grad Optimization**

**قبل:**
```python
self.optimizer.zero_grad()
```

**بعد:**
```python
self.optimizer.zero_grad(set_to_none=True)
```

**تاثیر:** کمی سریعتر و مصرف memory کمتر

---

### 7. 🔬 **PyTorch 2.0 Compile Support** (اختیاری)

**قابلیت جدید:**
```yaml
training:
  use_compile: false  # می‌توانید آن را true کنید برای speedup بیشتر
```

اگر PyTorch 2.0+ دارید، می‌توانید `use_compile: true` کنید برای 10-30% سرعت بیشتر.

---

## نتیجه انتظاری

با این تغییرات، سرعت training روی **A100** باید **3-5x سریعتر** از قبل باشد:

### قبل (تنظیمات قدیمی):
- 100 epoch: ~30 دقیقه روی A100
- 100 epoch: ~30 دقیقه روی RTX 2080 Ti
- **مشکل:** A100 و 2080 Ti تقریباً یکسان بودند! ❌

### بعد (تنظیمات جدید):
- 100 epoch: **~6-10 دقیقه** روی A100 ✅
- 100 epoch: ~15-20 دقیقه روی RTX 2080 Ti
- **A100 حالا 2-3x سریعتر از 2080 Ti است** ✅

---

## استفاده

### با تنظیمات پیش‌فرض (توصیه می‌شود):
```bash
python main.py train
```

### تنظیمات دستی برای GPU های مختلف:

#### برای A100 (40GB):
```bash
python main.py train --batch-size 512 --epochs 100
```

#### برای RTX 2080 Ti (11GB):
```bash
python main.py train --batch-size 128 --epochs 100
```

#### غیرفعال کردن AMP (اگر مشکلی پیش آمد):
```bash
# ایجاد یک config file موقت
cp config/default.yaml config/no_amp.yaml
# ویرایش: use_amp: false
python main.py train --config config/no_amp.yaml
```

---

## بررسی GPU Utilization

برای اطمینان از اینکه GPU به درستی استفاده می‌شود، در یک terminal جداگانه اجرا کنید:

```bash
watch -n 0.5 nvidia-smi
```

**چیزهایی که باید ببینید:**
- **GPU Utilization:** 95-100% (خوب ✅)
- **Memory Usage:** >50% از total memory در حال استفاده
- **Power Usage:** نزدیک به maximum TDP

اگر GPU Utilization کمتر از 80% است، ممکن است:
- `num_workers` را افزایش دهید (8 → 12 یا 16)
- `batch_size` را افزایش دهید
- `prefetch_factor` را افزایش دهید (2 → 4)

---

## Troubleshooting

### خطای Out of Memory (OOM)
```bash
# batch_size را کاهش دهید
python main.py train --batch-size 128
```

### Training خیلی کند است (هنوز)
1. بررسی کنید `nvidia-smi` چه می‌گوید
2. مطمئن شوید `use_amp: true` است
3. مطمئن شوید `cudnn.benchmark = True` است
4. `num_workers` را افزایش دهید

### NaN Loss یا Training Divergence
```bash
# AMP را غیرفعال کنید
# یا learning rate را کاهش دهید
python main.py train --learning-rate 0.0005
```

---

## تغییرات Technical

### فایل‌های تغییر یافته:
1. `main.py` - cudnn.benchmark = True
2. `src/training/trainer.py` - AMP support, non-blocking transfers
3. `src/datasets/dataset.py` - persistent_workers, prefetch_factor
4. `config/default.yaml` - batch_size, num_workers, use_amp

### کد قابل استفاده مجدد برای پروژه‌های دیگر:
اگر می‌خواهید این optimizations را در پروژه‌های دیگر استفاده کنید:

```python
# در train loop خود
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for images, labels in dataloader:
    images = images.to(device, non_blocking=True)
    
    with autocast():
        output = model(images)
        loss = criterion(output, labels)
    
    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

---

## مراجع و منابع بیشتر

- [PyTorch AMP Documentation](https://pytorch.org/docs/stable/amp.html)
- [PyTorch Performance Tuning Guide](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
- [NVIDIA A100 Best Practices](https://docs.nvidia.com/deeplearning/performance/)

---

**توجه:** این optimizations برای هر مدل و dataset متفاوت است. ممکن است نیاز به fine-tuning داشته باشید.

برای سوالات بیشتر، issues را در GitHub repository باز کنید.


