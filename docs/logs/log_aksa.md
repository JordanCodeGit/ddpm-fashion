# Log Interaksi LLM Kelompok KAJA: Aksa (Data Pipeline & Preprocessing Engineer)


Urutan kronologis log interaksi LLM pada branch `data`. Kolom `[DIISI ANGGOTA]` wajib diisi sendiri oleh Aksa (penalaran asli anggota & tautan share chat) sebelum dikumpulkan ke Hizkia.


## Log 1: perancangan dan pembuatan awal data/dataset.py (A1)

<!-- [DIISI ANGGOTA]: Jika data/dataset.py (commit 90b4eb7) dibuat pada sesi chat terpisah sebelumnya, tempel blok LOG lengkap dari sesi tersebut di sini beserta share link-nya. Jika tidak, hapus bagian ini dan sesuaikan penomoran log. -->

````
LOG INTERAKSI LLM - KELOMPOK KAJA
- Tanggal Sesi: [DIISI ANGGOTA]
- Anggota/Peran: Aksa / Data Pipeline & Preprocessing Engineer
- Target Modul: data/dataset.py (pembuatan awal)
- Tautan Sesi Chat: [DIISI ANGGOTA]

1. TAHAP INISIASI
   - Prompt Kelompok: [DIISI ANGGOTA]
   - Alasan Khusus: [DIISI ANGGOTA]
   - Output Model: [DIISI ANGGOTA]

2. TAHAP PENANGANAN EROR / DEBUGGING
   [DIISI ANGGOTA]

3. TAHAP OPTIMASI
   [DIISI ANGGOTA]
````


## Log 2: setup repo/environment dan verifikasi data/dataset.py + runs/data_check.png (A1)

````
LOG INTERAKSI LLM - KELOMPOK KAJA
- Tanggal Sesi: 25/11/2026
- Anggota/Peran: Aksa / Data Pipeline & Preprocessing Engineer
- Target Modul: setup repo/environment, verifikasi data/dataset.py + runs/data_check.png
- Tautan Sesi Chat: [DIISI ANGGOTA]

1. TAHAP INISIASI
   - Prompt Kelompok:
     "Setup repo & environment (clone github.com/JordanCodeGit/ddpm-fashion ke branch data dan pasang dependency),"
     Prompt lanjutan:
     "Lanjut"
   - Alasan Khusus: [DIISI ANGGOTA]
   - Output Model: Repo `ddpm-fashion` di-clone, branch lokal `data` dibuat dan disinkronkan fast-forward dengan `main` terbaru (`90b4eb7`), lalu dipasang virtual environment `.venv` dengan PyTorch CPU (`torch 2.15.0+cpu`, `torchvision 0.29.2+cpu`, `numpy 2.5.5`, `matplotlib 3.11.2`, `pandas 3.1.1`, `tqdm 4.70.1`) tanpa mengubah `requirements.txt`. Modul `config.py`, `models/embeddings.py`, dan `models/unet.py` lulus tes di CPU. Selanjutnya `python -m data.dataset` dijalankan: Fashion-MNIST terunduh ke `data_cache/`, semua assert kontrak lulus (`x=(128, 1, 28, 28)` float32 `[-1.0000, 1.0000]`, `y=(128,)` int64 `0..9`, train `468` batch dengan `drop_last=True`, test `79` batch), dan grid 10x10 tersimpan ke `runs/data_check.png`.
     Cuplikan kode krusial (`data/dataset.py`, verifikasi kontrak batch dan denormalisasi):
     ```python
     assert x.shape == (cfg.batch_size, cfg.in_channels, cfg.img_size, cfg.img_size), "shape x salah"
     assert x.dtype == torch.float32, "dtype x harus float32"
     assert x.min() >= -1.0 and x.max() <= 1.0, "x di luar rentang [-1, 1]"
     assert y.dtype == torch.int64, "dtype y harus int64"
     assert 0 <= y.min().item() and y.max().item() < cfg.num_classes, "label di luar 0..9"
     d = denormalize(x)
     assert d.min() >= 0.0 and d.max() <= 1.0, "denormalize di luar [0, 1]"
     ```

2. TAHAP PENANGANAN EROR / DEBUGGING
   Eror 1: disambiguasi nama branch `data` saat checkout
   - Pesan Eror dari Notebook/Konsol:
     "fatal: 'data' could be both a local file and a tracking branch.
     Please use -- (and optionally --no-guess) to disambiguate"
   - Prompt Perbaikan ke LLM: (terjadi dan langsung ditangani saat menjalankan perintah setup dari prompt inisiasi)
   - Analisis & Solusi Kelompok: [DIISI ANGGOTA]
   - Catatan LLM: Akar masalahnya adalah folder `data/` di root repo bernama sama dengan remote branch `origin/data`, sehingga `git checkout data` bersifat ambigu. Perbaikannya menggunakan `git checkout -b data origin/data` diikuti `git merge --ff-only main`.

3. TAHAP OPTIMASI
   Tidak ada.
````


## Log 3: eval/classifier.py (A2)

````
LOG INTERAKSI LLM - KELOMPOK KAJA
- Tanggal Sesi: 25/11/2026
- Anggota/Peran: Aksa / Data Pipeline & Preprocessing Engineer
- Target Modul: eval/classifier.py
- Tautan Sesi Chat: [DIISI ANGGOTA]

1. TAHAP INISIASI
   - Prompt Kelompok: "Lanjut sub tugas 2"
   - Alasan Khusus: [DIISI ANGGOTA]
   - Output Model: LLM menulis `eval/classifier.py` berisi `FashionCNN` dari nol (4 layer konvolusi + BatchNorm2d + ReLU dengan channel `(32, 64, 128)`, `AdaptiveAvgPool2d((2, 2))`, dan head MLP dengan `cfg.dropout`; total 196.842 parameter) yang memakai `get_dataloaders(cfg)` berskala `[-1, 1]`, optimizer AdamW (`--clf_lr 1e-3`, `--weight_decay 1e-4`), `CosineAnnealingLR`, gradient clipping `cfg.grad_clip`, serta helper `load_classifier`. Pelatihan 10 epoch di CPU (`python -m eval.classifier --device cpu`, total 1153.5s) mencapai best test accuracy `92.88%` (`test_loss=0.2086`, kelas terendah `Shirt` `77.00%` dan tertinggi `Bag` `98.80%`), menyimpan bobot terbaik ke `eval/classifier.pt` (799 KB), dan verifikasi `load_classifier` lulus.
     Cuplikan kode krusial (`eval/classifier.py`, `FashionCNN.forward`):
     ```python
     def forward(self, x: Tensor) -> Tensor:
         """Petakan gambar float32 [B, C, H, W] dalam [-1, 1] ke logits [B, num_classes]."""
         h = self.stage1(x)  # [B, C, 28, 28] -> [B, c1, 14, 14]
         h = self.stage2(h)  # [B, c1, 14, 14] -> [B, c2, 7, 7]
         h = self.stage3(h)  # [B, c2, 7, 7] -> [B, c3, 2, 2]
         return self.head(h)  # [B, c3, 2, 2] -> [B, num_classes]
     ```

2. TAHAP PENANGANAN EROR / DEBUGGING
   Tidak ada eror.

3. TAHAP OPTIMASI
   Tidak ada.
````


## Log 4: eval/evaluate.py dan uji kewajaran dengan stub (A3)

````
LOG INTERAKSI LLM - KELOMPOK KAJA
- Tanggal Sesi: 25/11/2026
- Anggota/Peran: Aksa / Data Pipeline & Preprocessing Engineer
- Target Modul: eval/evaluate.py
- Tautan Sesi Chat: [DIISI ANGGOTA]

1. TAHAP INISIASI
   - Prompt Kelompok: "Lanjut sub tugas 3"
   - Alasan Khusus: [DIISI ANGGOTA]
   - Output Model: LLM menulis `eval/evaluate.py` memakai pola `parse_known_args` -> `get_config(rest)`, membangun `UNet(cfg)` asli dari `models/unet.py`, memuat classifier via `load_classifier`, melakukan sampling dalam chunk `cfg.batch_size`, memvalidasi kontrak `guidance_scale` dan output tensor `[-1, 1]`, serta menulis `eval.json` (`overall_acc`, `per_class_acc`, `classifier_test_acc`) dan `confusion.png` di `cfg.run_dir`. Selama `diffusion/ddpm.py` belum ada, disertakan `StubTestSetDiffusion` (`--use_stub`) yang mengembalikan gambar test set asli. Uji kewajaran dengan `n_per_class=1000` menghasilkan `overall_acc=92.88%` (selisih `0.00 pp` dari `classifier_test_acc=92.88%`), uji default `--n_per_class 100 --from_json` menghasilkan `overall_acc=93.60%` (selisih `0.72 pp`), dan uji validasi kontrak `guidance_scale` (`w=-1.0`, `w=2.0` saat `cond_drop_prob=0.0`) ditolak sesuai spesifikasi.
     Cuplikan kode krusial (`eval/evaluate.py`, `generate_and_classify`):
     ```python
     y_all = torch.arange(cfg.num_classes, dtype=torch.int64).repeat_interleave(n_per_class)  # [N]
     total = y_all.size(0)
     preds_chunks: list[Tensor] = []
     classifier.eval()
     for start in tqdm(range(0, total, cfg.batch_size), desc="Sampling & evaluasi"):
         y_chunk = y_all[start : start + cfg.batch_size].to(device)  # [B]
         x_gen = sampler.sample(y_chunk, guidance_scale=guidance_scale)  # [B, 1, 28, 28]
         _check_generated_batch(x_gen, y_chunk.size(0), cfg)
         logits = classifier(x_gen.to(device))  # [B, num_classes]
         preds_chunks.append(logits.argmax(dim=1).cpu())  # [B]
     return y_all, torch.cat(preds_chunks, dim=0)  # [N]
     ```

2. TAHAP PENANGANAN EROR / DEBUGGING
   Tidak ada eror.

3. TAHAP OPTIMASI
   Tidak ada.
````


## Log 5: evaluasi checkpoint final (A4 — menunggu Gerbang G3)

<!-- [DIISI SETELAH EVALUASI FINAL]: Lengkapi log ini setelah `diffusion/ddpm.py` dan `runs/<run_name>/ckpt_ema.pt` tersedia lalu `eval/evaluate.py` dijalankan pada checkpoint final. -->
