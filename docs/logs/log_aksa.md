# Log Interaksi LLM Kelompok KAJA: Aksa (Data Pipeline & Preprocessing Engineer)


Urutan kronologis log interaksi LLM pada branch `data`.


## Log 1: perancangan dan pembuatan awal data/dataset.py (A1)

<!-- Tempel log & tautan share chat sebelumnya tempat data/dataset.py (commit 90b4eb7) dibuat jika berasal dari sesi terpisah; hapus bagian ini jika tidak ada sesi terpisah. -->

````
LOG INTERAKSI LLM - KELOMPOK KAJA
- Tanggal Sesi: 25/11/2026
- Anggota/Peran: Aksa / Data Pipeline & Preprocessing Engineer
- Target Modul: data/dataset.py (pembuatan awal, commit 90b4eb7)
- Tautan Sesi Chat: [ISI TAUTAN SHARE CHAT PEMBUATAN data/dataset.py JIKA TERPISAH]

1. TAHAP INISIASI
   - Prompt Kelompok: [ISI KUTIPAN PROMPT SESI SEBELUMNYA JIKA TERPISAH]
   - Alasan Khusus: Saya menggunakan file instruksi peran (`prompt_aksa.md`) dari ketua agar isi prompt dan kontrak antarmuka (`get_dataloaders`, `denormalize`, shape `[B, 1, 28, 28]` dalam `[-1, 1]`) seragam dengan seluruh anggota kelompok. `data/dataset.py` dikerjakan paling awal karena modul U-Net, training diffusion, dan evaluasi seluruh tim bergantung pada pipeline data ini.
   - Output Model: Dihasilkan `data/dataset.py` yang mengunduh Fashion-MNIST ke `cfg.data_root`, menormalisasi piksel ke `[-1, 1]` dengan mean 0.5 dan std 0.5, membangun `train_loader` (`shuffle=True`, `drop_last=True`, `pin_memory=True`, seed dari `cfg.seed`) dan `test_loader`, menyediakan `denormalize(x)` ke `[0, 1]`, serta skrip pemeriksaan yang mencetak statistik batch dan menyimpan grid ke `runs/data_check.png`.

2. TAHAP PENANGANAN EROR / DEBUGGING
   Tidak ada eror tercatat pada sesi ini.

3. TAHAP OPTIMASI
   Tidak ada.
````


## Log 2: setup repo/environment dan verifikasi data/dataset.py + runs/data_check.png (A1)

````
LOG INTERAKSI LLM - KELOMPOK KAJA
- Tanggal Sesi: 25/11/2026
- Anggota/Peran: Aksa / Data Pipeline & Preprocessing Engineer
- Target Modul: setup repo/environment, verifikasi data/dataset.py + runs/data_check.png
- Tautan Sesi Chat: https://github.com/JordanCodeGit/ddpm-fashion/blob/data/docs/logs/transcript_aksa.md (sesi Google Antigravity lokal, diekspor ke transkrip di repo)

1. TAHAP INISIASI
   - Prompt Kelompok:
     "Setup repo & environment (clone github.com/JordanCodeGit/ddpm-fashion ke branch data dan pasang dependency),"
     Prompt lanjutan:
     "Lanjut"
   - Alasan Khusus: Saya memakai file instruksi khusus (`prompt_aksa.md`) dari ketua untuk menyamakan isi prompt dan kontrak antarmuka agar output LLM sesuai kebutuhan kelompok. Sebelumnya saya sempat mengalami kendala saat mencoba di Google Colab, sehingga saya memutuskan mengerjakan seluruh modul secara lokal di VS Code dengan clone repo GitHub dan virtual environment `.venv` sendiri (memakai PyTorch CPU tanpa mengubah `requirements.txt` milik Jordan karena laptop tidak memiliki GPU NVIDIA). Untuk `data/dataset.py`, saya mempertahankan normalisasi `[-1, 1]` agar distribusi gambar simetris terhadap noise Gaussian standar `N(0, I)` yang dipakai DDPM (berbeda dengan `[0, 1]` yang berat ke positif atau statistik ImageNet yang ditujukan untuk foto RGB). Saya juga mempertahankan pilihan tanpa augmentasi: dari visualisasi `runs/data_check.png` terlihat seluruh alas kaki (Sandal, Sneaker, Ankle boot) menghadap ke kiri sehingga horizontal flip justru mengubah distribusi orientasi asli, sementara rotasi/crop pada resolusi kecil 28x28 berisiko memotong bentuk pakaian.
   - Output Model: Repo `ddpm-fashion` di-clone, branch lokal `data` dibuat dan disinkronkan fast-forward dengan `main` terbaru (`90b4eb7`), lalu dipasang virtual environment `.venv` dengan PyTorch CPU (`torch 2.15.0+cpu`, `torchvision 0.29.2+cpu`, `numpy 2.5.5`, `matplotlib 3.11.2`, `pandas 3.11.2`, `tqdm 4.70.1`) tanpa mengubah `requirements.txt`. Modul `config.py`, `models/embeddings.py`, dan `models/unet.py` lulus tes di CPU. Selanjutnya `python -m data.dataset` dijalankan: Fashion-MNIST terunduh ke `data_cache/`, semua assert kontrak lulus (`x=(128, 1, 28, 28)` float32 `[-1.0000, 1.0000]`, `y=(128,)` int64 `0..9`, train `468` batch dengan `drop_last=True`, test `79` batch), dan grid 10x10 tersimpan ke `runs/data_check.png`.
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
   - Analisis & Solusi Kelompok: Eror muncul bukan karena branch `data` tidak ada, melainkan karena di root repo sudah terdapat folder bernama `data/` sehingga Git bingung apakah argumen `data` merujuk ke path folder atau ke nama branch remote `origin/data`. Solusinya adalah membuat branch lokal secara eksplisit terhadap remote dengan `git checkout -b data origin/data`, lalu menyinkronkannya ke commit terbaru di `main` dengan `git merge --ff-only main`. Pelajaran penting: bila nama branch sama dengan nama folder di repo, perintah `git checkout` harus menyebutkan tracking branch secara eksplisit.
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
- Tautan Sesi Chat: https://github.com/JordanCodeGit/ddpm-fashion/blob/data/docs/logs/transcript_aksa.md (sesi Google Antigravity lokal, diekspor ke transkrip di repo)

1. TAHAP INISIASI
   - Prompt Kelompok: "Lanjut sub tugas 2"
   - Alasan Khusus: Karena aturan proyek melarang penggunaan bobot pretrained, kelompok membutuhkan classifier CNN kecil yang dilatih dari nol pada Fashion-MNIST untuk menilai apakah sampel hasil generate Conditional DDPM sesuai dengan kelas yang diminta. Classifier harus menggunakan `get_dataloaders(cfg)` dengan skala input `[-1, 1]` yang persis sama dengan output DDPM agar tidak terjadi perbedaan distribusi input saat evaluasi. Arsitektur ringkas (4 layer konvolusi, ~197 ribu parameter) dengan `AdaptiveAvgPool2d((2, 2))` dipilih supaya cukup ekspresif mempertahankan posisi spasial pakaian (terutama membedakan Shirt, T-shirt/top, Pullover, dan Coat) sekaligus cepat dilatih ulang di laptop Jordan (RTX 4060) karena file `eval/classifier.pt` tidak ikut di-commit ke Git.
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
   Tidak ada. Akurasi test langsung melampaui target 90% sejak epoch ke-3 (`90.26%`) dan mencapai `92.88%` pada epoch ke-10 tanpa perlu penyesuaian tambahan.
````


## Log 4: eval/evaluate.py dan uji kewajaran dengan stub (A3)

````
LOG INTERAKSI LLM - KELOMPOK KAJA
- Tanggal Sesi: 25/11/2026
- Anggota/Peran: Aksa / Data Pipeline & Preprocessing Engineer
- Target Modul: eval/evaluate.py
- Tautan Sesi Chat: https://github.com/JordanCodeGit/ddpm-fashion/blob/data/docs/logs/transcript_aksa.md (sesi Google Antigravity lokal, diekspor ke transkrip di repo)

1. TAHAP INISIASI
   - Prompt Kelompok: "Lanjut sub tugas 3"
   - Alasan Khusus: Skrip evaluasi harus siap sebelum training final Axan selesai agar integrasi di Gerbang G3 tidak terhambat. Karena `diffusion/ddpm.py` milik Axan belum tersedia di repo, saya meminta mode stub (`--use_stub`) yang mengembalikan gambar test set asli sesuai label target untuk uji kewajaran: jika akurasi pada stub test set penuh cocok dengan `classifier_test_acc`, artinya urutan label, normalisasi, batching, dan perhitungan confusion matrix di pipeline evaluasi bebas bug. Selain itu, sampling dipecah per chunk sebesar `cfg.batch_size` agar aman terhadap batas 8 GB VRAM di RTX 4060, memakai `UNet(cfg)` asli dari `models/unet.py`, mengikuti pola `parse_known_args` + `--from_json` sesuai kontrak `config.py`, serta tidak menggunakan library di luar daftar yang diizinkan.
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
   Tidak ada. Folder hasil uji sementara `runs/stub_test/` dihapus setelah verifikasi agar tidak terbaca oleh `scripts/make_figures.py` saat mengolah `runs/*/eval.json`.
````


## Log 5: evaluasi checkpoint final (A4 — menunggu Gerbang G3)

<!-- Lengkapi log ini setelah `diffusion/ddpm.py` dan `runs/<run_name>/ckpt_ema.pt` tersedia lalu `eval/evaluate.py` dijalankan pada checkpoint final. -->
