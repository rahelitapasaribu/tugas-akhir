# Pedoman Anotasi Sentimen Komentar YouTube

## 1. Tujuan
Pedoman ini digunakan untuk menentukan label sentimen
pada setiap komentar YouTube dalam penelitian.

## 2. Kategori Sentimen

### Positif
Komentar yang menunjukkan dukungan, apresiasi, simpati,
antusiasme, atau penilaian positif yang jelas.

### Negatif
Komentar yang menunjukkan kritik, kemarahan, kekecewaan,
tuduhan, penolakan, atau sindiran negatif.

### Netral
Komentar yang berupa pertanyaan, informasi, pernyataan
faktual, rasa penasaran, atau tidak menunjukkan kecenderungan
positif maupun negatif yang jelas.

## 3. Aturan Anotasi
1. Setiap komentar hanya memiliki satu label dominan.
2. Penentuan label harus mempertimbangkan konteks kalimat.
3. Emoji tidak boleh menjadi satu-satunya dasar penentuan label.
4. Pertanyaan tidak otomatis dikategorikan negatif.
5. Sarkasme dinilai berdasarkan maksud kalimat.
6. Komentar tanpa kecenderungan sentimen yang jelas
   dikategorikan sebagai netral.
7. Jika terdapat beberapa sentimen, pilih sentimen yang dominan.
8. Penentuan sentimen didasarkan pada kecenderungan sikap atau evaluasi yang disampaikan terhadap objek pembahasan, bukan hanya berdasarkan kata-kata positif atau negatif yang muncul.
9. Komentar yang hanya menunjukkan ekspresi emosi tanpa kecenderungan evaluasi yang jelas terhadap objek pembahasan dikategorikan sebagai Netral.
10. Pertanyaan retoris yang mengandung kritik, tuduhan, atau sindiran negatif dikategorikan sebagai Negatif.
11. Komentar yang mengandung sentimen positif dan negatif secara bersamaan diberi label berdasarkan sentimen yang paling dominan sesuai maksud utama komentar.
12. Sarkasme atau pujian yang digunakan untuk menyampaikan ejekan dan kritik dikategorikan sebagai Negatif.
13. Seluruh komentar yang telah dikumpulkan tetap dipertahankan dalam dataset penelitian, termasuk komentar yang berupa spam, promosi, atau kurang relevan dengan topik pembahasan.
14. Jika maksud komentar tidak dapat ditentukan dari teks dan konteks yang tersedia, komentar dikategorikan sebagai Netral.
15. Komentar spam, promosi, atau kurang relevan tetap diberikan satu label sentimen berdasarkan kecenderungan sentimen yang terkandung dalam komentar, yaitu Positif, Negatif, atau Netral.
16. Kata-kata positif atau negatif yang muncul dalam komentar promosi tidak secara otomatis menentukan label sentimen. Penentuan label tetap mempertimbangkan maksud dan konteks komentar.
17. Komentar yang tidak menunjukkan kecenderungan sentimen yang jelas terhadap objek pembahasan dikategorikan sebagai Netral.
18. Setiap komentar hanya boleh menghasilkan satu label, yaitu Positif, Negatif, atau Netral.
19. Label harus ditulis secara konsisten sesuai kategori yang telah ditentukan.
20. Jika diminta menghasilkan keluaran terstruktur, model harus mengikuti format yang ditentukan tanpa menambahkan label di luar tiga kategori tersebut.

## 4. Prioritas Penentuan Label
- Apresiasi, dukungan, simpati, dan antusiasme: Positif.
- Kritik, kemarahan, tuduhan, dan sindiran negatif: Negatif.
- Pertanyaan, informasi, dan pernyataan tanpa evaluasi: Netral.

## 5. Validasi
Label hasil anotasi otomatis harus diperiksa secara manual.
Label manual yang telah divalidasi digunakan sebagai acuan
dalam penyusunan dataset penelitian.