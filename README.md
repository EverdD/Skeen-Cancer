# Skin Cancer Classification with Xception

Aplikasi klasifikasi citra kanker kulit berbasis **Convolutional Neural Network (CNN)** dengan arsitektur **Xception** menggunakan metode transfer learning.  
Antarmuka dibangun menggunakan **Streamlit**, sehingga dapat dijalankan langsung melalui browser.

---

## 🚀 Instalasi & Cara Menjalankan

```bash
# 1. Clone Repository
git clone https://github.com/EverdD/Skeen-Cancer.git
cd Skeen-Cancer

# 2. Cek Versi Python
python --version
# Gunakan Python 3.10 atau lebih baru (disarankan Python 3.11.5).
# Jika belum ada, download di https://www.python.org/downloads/
# Pastikan saat instalasi centang "Add Python to PATH".

# 3. (Opsional) Buat Virtual Environment
python -m venv env

# Aktivasi environment
source env/bin/activate   # macOS/Linux
.\env\Scripts\activate    # Windows

# 4. Install dependencies
pip install -r requirements.txt

# 5. Jalankan aplikasi
streamlit run app.py
