import os
from flask import Flask, render_template, request
from transformers import pipeline
import tensorflow as tf

# Mengurangi log TensorFlow dan menonaktifkan oneDNN jika diperlukan
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Menyembunyikan INFO dan WARNING dari TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Nonaktifkan oneDNN untuk hasil yang konsisten

# Mengurangi level log TensorFlow agar hanya menampilkan error saja
tf.get_logger().setLevel('ERROR')

# Inisialisasi aplikasi Flask
app = Flask(__name__)

# Inisialisasi model parafrase
paraphraser = pipeline("text2text-generation", model="t5-base")

# Halaman utama dengan formulir input teks
@app.route("/", methods=["GET", "POST"])
def home():
    paraphrased_text = ""
    if request.method == "POST":
        original_text = request.form["original_text"]
        result = paraphraser(original_text, max_length=50, num_return_sequences=1)
        paraphrased_text = result[0]['generated_text']
    return render_template("index.html", paraphrased_text=paraphrased_text)

if __name__ == "_main_":
    app.run(debug=True)