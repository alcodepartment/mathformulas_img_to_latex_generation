# 🧮 Image2LaTeX

This project turns images of mathematical formulas into LaTeX code.  
It uses a small custom CNN to read the image and a Transformer decoder to write the formula.  
Everything can be trained, tested, and shown in one notebook.

---

## 🚀 What it does

You upload or point to an image with a formula — the system cleans it up, normalizes the size, runs it through a CNN to find features, and then uses a Transformer to generate the LaTeX sequence step by step.  
In short: **you show it a picture → it gives you the LaTeX**.

---

## ⚙️ Installation

Clone the repo and install dependencies:

```bash
git clone <repo-url>
cd project
pip install -r requirements.txt
```

Make sure you have **Python ≥ 3.9** and **PyTorch + TorchVision** installed.

---

## 🧠 Usage

Prepare your dataset in the `data` folder:

```
data/
  train/
    images/*.png
    labels/*.txt
  val/
    images/*.png
    labels/*.txt
```

Then train and test everything inside the notebook:

```bash
jupyter notebook demonstration.ipynb
```

Or do it from the command line:

```bash
python run.py train --cfg cfg.yaml
python run.py eval  --cfg cfg.yaml
python run.py infer --cfg cfg.yaml --img data/val/images/sample.png
```

After training, all logs, checkpoints, and sample results appear in the `artifacts/` folder.

---

## 📁 Project Structure

```
project/
├─ data/            
├─ artifacts/       
├─ core/            
│   ├─ preprocessing.py
│   ├─ tokenizer.py
│   ├─ encoder.py
│   ├─ decoder.py
│   └─ model.py
├─ run.py           
├─ cfg.yaml         
└─ demonstration.ipynb  
```

---

## 🧩 How it works

The CNN reads the formula image and turns it into a sequence of features.  
The Transformer decoder predicts LaTeX tokens one after another until it finishes the expression.  
You can use **greedy decoding** (fast) or **beam search** (more accurate).  
The notebook visualizes the generated LaTeX right next to the original image.

---

## 📊 After training

Everything is saved automatically in `artifacts/`:

- **checkpoints/** — trained model weights  
- **logs/** — training history and loss curves  
- **samples/** — image → predicted LaTeX examples  

You can load the final checkpoint to run inference on new images anytime.

---

## 🧾 License

This project is open for educational and research use.  
Default license: **MIT**.
