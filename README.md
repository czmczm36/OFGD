# OFGD: Object-Focused Guided Decoding for Hallucination Reduction in LVLMs

Official implementation of the paper:

> **OFGD: Object-Focused Guided Decoding for Hallucination Reduction in Large Vision-Language Models (LVLMs)**
> *NeurIPS 2026 (Under Review)*

---

## 🔍 Overview

Large Vision-Language Models (LVLMs) often suffer from **hallucination**, generating objects that do not exist in the image. This issue is particularly severe in **multi-subject scenes**.

We propose **OFGD**, a **subject-centric structured decoding framework** that improves visual grounding without modifying model parameters.

### 🚀 Key Ideas

* **Subject Selection & Ranking** (YOLO-based)
* **Subject-to-Patch Mapping**
* **Subject-Guided Region Expansion (core contribution)**
* **Memory-aware Sequential Captioning**
* **Text-only Fusion (no second image pass)**

---

## 🧠 Framework

Our pipeline consists of five stages:

1. Subject Selection and Ranking
2. Subject-to-Patch Mapping
3. Subject-Guided Region Expansion
4. Subject Caption Generation with Memory
5. Multi-Subject Fusion

---

## ⚙️ Installation

### 1. Clone repository

```bash
git clone https://github.com/czmczm36/OFGD.git
cd OFGD
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 📦 Dependencies

* Python 3.9+
* PyTorch
* Transformers
* LLaVA
* YOLOv8 (Ultralytics)

---

## ▶️ Usage

### Run multi-subject captioning

```bash
python run_multi_subject_caption.py \
    --config config_multi_subject.json
```

---

## 📁 Project Structure

```bash
OFGD/
├── run_multi_subject_caption.py        # Main entry
├── multi_subject_caption_wrapper.py   # Core pipeline wrapper
├── subject_ranker_two.py              # Subject ranking module
├── config_multi_subject.json          # Config file
├── requirements.txt
```

---

## 🔧 Configuration

Main parameters can be modified in:

```bash
config_multi_subject.json
```

Key settings include:

* Number of subjects (K)
* Detection thresholds
* Generation parameters
* Expansion strategy

---

## 🧪 Supported Models

Current implementation supports:

* LLaVA-v1.5-7B
* LLaVA-v1.5-13B

(Other LVLMs such as InstructBLIP can be added with minor changes.)

---

## 📊 Evaluation

The method is evaluated using:

* **CHAIR** (object-level hallucination)
* **POPE** (binary hallucination detection)
* Precision / Recall / F1

---

## 📌 Notes

* This is a **plug-and-play inference framework**
* No model training required
* Compatible with different LVLM backbones

---

## ⚠️ Limitations

* Depends on object detection quality (YOLO)
* Sequential generation may reduce global fluency in complex scenes

---

## 📜 License

This project follows the licenses of its dependencies (LLaVA, YOLO, etc.).
Code will be released under an open-source license after publication.

---

## 🙏 Acknowledgements

We thank the authors of:

* LLaVA
* InstructBLIP
* YOLOv8
* CHAIR / POPE benchmarks

---

## ⭐ Citation

If you find this work useful, please cite:

```bibtex
@article{ofgd2026,
  title={OFGD: Object-Focused Guided Decoding for Hallucination Reduction in LVLMs},
  author={Anonymous},
  journal={NeurIPS},
  year={2026}
}
```

---

## 📬 Contact

For questions, feel free to open an issue.
