"""Export fine-tuned model WITH classification head to ONNX."""

import json
from pathlib import Path

import numpy as np


def main():
    from optimum.onnxruntime import ORTModelForSequenceClassification
    from transformers import AutoTokenizer

    model_path = "training/scripts/models/all-MiniLM-L6-v2/final"
    output_dir = Path("training/models/onnx/classifier")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Exporting model with classification head to ONNX...")
    model = ORTModelForSequenceClassification.from_pretrained(model_path, export=True)
    model.save_pretrained(str(output_dir))

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.save_pretrained(str(output_dir))

    # Show exported files
    for f in sorted(output_dir.iterdir()):
        size = f.stat().st_size
        print(f"  {f.name}: {size / 1024 / 1024:.1f} MB" if size > 10000 else f"  {f.name}: {size} bytes")

    # Test
    import onnxruntime as ort

    onnx_file = output_dir / "model.onnx"
    session = ort.InferenceSession(str(onnx_file), providers=["CPUExecutionProvider"])
    print(f"\nOutputs: {[o.name for o in session.get_outputs()]}")

    with open(output_dir / "config.json") as f:
        config = json.load(f)
    id2label = config.get("id2label", {})

    tests = [
        ("Hola, como estas?", "chitchat"),
        ("Quiero ser CTO en 5 anos", "aspirational"),
        ("Tengo 10 anos de experiencia en Python", "professional"),
        ("Me siento frustrado con mi trabajo", "emotional"),
        ("Que es POP Skills?", "rag_query"),
        ("Como puedo mejorar mis habilidades de liderazgo?", "learning"),
        ("Receta de pizza", "off_topic"),
    ]

    print("\nClassification results:")
    correct = 0
    for text, expected in tests:
        inputs = tokenizer(text, return_tensors="np", padding=True, truncation=True, max_length=128)
        feed = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "token_type_ids": inputs.get("token_type_ids", np.zeros_like(inputs["input_ids"])),
        }
        outputs = session.run(None, feed)
        logits = outputs[0][0]
        probs = np.exp(logits) / np.sum(np.exp(logits))
        best = np.argmax(probs)
        label = id2label.get(str(best), f"label_{best}")
        ok = "OK" if label == expected else "X "
        if label == expected:
            correct += 1
        print(f"  {ok} {text[:45]:<45} -> {label:<15} (prob: {probs[best]:.3f})")

    print(f"\nAccuracy: {correct}/{len(tests)} ({100 * correct / len(tests):.0f}%)")
    print(f"Model size: {onnx_file.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
