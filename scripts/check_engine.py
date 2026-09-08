"""Самотест движка обучения: unsloth, один микрошаг forward+backward.

Запускается в отдельном процессе (import unsloth необратимо патчит
transformers в текущем процессе). Код возврата 0 — движок работоспособен.

Запуск: python scripts/check_engine.py
"""
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.abspath(os.path.join(ROOT, os.pardir, "models"))  # ../models
BASE = os.path.join(MODELS_DIR, "Qwen2.5-3B-Instruct")
os.environ.setdefault("UNSLOTH_COMPILE_LOCATION",
                      os.path.join(MODELS_DIR, "unsloth_compiled_cache"))


def main() -> int:
    try:
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=BASE,
            max_seq_length=512,
            dtype=torch.bfloat16,
            load_in_4bit=False,
            full_finetuning=True,
        )
        ids = tokenizer("Engine self-test: one forward+backward microstep.",
                        return_tensors="pt").to("cuda")
        out = model(**ids, labels=ids["input_ids"])
        out.loss.backward()
        print(f"ENGINE OK: unsloth full, loss={float(out.loss):.4f}")
        return 0
    except Exception as e:
        print(f"ENGINE FAIL: {type(e).__name__}: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
