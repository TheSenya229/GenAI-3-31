import argparse
from transformers import pipeline, TextGenerationPipeline

# ------------------------- Константы -------------------------
MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
MAX_OUTPUT_TOKENS = 512
MAX_INPUT_TOKENS = 8192

SYSTEM_PROMPT_STYLE = (
    "Ты — помощник, который демонстрирует заданный стиль письма. "
    "На вход подаётся название стиля (например, 'тёплый', 'юмористический', 'официальный'). "
    "Твоя задача — написать короткий пример (1–2 предложения), отражающий этот стиль."
)

SYSTEM_PROMPT_CONGRATS = (
    "Ты — доброжелательный автор персональных поздравлений. "
    "Создай искреннее и оригинальное поздравление, не более 100 слов, "
    "используя имя, возраст и интересы получателя. "
    "Сохрани стиль текста, который я тебе передам."
)

# ------------------------- Функции -------------------------

def init_pipeline() -> TextGenerationPipeline:
    """Инициализация пайплайна модели."""
    return pipeline(
        task="text-generation",
        model=MODEL_NAME,
        device_map="auto",
        dtype="auto"
    )

def get_style_example(style_name: str, pipe: TextGenerationPipeline) -> str:
    """Генерирует короткий пример текста в заданном стиле (GenAI-1-21)."""
    message = [
        {"role": "system", "content": SYSTEM_PROMPT_STYLE},
        {"role": "user", "content": f"Стиль: {style_name}"}
    ]
    result = pipe(message, max_new_tokens=128, return_full_text=False)[0]["generated_text"]
    return result.strip()

def generate_congratulation(name: str, age: int, interests: list[str],
                            style_text: str, pipe: TextGenerationPipeline) -> str:
    """Создаёт персонализированное поздравление в стиле style_text (GenAI-2-21)."""
    interests_str = ", ".join(interests)
    user_prompt = (
        f"Имя: {name}\nВозраст: {age}\nИнтересы: {interests_str}\n\n"
        f"Стиль примера:\n{style_text}\n\n"
        f"Сгенерируй тёплое поздравление в этом стиле:"
    )
    message = [
        {"role": "system", "content": SYSTEM_PROMPT_CONGRATS},
        {"role": "user", "content": user_prompt}
    ]
    result = pipe(message, max_new_tokens=MAX_OUTPUT_TOKENS, return_full_text=False)[0]["generated_text"]
    text = result.strip()

    # Проверка длины (≤100 слов)
    words = len(text.split())
    if words > 100:
        print(f" Предупреждение: поздравление содержит {words} слов (лимит 100).")
    return text

def save_txt(text: str, path: str):
    """Сохраняет результат в текстовый файл."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"  Поздравление сохранено в {path}")

# ------------------------- Основной скрипт -------------------------

def main():
    parser = argparse.ArgumentParser("GenAI-3-31")
    parser.add_argument("--name", required=True, help="Имя получателя")
    parser.add_argument("--age", required=True, type=int, help="Возраст получателя")
    parser.add_argument("--interests", nargs="+", required=True, help="Интересы (через пробел)")
    parser.add_argument("--style", default="тёплый", help="Стиль поздравления (по умолчанию 'тёплый')")
    parser.add_argument("-o", "--output", default="congratulation.txt", help="Имя выходного .txt файла")
    args = parser.parse_args()

    print("Инициализация модели...")
    try:
        pipe = init_pipeline()
        print(" Модель успешно загружена.")
    except Exception as e:
        print(f"\033[31mОшибка инициализации модели:\n{e}\033[0m")
        return

    try:
        print(f" Генерация стиля: {args.style!r}")
        style_text = get_style_example(args.style, pipe)
        print(f" Пример стиля:\n{style_text}\n")

        print(" Генерация поздравления...")
        congrats = generate_congratulation(args.name, args.age, args.interests, style_text, pipe)
        save_txt(congrats, args.output)
        print("\nГотово 🎁")
    except Exception as e:
        print(f"\033[31mОшибка при генерации поздравления:\n{e}\033[0m")

if __name__ == "__main__":
    main()
