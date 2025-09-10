from openai import OpenAI
from pathlib import Path
import sys, json

client = OpenAI(api_key="")

prompt = """
アップロードしたPDFをスキャンし、指定のyamlフォーマットに変換してください。PDFの緑色のマスク内の領域が答え、薄い赤色のマスク内の領域が大問と小問
マスクの左上隅にある番号を抽出しない！！！！！！！！！！！！
フォーマット: 単元　→　本文(あれば) → 大問 → 小問　→ 答え
"""
#フォーマット: 単元 → 本文(あれば) → 大問 → {小問, 答え}
pdf_path = Path("P58.pdf")
if not pdf_path.exists():
    sys.exit("PDF Not exist")

# (1) PDF アップロード
uploaded = client.files.create(file=open(pdf_path, "rb"), purpose="user_data")

# (2) Responses API 呼び出し
response = client.responses.create(
    model="gpt-5",
    input=[
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": prompt},
                {"type": "input_file", "file_id": uploaded.id}
            ]
        }
    ]
)

# (3) 結果取り出し
text = response.output_text
print(text)

# json 保存
with open("demofile1.yaml", "w", encoding="utf-8") as f:
    f.write(text or "")
