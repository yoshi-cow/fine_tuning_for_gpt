# OpenAIによるカスタムモデルの評価方法(eval)について
* https://platform.openai.com/docs/guides/evals

## Step
1. eval用のタスクを用意する
2. タスクについてのインプットデータ(プロンプトと必要データ)でカスタムモデルを走らせる
3. 回答を分析し、プロンプトを修正しつつ上記を繰り返す

| ステップ                      | 目的                               | 主要 API エンドポイント                                     |
| ------------------------- | -------------------------------- | -------------------------------------------------- |
| ① **Eval 定義を作る**          | テストタスクと採点基準を JSON で登録            | `POST /v1/evals`                                   |
| ② **テストデータを用意し Eval を実行** | JSONL ファイルをアップロード → Eval Run を開始 | `POST /v1/files` → `POST /v1/evals/{eval_id}/runs` |
| ③ **結果を解析し改善**            | Run の進捗を取得し合否・使用量を確認             | `GET /v1/evals/{eval_id}/runs/{run_id}`            |

<br>

以下、「ITサポートチケット分類タスク」を題材に、OpenAI Evals APIだけでモデル評価を行う手順

### 1. eval用のタスク用意 (Evalを定義)
ここでは、IT support ticketが Hardware, Software, Otherの3つに正しく分類されるかのテスト。以下が、対象のchatcomplition api：

```python

from openai import OpenAI
client = OpenAI()

instructions = """
You are an expert in categorizing IT support tickets. Given the support 
ticket below, categorize the request into one of "Hardware", "Software", 
or "Other". Respond with only one of those words.
"""

ticket = "My monitor won't turn on - help!"

completion = client.chat.completions.create(
    model="gpt-4.1",
    messages=[
        {"role": "developer", "content": instructions},
        {"role": "user", "content": ticket}
    ]
)

print(completion.choices[0].message.content)
```

* eval apiには、以下2つのキーが必要
  * <b>data_source_config</b> : 評価用テストデータのスキーマを表す
  * <b>testing_criteria</b> : モデルの出力が正しいかを判定する基準

```python
from openai import OpenAI, APIError
from dotenv import load_dotenv
import os, json, time

load_dotenv()
client = OpenAI()          # OPENAI_API_KEY は自動ロード
# 1_create_eval.py
from common import client

resp = client.evals.create(
    name="it_ticket_categorization_eval",
    data_source_config={
        "type": "custom",
        "item_schema": {          # テストデータ 1 件の構造
            "type": "object",
            "properties": {
                "ticket_text":   {"type": "string"}, # 投入プロンプト
                "correct_label": {"type": "string"} # 正解ラベル
            },
            "required": ["ticket_text", "correct_label"]
        },
        "include_sample_schema": True
    },
    testing_criteria=[            # 正解文字列と完全一致するか
        {
            "type": "string_check",
            "name": "label_match",
            "input": "{{ sample.output_text }}",
            "operation": "eq",
            "reference": "{{ item.correct_label }}"
        }
    ]
)

print("Eval created:", resp.id)     # メモ：EVAL_ID
```

* 実行 → EVAL_ID = eval_xxxxx を控える。


### 2. テストデータの作成と実行
#### テストデータ作成(JSONLファイル)
* tickets.jsonl
```json
{ "item": { "ticket_text": "My monitor won't turn on!",      "correct_label": "Hardware" } }
{ "item": { "ticket_text": "I'm in vim and I can't quit!",   "correct_label": "Software" } }
{ "item": { "ticket_text": "Best restaurants in Cleveland?", "correct_label": "Other"    } }
```
* item の中身は Eval 定義と同じフィールド名 (ticket_text, correct_label) にする。
* 
#### ファイルのアップロード
```python
# 2_upload_test_file.py
from common import client
FILE_ID = client.files.create(
    file=open("tickets.jsonl", "rb"),
    purpose="evals"
).id
print("File uploaded:", FILE_ID)
```

#### Eval Runの起動
* ファインチューニング後のモデルIDが評価対象

```python
# 3_start_eval_run.py
from common import client
EVAL_ID = "eval_xxxxx"          # ① 手順 1 で得た ID を貼る
FILE_ID = "file_xxxxx"          # ② 手順 2 で得た ID を貼る
MODEL_ID = "ft:gpt-3.5-turbo:acme_corp:helpdesk-v1:8mOq3W5J"  # 例：Fine-tuned モデル

run = client.evals.create_run(
    eval_id=EVAL_ID,
    name="ticket_categorization_run_v1",
    data_source={
        "type": "completions",
        "model": MODEL_ID,
        "input": [                               # チャットテンプレート
            {
                "role": "developer",
                "content": (
                    "You are an expert in categorizing IT support tickets. "
                    "Given the support ticket below, categorize it into one of "
                    "\"Hardware\", \"Software\", or \"Other\". "
                    "Respond with only that word."
                )
            },
            {
                "role": "user",
                "content": "{{ item.ticket_text }}"
            }
        ],
        "source": { "type": "file_id", "id": FILE_ID }
    }
)

RUN_ID = run.id
print("Eval Run started:", RUN_ID, "status:", run.status)
```

#### 進捗待って、結果を表示
```python
# 4_wait_and_report.py
from common import client, time
EVAL_ID = "eval_xxxxx"
RUN_ID  = "evalrun_xxxxx"

while True:
    run = client.evals.get_run(eval_id=EVAL_ID, run_id=RUN_ID)
    print("⏳", run.status)
    if run.status in ("completed", "failed"):
        break
    time.sleep(5)

if run.status == "completed":
    rc = run.result_counts
    print(f"{rc.passed}/{rc.total} passed  ({rc.failed} failed)")
    print("Report:", run.report_url)        # ダッシュボードで詳細確認
else:
    print("Error:", run.error)
```
* ダッシュボード (run.report_url) では 各チケットの生成出力 vs 正解、トークン使用量、合否の内訳を確認できる。

### 3. 改善ループ
| 状況         | 施策                                                         |
| ---------- | ---------------------------------------------------------- |
| 合格率が低い     | プロンプトを修正（出力フォーマットを明示・繰り返し指示を追加）→ **3. Run を再作成**           |
| 特定パターンのみ失敗 | テストデータを拡充し、Fine-tuning データ or プロンプトを追加学習                   |
| モデル差比較     | 同じ `EVAL_ID` を使い、`model` を切り替えて複数 `RUN_ID` を作成しダッシュボードで並べる |

### 補：失敗ケースを抜き出すスニペット
```python
records = client.evals.get_run_records(eval_id=EVAL_ID, run_id=RUN_ID)
failed = [r for r in records.data if not r.result["label_match"]["passed"]]
for r in failed:
    print("✗", r.input["ticket_text"], "→", r.output_text,
          "(answer should be", r.item["correct_label"] + ")")
```

### まとめ
1. **Eval 定義**＝タスク仕様書
2. **JSONL データ**＝試験項目
3. **Run**＝自動採点セッション

この 3 つをコードで回すことで、Fine-tuned モデルの **品質を定量確認 → 改善 → 再確認** がシームレスに行える。

