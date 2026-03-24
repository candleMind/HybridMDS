import os, re, json, time
import pandas as pd
from tqdm import tqdm
from huggingface_hub import InferenceClient
from pathlib import Path
import requests
from LLM_as_judge_save_responses import extract_json_array, extract_json_objects, save_results

HF_API_KEY   = ""
MODEL_ID     = "Qwen/Qwen3-8B"
RETRY_DELAY  = 5
RETRIES      = 3

client = InferenceClient(token=HF_API_KEY)

SUMMARIZE_INSTRUCTION = """\
Bạn là chuyên gia tóm tắt văn bản. Với mỗi CỤM VĂN BẢN, hãy sinh một bản tóm tắt \
tiếng Việt ngắn gọn từ 3-6 câu, súc tích, bao quát các điểm chính từ TẤT CẢ các văn bản trong cụm.

Trả về ONLY một JSON array, không có text thừa, không có ký tự ngoặc kép (") trong bản tóm tắt để tránh parse json sai:
[
  {"id": <id dạng string>, "summary": "<tóm tắt tiếng Việt>"},
  ...
]
"""

PROMPT_DIR  = "VLSP Dataset/LLM_summ_prompts"
RESPONSE_DIR = "VLSP Dataset/LLM_summaries"
OUTPUT_CSV   = "Qwen3_8b_summaries.xlsx"


def build_summarize_prompt(items: list[dict]) -> str:
    """
    items = [{"id": ..., "docs": {"doc_1": "...", "doc_2": "...", ...}}, ...]
    Mỗi item là một cụm văn bản cần tóm tắt.
    """
    blocks = []
    for it in items:
        doc_lines = "\n\n".join(
            f"  [Văn bản {doc_id}]\n  {text.strip()}"
            for doc_id, text in it["docs"].items()
        )
        blocks.append(f"=== CỤM {it['id']} ===\n{doc_lines}")
    return f"{SUMMARIZE_INSTRUCTION}\n\n" + "\n\n".join(blocks) + "\n\nOUTPUT:"

def count_words(text: str) -> int:
    return len(text.split())

def generate_summarize_prompts(items: list[dict],
                                word_limit: int = 3000,
                                prompt_dir: str = PROMPT_DIR) -> list[str]:
    """
    df cột: cluster_id, doc_id, text
    Chia thành batch theo word_limit, lưu mỗi batch ra file txt.
    """
    os.makedirs(prompt_dir, exist_ok=True)
    overhead = count_words(SUMMARIZE_INSTRUCTION) + 20

    def item_words(it):
        return (sum(count_words(t) for t in it["docs"].values())
                + 10 * len(it["docs"]))   # overhead per doc header

    batches, current_batch, current_words = [], [], 0
    for it in items:
        cost = item_words(it)
        if current_batch and (current_words + cost + overhead > word_limit):
            batches.append(current_batch)
            current_batch, current_words = [], 0
        current_batch.append(it)
        current_words += cost
    if current_batch:
        batches.append(current_batch)

    paths = []
    for idx, batch in enumerate(batches, start=1):
        all_ids  = [it["id"] for it in batch]
        prompt   = build_summarize_prompt(batch)
        filename = os.path.join(prompt_dir, f"batch_{idx:03d}.txt")

        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"# batch={idx}  clusters={len(all_ids)}  ids={all_ids}"
                    f"  words≈{count_words(prompt)}\n")
            f.write("# Paste model response below OUTPUT:\n\n")
            f.write(prompt)

        paths.append(filename)
        print(f"  batch_{idx:03d}.txt — {len(all_ids)} clusters, ~{count_words(prompt)} words")

    print(f"\nDone. {len(paths)} batch(es) → '{prompt_dir}/'  (limit={word_limit} words)")
    return paths


def call_qwen(prompt: str, max_new_tokens: int = 4096, temperature: float = 0.3,
              thinking: bool = False) -> str:
    url = "https://router.huggingface.co/v1/chat/completions"
    headers = {"Authorization": f"Bearer {HF_API_KEY}", "Content-Type": "application/json"}

    content = prompt if thinking else f"/no_think\n{prompt}"

    payload = {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_new_tokens,
        "temperature": temperature,
    }
    for attempt in range(1, RETRIES + 1):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=120)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"  [attempt {attempt}/{RETRIES}] {e}")
            if attempt < RETRIES:
                time.sleep(RETRY_DELAY * attempt)
    return ""


def extract_prompt_body(content: str) -> str:
    """Lấy phần prompt (bỏ header comment), giữ nguyên OUTPUT: ở cuối."""
    lines = content.splitlines()
    body = [l for l in lines if not l.startswith("#")]
    return "\n".join(body).strip()


def process_folder(prompt_dir: str = PROMPT_DIR,
                   response_dir: str = RESPONSE_DIR) -> None:
    os.makedirs(response_dir, exist_ok=True)
    txt_files = sorted(Path(prompt_dir).glob("*.txt"))

    if not txt_files:
        print(f"Không tìm thấy file .txt trong '{prompt_dir}'")
        return

    print(f"Found {len(txt_files)} file(s) in '{prompt_dir}'\n")
    all_rows = []

    for fpath in tqdm(txt_files):
        content = fpath.read_text(encoding="utf-8")
        prompt = extract_prompt_body(content)
        id_match = re.search(r"ids=(\[.*?\])", content)
        expected_ids = json.loads(id_match.group(1)) if id_match else []
        if expected_ids[-1] < 273 or expected_ids[0] > 273: continue

        response = call_qwen(prompt)
        if not response:
            print("✗ empty response")
            continue

        # Parse response
        rows = extract_json_array(response, return_json=False)

        with open(RESPONSE_DIR + '/qwen_texts.txt', "a", encoding="utf-8") as f:
            f.write(rows + "\n")

    print(f"\nDone. Results → '{OUTPUT_CSV}'")


def parse_summarize_response(response_text: str, return_json=True,
                             expected_ids: list = None,
                             output_csv: str = OUTPUT_CSV) -> pd.DataFrame:
    """
    Parse output string từ model → DataFrame.

    Args:
        response_text : chuỗi output từ model (có thể có markdown fence)
        expected_ids  : list cluster_id mong đợi, dùng để cảnh báo nếu thiếu
        source_file   : tên file nguồn (để tracking), có thể bỏ trống
    """
    parsed = extract_json_array(response_text)
    if parsed is None:
        parsed = extract_json_objects(response_text)

    rows = []
    for item in parsed:
        rows.append({
            "cluster_id": str(item.get("id", "unknown")),
            "summary": item.get("summary", ""),
            # "source_file": source_file,
        })

    if expected_ids:
        parsed_ids = {r["cluster_id"] for r in rows}
        missing = set(str(i) for i in expected_ids) - parsed_ids
        if missing:
            print(f"  ⚠ Missing cluster_ids: {missing}")

    df = pd.DataFrame(rows)
    return df


def main():
    data = []
    df = pd.read_csv('VLSP Dataset\public.csv', header=None)
    public_ids = list(map(lambda x: x - 1, df[0]))
    with open(r'VLSP Dataset\testing_data_new.jsonl', 'r', encoding='utf-8') as file:
        for clusterid, cluster in enumerate(file.readlines()):
            if clusterid in public_ids: continue
            cur_data = {"id": clusterid, "docs": {}}
            cluster = json.loads(cluster)['single_documents']
            for docid, doc in enumerate(cluster):
                cur_data['docs'][docid + 1] = doc['raw_text']
            data.append(cur_data)

    generate_summarize_prompts(data, word_limit=10000)

    # Call Qwen
    process_folder()

    # Parse response
    response = """
    [
      {"id": "273", "summary": "Luật Cảnh sát cơ động 2022 quy định rõ nhiệm vụ, quyền hạn, chế độ chính sách và các hành vi bị nghiêm cấm đối với lực lượng này. Luật bổ sung các hành vi cản trở hoạt động, trả thù, mua chuộc cán bộ chiến sĩ, đồng thời quy định cụ thể về sử dụng vũ khí, phương tiện và quyền huy động lực lượng trong trường hợp khẩn cấp. Luật có hiệu lực từ ngày 1/1/2023."}]
    ]

    """
    df = parse_summarize_response(response)
    save_results(df, RESPONSE_DIR + '/' + OUTPUT_CSV)
