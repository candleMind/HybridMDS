import os, json
import pandas as pd
from huggingface_hub import InferenceClient

PROMPT_DIR  = "VLSP Dataset/LLM_summ_prompts"

HF_API_KEY   = ""
MODEL_ID     = "Qwen/Qwen3-8B"
RETRY_DELAY  = 5
RETRIES      = 3

client = InferenceClient(token=HF_API_KEY)

SCORE_FMT = '''\
For each dimension, reason through the sub-criteria, then give a final integer score.
For each item return ONLY a JSON object, without character '"' in the response to avoid wrongly json parsing:
{"id":<id>, "coherence":<1-5>, "consistency":<1-5>, "fluency":<1-5>, "relevance":<1-5>}'''

REASON_FMT = '''\
For each dimension, reason through the sub-criteria, then give a final integer score.
For each item return ONLY a JSON object, without character '"' in the response to avoid wrongly json parsing:
{"id":<id>, "coherence":<1-5>, "consistency":<1-5>, "fluency":<1-5>, "relevance":<1-5>,
 "r_coherence":"<sub-criteria results and reasoning>", "r_consistency":"<sub-criteria results and reasoning>", "r_fluency":"<sub-criteria results and reasoning>", "r_relevance":"<sub-criteria results and reasoning>"}'''
RUBRIC = """\
You are an expert summary evaluator. For each SUMMARY, first check each sub-criterion 
(Yes/No), then assign a final score 1–5 for the dimension based on your reasoning (1=Very poor, 2=Poor, 3=Fair, 4=Good, 5=Excellent):

1. COHERENCE  (1–5)
   • Topic Maintenance  – Does the summary consistently focus on the central topic 
                          without deviating into unrelated areas?
   • Logical Flow       – Does the summary present information in a logical order?
   • Consistent POV     – Is the point of view or perspective consistent with the source?

2. CONSISTENCY  (1–5)
   • Factual Consistency   – Does the summary accurately represent the facts from the source?
   • No New Information    – Does the summary avoid introducing information not present 
                             in the original source?
   • Contextual Accuracy   – Does the summary preserve the original purpose or intent 
                             of the source document?

3. FLUENCY  (1–5)
   • Formatting     – Is the summary free from formatting issues and correctly capitalized?
   • Grammar        – Are all sentences grammatically correct and free from errors?
   • Completeness   – Are all sentences complete, with no fragments or missing components?
   • Readability    – Is the summary easy to read, without unnecessary complexity?

4. RELEVANCE  (1–5)
   • Content Coverage              – Does the summary encapsulate all critical points?
   • Topic Consistency             – Does the summary maintain the main topic of the source?
   • Consistent Use of Terminology – Does the summary use the same terminology as the source?
   • Use of Key Terms and Phrases  – Does the summary incorporate key terms effectively?
   • Importance                    – Is each point mentioned important to understanding the text?
"""

def build_batch_prompt(items: list[dict], reasoning: bool) -> str:
    """items = [{"id": ..., "source": ..., "summaries": {"sys_A": "...", "sys_B": "..."}}, ...]"""
    fmt = REASON_FMT if reasoning else SCORE_FMT
    blocks = []
    for it in items:
        source_block = f"=== SOURCE {it['doc_id']} ===\n{it['source'].strip()}"
        blocks.append(source_block)
        for sys_id, summary_text in it["summaries"].items():
            blocks.append(
                f"SUMMARY [{it['doc_id']}|{sys_id}]:\n{summary_text.strip()}"
            )
    body = "\n\n".join(blocks)
    return f"{RUBRIC}\n{fmt}\nReturn a JSON array — one object per item, no extra text, no markdown fences.\n\n{body}\n\nOUTPUT:"

def count_words(text: str) -> int:
    return len(text.split())

def generate_prompts(items,
                     word_limit,      # max words per batch prompt
                     reasoning,
                     prompt_dir: str = PROMPT_DIR) -> list[str]:
    os.makedirs(prompt_dir, exist_ok=True)
    overhead = count_words(RUBRIC) + count_words(REASON_FMT if reasoning else SCORE_FMT) + 20

    # ── batch theo word limit ────────────────────────────────
    # word cost của 1 item = source + tất cả summaries + headers
    def item_words(it):
        return (count_words(it["source"])
                + sum(count_words(s) for s in it["summaries"].values())
                + 15 * len(it["summaries"]))   # overhead per summary line

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

    # ── lưu file ─────────────────────────────────────────────
    paths = []
    for idx, batch in enumerate(batches, start=1):
        # thu thập tất cả summary ids trong batch
        all_ids = [f"{it['doc_id']}|{sys}" for it in batch
                   for sys in it["summaries"]]
        prompt   = build_batch_prompt(batch, reasoning)
        filename = os.path.join(prompt_dir, f"batch_{idx:03d}.txt")

        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"# batch={idx}  summaries={len(all_ids)}  ids={all_ids}\n")
            f.write("# Paste model response below OUTPUT:\n\n")
            f.write(prompt)

        paths.append(filename)
        print(f"  batch_{idx:03d}.txt — {len(all_ids)} summaries "
              f"({len(batch)} sources), ~{count_words(prompt)} words")

    print(f"\nDone. {len(paths)} batch(es) → '{prompt_dir}/'  (limit={word_limit} words)")
    return paths

def main():
    # 1. Đọc file tóm tắt (txt) - Giả định mỗi dòng là 1 tóm tắt cho 1 ID
    with open(f'output_summs/results.txt', 'r', encoding='utf-8') as f:
        new_ff_summs = f.readlines()
    summaries = {i: summary for i, summary in enumerate(new_ff_summs)}

    # 2. Đọc file nguồn (csv) và gộp các Answer cùng question_id
    source = {}
    df = pd.read_csv('VLSP Dataset/public.csv', header=None)
    public_ids = list(map(lambda x: x - 1, df[0]))
    with open(r'VLSP Dataset\testing_data_new.jsonl', 'r', encoding='utf-8') as file:
        for clusterid, cluster in enumerate(file.readlines()):
            if clusterid in public_ids: continue
            docs = [d['raw_text'] for d in json.loads(cluster)['single_documents']]
            source[clusterid] = '\n'.join(docs)

    # Mapping ID với nội dung
    # Lưu ý: Đảm bảo thứ tự trong file .txt khớp với ID trong file .csv
    data = []
    for i in source.keys():
        if i < len(summaries):
            data.append({
                "doc_id": i,
                "source": source[i].replace('\n', ' ').replace('\r', ''),
                "summaries": {"0": summaries[i]}
            })

    generate_prompts(data, word_limit=10000, reasoning=True)