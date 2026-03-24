import os, re, json
import pandas as pd

DIMS = ["coherence", "consistency", "fluency", "relevance"]
EMPTY = {d: -1 for d in DIMS}
RESPONSE_FILE = "VLSP Dataset/LLM_summaries/summeval_gpt53.xlsx"  # thư mục chứa file txt output từ model


def extract_json_array(text: str, return_json=True) -> list | None:
    """Tìm JSON array [...] đầu tiên trong text."""
    # Xóa markdown fences nếu có
    text = re.sub(r"```(?:json)?|```", "", text).strip()
    # Tìm [...] ngoài cùng
    start = text.find("[")
    if start == -1:
        return None
    depth, end = 0, -1
    for i, ch in enumerate(text[start:], start):
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    # print('\n\n', text[start:end])
    # print(json.loads(text[start:end]))
    if end == -1:
        return None
    if return_json:
        return json.loads(text[start:end])
    return text[start:end]


def extract_json_objects(text: str) -> list:
    """Fallback: tìm tất cả {...} objects trong text."""
    text = re.sub(r"```(?:json)?|```", "", text)
    results = []
    depth, start = 0, -1
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0: start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start != -1:
                try:
                    results.append(json.loads(text[start:i + 1]))
                except json.JSONDecodeError:
                    pass
                start = -1
    return results


def parse_response(response_text: str, return_json=True) -> list[dict]:
    """
    Parse output string từ model.
    Trả về list of dicts với keys: id, coherence, consistency, fluency, relevance, [r_*]
    """
    parsed = extract_json_array(response_text, return_json=return_json)
    if parsed is None:
        parsed = extract_json_objects(response_text)
    rows = []
    for item in parsed:
        row = {"id": str(item.get("id", "unknown"))}
        for d in DIMS:
            val = item.get(d, -1)
            try:
                row[d] = int(val)
            except (ValueError, TypeError):
                row[d] = -1
        # Reasoning fields (optional)
        for d in DIMS:
            key = f"r_{d}"
            if key in item:
                row[key] = str(item[key])
        rows.append(row)
    return rows


def parse_raw_string(response_text: str, expected_ids: list = None) -> list[dict]:
    """
    Parse trực tiếp từ 1 chuỗi string (dùng khi có response sẵn trong notebook).
    expected_ids: dùng để gán id nếu model không trả về id trong JSON.
    """
    rows = parse_response(response_text)
    if expected_ids and rows:
        for i, row in enumerate(rows):
            if row["id"] == "unknown" and i < len(expected_ids):
                row["id"] = str(expected_ids[i])
    return rows


def save_results(df: pd.DataFrame, output_csv: str = OUTPUT_CSV) -> None:
    if os.path.exists(output_csv):
        existing = pd.read_excel(output_csv)  #, encoding="utf-8-sig"
        combined = pd.concat([existing, df], ignore_index=True)
        combined.to_excel(output_csv, index=False)
        print(f"✓ Appended {len(df)} rows → '{output_csv}' (total: {len(combined)})")
    else:
        df.to_excel(output_csv, index=False)
        print(f"✓ Created '{output_csv}' with {len(df)} rows")


def main():
    sample_response = """
    [
    {"id":297, "coherence":4, "consistency":4, "fluency":4, "relevance":4,
    "r_coherence":"Topic Maintenance: Yes – summary stays on Hyundai and sedan sales context. Logical Flow: Yes – moves from brand overview to market sales data logically. Consistent POV: Yes – maintains neutral informational perspective consistent with the source.",
    "r_consistency":"Factual Consistency: Yes – key facts like Hyundai Accent leading B-segment and sedan sales drop are correct. No New Information: Yes – no extra facts added. Contextual Accuracy: Yes – overall intent about Hyundai presence and sedan market trends is preserved.",
    "r_fluency":"Formatting: Minor spacing issues but generally acceptable. Grammar: Mostly correct with small punctuation inconsistencies. Completeness: Sentences are complete. Readability: Clear and easy to follow.",
    "r_relevance":"Content Coverage: Captures main ideas about Hyundai position and sedan sales decline. Topic Consistency: Yes – remains focused on the automobile market topic. Terminology: Uses original terms like Hyundai Accent and sedan segment. Key Terms: Present and appropriate. Importance: Selected points are relevant to understanding the source."
    },
    {"id":299, "coherence":3, "consistency":4, "fluency":4, "relevance":3,
    "r_coherence":"Topic Maintenance: Mostly – focuses on fire incidents though multiple events are combined. Logical Flow: Acceptable but somewhat list-like across incidents. Consistent POV: Yes – neutral reporting consistent with the source.",
    "r_consistency":"Factual Consistency: Yes – events, locations and response actions match the source information. No New Information: Yes – no added facts. Contextual Accuracy: Yes – maintains the narrative of fire incidents and firefighting response.",
    "r_fluency":"Formatting: Minor capitalization inconsistencies but readable. Grammar: Generally correct. Completeness: Sentences are complete. Readability: Clear and straightforward.",
    "r_relevance":"Content Coverage: Includes several key fire incidents and response actions though not all details. Topic Consistency: Yes – stays on the fire events. Terminology: Consistent with the source (PCCC, chữa cháy, lực lượng). Key Terms: Used appropriately. Importance: Main operational details are included."
    }
    ]

    """
    rows = parse_raw_string(sample_response)
    df = pd.DataFrame(rows)
    print("Parsed result:")
    # print(df.to_string(index=False))
    save_results(df, RESPONSE_FILE)
