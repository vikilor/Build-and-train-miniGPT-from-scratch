import json

input_path = "Belle_open_source_1M.json"        # 实际上是 JSONL 格式
output_path = "belle_general_qa.jsonl"

with open(input_path, "r", encoding="utf-8") as fin, \
     open(output_path, "w", encoding="utf-8") as fout:
    
    for line in fin:
        try:
            data = json.loads(line.strip())
            instruction = data.get("instruction", "").strip()
            input_text = data.get("input", "").strip()
            output_text = data.get("output", "").strip()

            final_input = instruction if input_text == "" else f"{instruction}\n{input_text}"

            new_data = {
                "input": final_input,
                "target": output_text,
                "task": "general_qa"
            }

            fout.write(json.dumps(new_data, ensure_ascii=False) + "\n")
        except json.JSONDecodeError:
            continue  # 跳过不合法的 JSON 行

