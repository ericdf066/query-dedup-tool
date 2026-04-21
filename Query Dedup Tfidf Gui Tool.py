import re
import os
import hashlib
import traceback
from typing import List

import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def normalize_text(text: str) -> str:
    if pd.isna(text):
        return ""
    text = str(text).lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[，。！？；：、“”‘’\"'()（）【】\[\]<>《》,.!?;:]", "", text)
    return text


def char_ngrams(text: str, n: int = 2) -> List[str]:
    if not text:
        return [""]
    if len(text) <= n:
        return [text]
    return [text[i:i+n] for i in range(len(text) - n + 1)]


def hash_token(token: str) -> int:
    md5 = hashlib.md5(token.encode("utf-8")).hexdigest()
    return int(md5[:16], 16)


def simhash(text: str, ngram: int = 2, hash_bits: int = 64) -> int:
    text = normalize_text(text)
    features = char_ngrams(text, n=ngram)

    vector = [0] * hash_bits
    for feature in features:
        h = hash_token(feature)
        for i in range(hash_bits):
            bitmask = 1 << i
            if h & bitmask:
                vector[i] += 1
            else:
                vector[i] -= 1

    fingerprint = 0
    for i in range(hash_bits):
        if vector[i] > 0:
            fingerprint |= (1 << i)
    return fingerprint


def hamming_distance(x: int, y: int) -> int:
    return bin(x ^ y).count("1")


def load_csv_with_fallback(file_path: str) -> pd.DataFrame:
    encodings = ["utf-8-sig", "utf-8", "gb18030", "gbk"]
    last_error = None
    for enc in encodings:
        try:
            return pd.read_csv(file_path, encoding=enc)
        except Exception as e:
            last_error = e
    raise last_error


def deduplicate_queries_from_csv(
    input_csv: str,
    query_column: str,
    output_dir: str,
    distance_threshold: int = 3,
    ngram: int = 2,
):
    df = load_csv_with_fallback(input_csv)

    if query_column not in df.columns:
        raise ValueError(f"列名 '{query_column}' 不存在。当前列有: {list(df.columns)}")

    work_df = df.copy()
    work_df["original_query"] = work_df[query_column].astype(str)
    work_df["normalized_query"] = work_df["original_query"].apply(normalize_text)

    work_df = work_df[work_df["normalized_query"] != ""].copy()
    work_df = work_df.reset_index(drop=True)

    exact_dedup_df = work_df.drop_duplicates(subset=["normalized_query"]).copy()
    exact_dedup_df = exact_dedup_df.reset_index(drop=True)

    representatives = []
    representative_hashes = []
    group_records = []

    for _, row in exact_dedup_df.iterrows():
        query = row["original_query"]
        normalized_query = row["normalized_query"]
        sh = simhash(query, ngram=ngram)

        matched_group_id = None
        matched_rep_query = None
        matched_distance = None

        for group_id, rep_hash in enumerate(representative_hashes):
            dist = hamming_distance(sh, rep_hash)
            if dist <= distance_threshold:
                matched_group_id = group_id
                matched_rep_query = representatives[group_id]["representative_query"]
                matched_distance = dist
                break

        if matched_group_id is None:
            matched_group_id = len(representatives)
            matched_rep_query = query
            matched_distance = 0

            representatives.append({
                "group_id": matched_group_id,
                "representative_query": query,
                "representative_normalized_query": normalized_query,
                "simhash": sh,
            })
            representative_hashes.append(sh)

        group_records.append({
            "group_id": matched_group_id,
            "original_query": query,
            "normalized_query": normalized_query,
            "representative_query": matched_rep_query,
            "distance_to_representative": matched_distance,
        })

    groups_df = pd.DataFrame(group_records)

    final_df = work_df.merge(
        groups_df[["normalized_query", "group_id", "representative_query", "distance_to_representative"]],
        on="normalized_query",
        how="left"
    )

    deduped_df = pd.DataFrame(representatives)[
        ["group_id", "representative_query", "representative_normalized_query"]
    ].copy()

    deduped_path = os.path.join(output_dir, "deduped_queries.csv")
    groups_path = os.path.join(output_dir, "query_groups.csv")

    deduped_df.to_csv(deduped_path, index=False, encoding="utf-8-sig")
    final_df.to_csv(groups_path, index=False, encoding="utf-8-sig")

    summary = {
        "input_rows": len(df),
        "non_empty_rows": len(work_df),
        "exact_dedup_rows": len(exact_dedup_df),
        "final_rows": len(deduped_df),
        "deduped_path": deduped_path,
        "groups_path": groups_path,
        "deduped_df": deduped_df,
    }
    return summary


def build_semantic_groups_tfidf(queries: List[str], similarity_threshold: float):
    if not queries:
        return []

    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 3))
    matrix = vectorizer.fit_transform(queries)
    sim_matrix = cosine_similarity(matrix)

    n = len(queries)
    visited = [False] * n
    groups = []

    for i in range(n):
        if visited[i]:
            continue
        stack = [i]
        component = []
        visited[i] = True

        while stack:
            node = stack.pop()
            component.append(node)
            for j in range(n):
                if not visited[j] and sim_matrix[node][j] >= similarity_threshold:
                    visited[j] = True
                    stack.append(j)

        groups.append(sorted(component))

    return groups


def semantic_deduplicate_queries_tfidf(
    base_queries_df: pd.DataFrame,
    output_dir: str,
    similarity_threshold: float = 0.85,
):
    queries = base_queries_df["representative_query"].astype(str).tolist()
    groups = build_semantic_groups_tfidf(queries, similarity_threshold)

    semantic_records = []
    representatives = []

    for group_id, group_indices in enumerate(groups):
        group_queries = [queries[idx] for idx in group_indices]
        representative_query = group_queries[0]

        representatives.append({
            "semantic_group_id": group_id,
            "representative_query": representative_query,
            "group_size": len(group_queries),
        })

        for idx in group_indices:
            semantic_records.append({
                "semantic_group_id": group_id,
                "original_representative_query": queries[idx],
                "semantic_representative_query": representative_query,
            })

    semantic_deduped_df = pd.DataFrame(representatives)
    semantic_groups_df = pd.DataFrame(semantic_records)

    semantic_deduped_path = os.path.join(output_dir, "semantic_deduped_queries.csv")
    semantic_groups_path = os.path.join(output_dir, "semantic_query_groups.csv")

    semantic_deduped_df.to_csv(semantic_deduped_path, index=False, encoding="utf-8-sig")
    semantic_groups_df.to_csv(semantic_groups_path, index=False, encoding="utf-8-sig")

    return {
        "semantic_input_rows": len(queries),
        "semantic_final_rows": len(semantic_deduped_df),
        "semantic_deduped_path": semantic_deduped_path,
        "semantic_groups_path": semantic_groups_path,
    }


class QueryDedupApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Query 去重工具")
        self.root.geometry("850x680")

        self.csv_path_var = tk.StringVar()
        self.column_var = tk.StringVar(value="query")
        self.output_dir_var = tk.StringVar()
        self.threshold_var = tk.StringVar(value="3")
        self.ngram_var = tk.StringVar(value="2")
        self.enable_semantic_var = tk.BooleanVar(value=False)
        self.semantic_threshold_var = tk.StringVar(value="0.85")
        self.semantic_mode_var = tk.StringVar(value="TF-IDF（本地模式，无需下载模型）")

        self._build_ui()

    def _build_ui(self):
        container = ttk.Frame(self.root, padding=16)
        container.pack(fill="both", expand=True)

        title = ttk.Label(container, text="Query 去重工具（文本去重 + 本地语义近似去重）", font=("Arial", 16, "bold"))
        title.pack(anchor="w", pady=(0, 12))

        desc = ttk.Label(
            container,
            text="使用方式：选择 CSV 文件 → 确认列名 → 可选开启语义近似去重 → 点击开始处理 → 在输出目录查看结果文件。",
            wraplength=820,
        )
        desc.pack(anchor="w", pady=(0, 16))

        file_frame = ttk.LabelFrame(container, text="1. 选择输入文件", padding=12)
        file_frame.pack(fill="x", pady=6)

        ttk.Entry(file_frame, textvariable=self.csv_path_var).pack(side="left", fill="x", expand=True, padx=(0, 8))
        ttk.Button(file_frame, text="选择 CSV", command=self.select_csv).pack(side="left")

        setting_frame = ttk.LabelFrame(container, text="2. 文本去重参数", padding=12)
        setting_frame.pack(fill="x", pady=6)

        row1 = ttk.Frame(setting_frame)
        row1.pack(fill="x", pady=4)
        ttk.Label(row1, text="Query 列名：", width=14).pack(side="left")
        ttk.Entry(row1, textvariable=self.column_var, width=24).pack(side="left")

        row2 = ttk.Frame(setting_frame)
        row2.pack(fill="x", pady=4)
        ttk.Label(row2, text="相似阈值：", width=14).pack(side="left")
        ttk.Entry(row2, textvariable=self.threshold_var, width=10).pack(side="left", padx=(0, 20))
        ttk.Label(row2, text="ngram：", width=8).pack(side="left")
        ttk.Entry(row2, textvariable=self.ngram_var, width=10).pack(side="left")

        row3 = ttk.Frame(setting_frame)
        row3.pack(fill="x", pady=4)
        ttk.Label(row3, text="输出目录：", width=14).pack(side="left")
        ttk.Entry(row3, textvariable=self.output_dir_var).pack(side="left", fill="x", expand=True, padx=(0, 8))
        ttk.Button(row3, text="选择目录", command=self.select_output_dir).pack(side="left")

        tips = ttk.Label(
            setting_frame,
            text="建议：中文 query 默认使用 列名=query、相似阈值=3、ngram=2。",
            foreground="#555555"
        )
        tips.pack(anchor="w", pady=(8, 0))

        semantic_frame = ttk.LabelFrame(container, text="3. 语义近似去重参数（本地模式）", padding=12)
        semantic_frame.pack(fill="x", pady=6)

        row4 = ttk.Frame(semantic_frame)
        row4.pack(fill="x", pady=4)
        ttk.Checkbutton(row4, text="启用语义近似去重（TF-IDF + Clustering，本地模式）", variable=self.enable_semantic_var).pack(side="left")

        row5 = ttk.Frame(semantic_frame)
        row5.pack(fill="x", pady=4)
        ttk.Label(row5, text="语义阈值：", width=14).pack(side="left")
        ttk.Entry(row5, textvariable=self.semantic_threshold_var, width=10).pack(side="left", padx=(0, 20))
        ttk.Label(row5, text="模式说明：", width=10).pack(side="left")
        ttk.Entry(row5, textvariable=self.semantic_mode_var, state="readonly").pack(side="left", fill="x", expand=True)

        semantic_tip = ttk.Label(
            semantic_frame,
            text="建议先用 0.85；此模式基于本地 TF-IDF 相似度，不需要下载外部模型。",
            foreground="#555555",
            wraplength=820,
        )
        semantic_tip.pack(anchor="w", pady=(8, 0))

        action_frame = ttk.Frame(container)
        action_frame.pack(fill="x", pady=12)
        ttk.Button(action_frame, text="开始处理", command=self.run_dedup).pack(side="left")
        ttk.Button(action_frame, text="打开输出目录", command=self.open_output_dir).pack(side="left", padx=8)

        result_frame = ttk.LabelFrame(container, text="4. 运行结果", padding=12)
        result_frame.pack(fill="both", expand=True, pady=6)

        self.result_text = tk.Text(result_frame, height=20, wrap="word")
        self.result_text.pack(fill="both", expand=True)
        self.result_text.insert("end", "运行结果会显示在这里。\n")
        self.result_text.config(state="disabled")

    def log(self, message: str):
        self.result_text.config(state="normal")
        self.result_text.insert("end", message + "\n")
        self.result_text.see("end")
        self.result_text.config(state="disabled")

    def select_csv(self):
        path = filedialog.askopenfilename(
            title="选择 CSV 文件",
            filetypes=[("CSV 文件", "*.csv"), ("所有文件", "*.*")]
        )
        if path:
            self.csv_path_var.set(path)
            if not self.output_dir_var.get():
                self.output_dir_var.set(os.path.dirname(path))

    def select_output_dir(self):
        path = filedialog.askdirectory(title="选择输出目录")
        if path:
            self.output_dir_var.set(path)

    def open_output_dir(self):
        output_dir = self.output_dir_var.get().strip()
        if not output_dir:
            messagebox.showwarning("提示", "请先选择输出目录。")
            return
        if not os.path.isdir(output_dir):
            messagebox.showwarning("提示", "输出目录不存在。")
            return
        try:
            os.system(f'open "{output_dir}"')
        except Exception as e:
            messagebox.showerror("错误", str(e))

    def run_dedup(self):
        csv_path = self.csv_path_var.get().strip()
        query_column = self.column_var.get().strip()
        output_dir = self.output_dir_var.get().strip()
        threshold_text = self.threshold_var.get().strip()
        ngram_text = self.ngram_var.get().strip()
        enable_semantic = self.enable_semantic_var.get()
        semantic_threshold_text = self.semantic_threshold_var.get().strip()

        if not csv_path:
            messagebox.showwarning("提示", "请先选择 CSV 文件。")
            return
        if not os.path.isfile(csv_path):
            messagebox.showwarning("提示", "CSV 文件不存在，请重新选择。")
            return
        if not query_column:
            messagebox.showwarning("提示", "请填写 query 列名。")
            return
        if not output_dir:
            messagebox.showwarning("提示", "请选择输出目录。")
            return
        if not os.path.isdir(output_dir):
            messagebox.showwarning("提示", "输出目录不存在。")
            return

        try:
            threshold = int(threshold_text)
            ngram = int(ngram_text)
        except ValueError:
            messagebox.showwarning("提示", "相似阈值 和 ngram 必须是整数。")
            return

        if enable_semantic:
            try:
                semantic_threshold = float(semantic_threshold_text)
            except ValueError:
                messagebox.showwarning("提示", "语义阈值必须是小数，例如 0.85。")
                return
        else:
            semantic_threshold = 0.85

        self.log("开始处理...")
        self.log(f"输入文件：{csv_path}")
        self.log(f"query 列名：{query_column}")
        self.log(f"输出目录：{output_dir}")

        try:
            summary = deduplicate_queries_from_csv(
                input_csv=csv_path,
                query_column=query_column,
                output_dir=output_dir,
                distance_threshold=threshold,
                ngram=ngram,
            )

            self.log("文本去重完成。")
            self.log(f"输入总行数: {summary['input_rows']}")
            self.log(f"去空后行数: {summary['non_empty_rows']}")
            self.log(f"精确去重后行数: {summary['exact_dedup_rows']}")
            self.log(f"近似去重后代表 query 数: {summary['final_rows']}")
            self.log(f"已输出: {summary['deduped_path']}")
            self.log(f"已输出: {summary['groups_path']}")

            if enable_semantic:
                self.log("开始本地语义近似去重...")
                semantic_summary = semantic_deduplicate_queries_tfidf(
                    base_queries_df=summary["deduped_df"],
                    output_dir=output_dir,
                    similarity_threshold=semantic_threshold,
                )
                self.log("本地语义近似去重完成。")
                self.log(f"语义去重输入数: {semantic_summary['semantic_input_rows']}")
                self.log(f"语义去重后代表 query 数: {semantic_summary['semantic_final_rows']}")
                self.log(f"已输出: {semantic_summary['semantic_deduped_path']}")
                self.log(f"已输出: {semantic_summary['semantic_groups_path']}")

            messagebox.showinfo("完成", "处理完成，结果文件已生成。")
        except Exception as e:
            self.log("处理失败。")
            self.log(str(e))
            self.log(traceback.format_exc())
            messagebox.showerror("错误", f"运行失败：\n{e}")


if __name__ == "__main__":
    root = tk.Tk()
    app = QueryDedupApp(root)
    root.mainloop()