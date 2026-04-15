from argparse import ArgumentParser
import csv
import re
import os
import pickle

from utils import compute_metrics, read_queries, compute_records


def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "-ps",
        "--predicted_sql",
        dest="pred_sql",
        required=True,
        help="path to your model's predicted SQL queries",
    )
    parser.add_argument(
        "-pr",
        "--predicted_records",
        dest="pred_records",
        required=False,
        default=None,
        help="path to the predicted development database records (if empty, will recompute)",
    )
    parser.add_argument(
        "-ds",
        "--development_sql",
        dest="dev_sql",
        required=True,
        help="path to the ground-truth development SQL queries",
    )
    parser.add_argument(
        "-dr",
        "--development_records",
        dest="dev_records",
        required=True,
        help="path to the ground-truth development database records",
    )
    parser.add_argument(
        "--error_table_csv",
        dest="error_table_csv",
        default="",
        help="optional path to save the error analysis table as CSV",
    )
    return parser.parse_args()


def classify_error(error_msg):
    """Get error message type"""
    msg = (error_msg or "").strip()
    low = msg.lower()

    if msg == "":
        return "No SQL Execution Error"
    if "query timed out" in low:
        return "Query Timed Out"
    if "incomplete input" in low:
        return "Incomplete Input"
    if "no such column" in low:
        return "No Such Column"
    if "unrecognized token" in low:
        return "Unrecognized Token"
    if "syntax error" in low:
        return "SQL Syntax Error"
    if "operationalerror" in low:
        return "Other OperationalError"
    return "Other Error"


def clip_snippet(text, max_chars=140):
    """Clean and clip a text snippet for concise display in the error table."""
    clean = " ".join((text or "").split())
    if len(clean) <= max_chars:
        return clean
    return clean[: max_chars - 3] + "..."


def sql_context_window(query_text, marker, radius=70):
    """Return a focused SQL window around a marker token/fragment."""
    clean = " ".join((query_text or "").split())
    if not clean:
        return ""
    if not marker:
        return clip_snippet(clean, max_chars=2 * radius)

    low = clean.lower()
    marker_low = marker.lower()
    pos = low.find(marker_low)
    if pos < 0:
        return clip_snippet(clean, max_chars=2 * radius)

    start = max(0, pos - radius)
    end = min(len(clean), pos + len(marker) + radius)
    window = clean[start:end]
    if start > 0:
        window = "..." + window
    if end < len(clean):
        window = window + "..."
    return window


def extract_relevant_snippet(error_msg, query_text):
    """Extract a concise SQL-relevant snippet without extra explanatory text."""
    msg = (error_msg or "").strip()
    low = msg.lower()
    clean_query = " ".join((query_text or "").split())

    if msg == "":
        return clip_snippet(clean_query, max_chars=180), msg

    column_match = re.search(r"no such column:\s*([^\n]+)", msg, flags=re.IGNORECASE)
    if column_match:
        marker = column_match.group(1).strip()
        return sql_context_window(clean_query, marker), msg

    token_match = re.search(r"unrecognized token:\s*\"([^\"]+)\"", msg, flags=re.IGNORECASE)
    if token_match:
        marker = token_match.group(1).strip()
        return sql_context_window(clean_query, marker), msg

    near_match = re.search(r"near\s+\"([^\"]+)\":\s*syntax error", msg, flags=re.IGNORECASE)
    if near_match:
        marker = near_match.group(1).strip()
        return sql_context_window(clean_query, marker), msg

    if "incomplete input" in low:
        return clip_snippet(clean_query[-140:], max_chars=140), msg

    if "query timed out" in low:
        return clip_snippet(clean_query, max_chars=180), msg

    return clip_snippet(clean_query, max_chars=180), msg


def build_error_table(pred_queries, error_msgs):
    """Group and summarize execution errors for analysis."""
    total = len(error_msgs)
    grouped = {}

    for idx, msg in enumerate(error_msgs):
        query_text = pred_queries[idx] if idx < len(pred_queries) else ""
        err_type = classify_error(msg)
        relevant_snippet, raw_msg = extract_relevant_snippet(msg, query_text)
        if err_type not in grouped:
            grouped[err_type] = {
                "error_type": err_type,
                "raw_error_message": raw_msg,
                "relevant_snippet": clip_snippet(relevant_snippet, max_chars=220),
                "count": 0,
                "total": total,
            }
        grouped[err_type]["count"] += 1

    rows = sorted(grouped.values(), key=lambda r: r["count"], reverse=True)
    for row in rows:
        row["statistics"] = f"{row['count']}/{row['total']}"
    return rows

def save_table_csv(rows, csv_path):
    """Save the error analysis table to a CSV file."""
    fieldnames = [
        "Error Type",
        "Raw Error Message",
        "Relevant Snippet",
        "Statistics",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "Error Type": row["error_type"],
                    "Raw Error Message": row.get("raw_error_message", ""),
                    "Relevant Snippet": row.get("relevant_snippet", ""),
                    "Statistics": row["statistics"],
                }
            )


def main():
    args = get_args()
    
    # If predicted_records not provided, compute and save them
    if args.pred_records is None:
        pred_queries = read_queries(args.pred_sql)
        print(f"Computing records for {len(pred_queries)} queries...")
        recs, error_msgs = compute_records(pred_queries)
        
        # Infer output path from pred_sql
        pred_records_path = args.pred_sql.replace('.sql', '.pkl')
        os.makedirs(os.path.dirname(pred_records_path), exist_ok=True)
        with open(pred_records_path, 'wb') as f:
            pickle.dump((recs, error_msgs), f)
        print(f"Saved recomputed records to: {pred_records_path}")
        
        args.pred_records = pred_records_path
    
    sql_em, record_em, record_f1, model_error_msgs = compute_metrics(
        args.dev_sql,
        args.pred_sql,
        args.dev_records,
        args.pred_records,
    )
    print("SQL Exact Match: ", sql_em * 100)
    print("Record Exact Match: ", record_em * 100)
    print("Record F1: ", record_f1 * 100)

    pred_queries = read_queries(args.pred_sql)
    error_table = build_error_table(pred_queries, model_error_msgs)

    if args.error_table_csv:
        save_table_csv(error_table, args.error_table_csv)
        print(f"\nSaved error table to: {args.error_table_csv}")

if __name__ == "__main__":
    main()