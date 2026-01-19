import argparse
import os
import pandas as pd
from datetime import datetime
from src import (
    load_model_and_processor,
    load_wmt_data,
    batch_translate,
    comet22_eval,
    metricx24_eval
)

def main():
    parser = argparse.ArgumentParser(description="Translation Pipeline")
    parser.add_argument("--force", action="store_true", help="Run translation model even if cache exists")
    parser.add_argument("--limit", type=int, default=None, help="Number of sentences (default: all)")
    parser.add_argument("--cache", type=str, default=None, help="Custom path for the translation cache")
    args = parser.parse_args()

    # path to cache data
    if args.cache:
        cache_file = args.cache
    else:
        limit_str = f"limit_{args.limit}" if args.limit else "all"
        cache_file = f"outputs/translations_{limit_str}.csv"
    
    os.makedirs("outputs", exist_ok=True)

    # make new translations or load existing ones
    if args.force or not os.path.exists(cache_file):
        print(f"No valid cache found at {cache_file}. Running translation model...")
        model, processor = load_model_and_processor()
        sources, targets = load_wmt_data(lang="en-nl_NL", limit=args.limit)
        
        translations = batch_translate(model, processor, sources)

        # save to cache
        df = pd.DataFrame({"source": sources, "target": targets, "translation": translations})
        df.to_csv(cache_file, index=False)
        print(f"Saved to {cache_file}")
    else:
        print(f"Found cache translations in {cache_file}")
        df = pd.read_csv(cache_file)
        sources, targets, translations = df['source'].tolist(), df['target'].tolist(), df['translation'].tolist()

    # evaluate comet
    print(f"Evaluating {len(translations)} sentences with COMET-22...")
    comet_results = comet22_eval(sources, translations, targets)
    df["comet22_score"] = comet_results.scores

    # evaluate metricx
    print(f"Evaluating {len(translations)} sentences with MetricX-24...")
    metricx_scores = metricx24_eval(sources, translations)
    df["metricx24_score"] = metricx_scores
    avg_metricx = sum(metricx_scores) / len(metricx_scores)
    
    # create results filename with timestamp (YearMonthDay_HourMinute)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    results_file = cache_file.replace("translations_", f"results_{timestamp}_")
    
    # save full results csv
    df.to_csv(results_file, index=False)
    
    # make summary of results text file
    with open(results_file.replace(".csv", "_summary.txt"), "w") as f:
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Average COMET-22 (Higher is better): {comet_results.system_score:.4f}\n")
        f.write(f"Average MetricX-24 (Lower is better): {avg_metricx:.4f}\n")
        f.write(f"Sample Count: {len(translations)}\n")

    print(f"\nAverage COMET-22: {comet_results.system_score:.4f}")
    print(f"Average MetricX-24: {avg_metricx:.4f}")
    print(f"Results saved to: {results_file}")

if __name__ == "__main__":
    main()