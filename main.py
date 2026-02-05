import argparse
import os
import torch
import pandas as pd
from datetime import datetime
from src import (
    load_model_and_processor,
    load_wmt_data,
    load_flores_data,
    batch_translate,
    comet22_eval,
    metricx24_eval,
    analyze_hypos
)
from src.config import LANG_MAP

def main():
    torch.set_float32_matmul_precision('high')

    parser = argparse.ArgumentParser(description="Translation Pipeline")
    parser.add_argument("--force", action="store_true", help="Run translation model even if cache exists")
    parser.add_argument("--limit", type=int, default=None, 
                        help="Number of evaluation sentences per language (default: all).")
    parser.add_argument("--cache", type=str, default=None, help="Custom path for the translation cache")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to fine-tune checkpoint")
    parser.add_argument("--langs", type=str, nargs="+", default=["nl"], choices=LANG_MAP.keys(), 
                        help="Languages to process (e.g., --langs nl zh)")
    parser.add_argument("--num_samples", type=int, default=1, help="Number of output translations per source sentence (default: 1)")
    parser.add_argument("--dataset", type=str, default="wmt", choices=["wmt", "flores"])
    args = parser.parse_args()

    os.makedirs("outputs", exist_ok=True)

    # initialize model as None to save memory when using cache
    model, processor = None, None

    # loop through provided languages
    for lang_key in args.langs:
        print(f"\n{'='*40}")
        print(f"Processing Language: {lang_key.upper()}")
        print(f"{'='*40}")

        # path to cache data for this language
        limit_str = f"limit_{args.limit}" if args.limit else "all"
        samples_str = f"_{args.num_samples}_samples" if args.num_samples > 1 else ""
        checkpt_suffix = f"_{os.path.basename(os.path.normpath(args.checkpoint))}" if args.checkpoint else ""
        
        # check for existing cache file
        if args.cache:
            cache_file = f"{args.cache.replace('.csv', '')}_{lang_key}.csv"
        else:
            cache_file = f"outputs/{args.dataset}_translations_{lang_key}_{limit_str}{samples_str}{checkpt_suffix}.csv"

        # make new translations or load existing ones
        if args.force or not os.path.exists(cache_file):
            print(f"No valid cache found at {cache_file}. Running translation model...")
            
            # only load the model if needed
            if model is None:
                print("Loading model and processor...")
                model, processor = load_model_and_processor(checkpoint_path=args.checkpoint)
            
            # load dataset for this language
            lang_code = LANG_MAP[lang_key][args.dataset]
            if args.dataset == "wmt":
                sources, targets = load_wmt_data(lang=lang_code, limit=args.limit)
            else:
                sources, targets = load_flores_data(lang=lang_code, limit=args.limit)

            # translate
            # returns list of dicts: [{'source', 'translation', 'likelihood'}, ...]
            gemma_code = LANG_MAP[lang_key]["gemma"]
            results_dicts = batch_translate(
                model, processor, sources, 
                target_lang=gemma_code, 
                num_samples=args.num_samples, 
                batch_size=2
            )

            # copy targets for num_samples>1 (multiple translations for same source)
            df = pd.DataFrame(results_dicts)
            expanded_targets = []
            for t in targets:
                expanded_targets.extend([t] * args.num_samples)

            # save to cache
            df["target"] = expanded_targets
            df.to_csv(cache_file, index=False)
            print(f"Saved to {cache_file}")
        else:
            print(f"Found cache translations in {cache_file}")
            df = pd.read_csv(cache_file)

        # prepare lists for evaluation
        curr_sources = df['source'].tolist()
        curr_translations = df['translation'].tolist()
        curr_targets = df['target'].tolist()

        # evaluate comet
        print(f"Evaluating {len(curr_translations)} {lang_key} candidates with COMET-22...")
        comet_results = comet22_eval(curr_sources, curr_translations, curr_targets)
        df["comet22_score"] = comet_results.scores

        # evaluate metricx
        print(f"Evaluating {len(curr_translations)} {lang_key} candidates with MetricX-24...")
        metricx_scores = metricx24_eval(curr_sources, curr_translations)
        df["metricx24_score"] = metricx_scores
        avg_metricx = sum(metricx_scores) / len(metricx_scores)
        
        # create results filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        results_file = cache_file.replace("translations_", f"results_{timestamp}_")
        
        # save full results csv
        df.to_csv(results_file, index=False)
        
        # make summary of results text file
        with open(results_file.replace(".csv", "_summary.txt"), "w") as f:
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Language: {lang_key}\n")
            f.write(f"Checkpoint: {args.checkpoint if args.checkpoint else 'None'}\n")
            f.write(f"Samples per Sentence: {args.num_samples}\n")
            f.write(f"Average COMET-22 (Higher is better): {comet_results.system_score:.4f}\n")
            f.write(f"Average MetricX-24 (Lower is better): {avg_metricx:.4f}\n")
            f.write(f"Total Candidate Count: {len(curr_translations)}\n")

        print(f"\nResults for {lang_key}:")
        print(f"Average COMET-22: {comet_results.system_score:.4f}")
        print(f"Average MetricX-24: {avg_metricx:.4f}")
        print(f"Results saved to: {results_file}")

        # if num_samples > 1: analyze results for translate-again
        if args.num_samples > 1:
            print(f"Analyzing hypotheses for translate-again data...")
            analyze_hypos(results_file, lang_key, remove_canary=(args.dataset=="wmt"))

if __name__ == "__main__":
    main()