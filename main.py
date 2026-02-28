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
from src.config import LANG_MAP, MODEL_ID_MAP, DEFAULT_MODEL
from src.translate import make_messages

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
    parser.add_argument("--rounds", type=int, default=1, 
                    help="Number of translation rounds (1 = standard, 2 = again, 3+ = again with more rounds)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, choices=MODEL_ID_MAP.keys())
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
            mode_suffix = f"_{args.rounds}rounds" if args.rounds > 1 else ""
            cache_file = f"outputs/{args.dataset}_translations_{lang_key}_{limit_str}{samples_str}{checkpt_suffix}{mode_suffix}.csv"

        # make new translations or load existing ones
        if args.force or not os.path.exists(cache_file):
            print(f"No valid cache found at {cache_file}. Running translation model...")
            
            # only load the model if needed
            if model is None:
                print("Loading model and processor...")
                model, processor = load_model_and_processor(checkpoint_path=args.checkpoint, model_name=args.model)
            
            # load dataset for this language
            lang_code = LANG_MAP[lang_key][args.dataset]
            if args.dataset == "wmt":
                sources, targets = load_wmt_data(lang=lang_code, limit=args.limit)
            else:
                sources, targets = load_flores_data(lang=lang_code, limit=args.limit)

            # translate (iteratively if rounds > 1)
            current_hypos = None
            all_round_results = {}

            for r in range(1, args.rounds + 1):
                print(f"--- Round {r} of {args.rounds} ---")
                
                if r == 1:
                    # round 1: standard translation/prompt
                    # returns list of dicts: [{'source', 'translation', 'likelihood'}, ...]
                    results_dicts = batch_translate(
                        model, processor, sources, 
                        model_name=args.model,
                        lang_key=lang_key, 
                        num_samples=args.num_samples, 
                        batch_size=2
                    )
                else:
                    # again rounds (with refeinement prompt)
                    # extend the original sources and likelihoods to match the number 
                    # of samples (for correct indexing in the next step)
                    if r == 2:
                        extended_sources = []
                        for s in sources:
                            extended_sources.extend([s] * args.num_samples)
                    
                    results_dicts = batch_translate(
                        model, processor, extended_sources, 
                        hypos=current_hypos,
                        model_name=args.model,
                        lang_key=lang_key, 
                        num_samples=1,
                        batch_size=2,
                        mode="again"
                    )
                    # restore original sources (for correct evaluation)
                    for i, res in enumerate(results_dicts):
                        res['source'] = extended_sources[i]

                # set hypothesis for next round
                current_hypos = [res['translation'] for res in results_dicts]
                
                # store results of this round
                all_round_results[f'translation_round{r}'] = current_hypos
                all_round_results[f'likelihood_round{r}'] = [res['likelihood'] for res in results_dicts]

            # put all round results in the final results dicts
            for i in range(len(results_dicts)):
                for r_name, r_list in all_round_results.items():
                    results_dicts[i][r_name] = r_list[i]
            
            # copy targets for num_samples>1 (multiple translations for same source)
            df = pd.DataFrame(results_dicts)
            expanded_targets = []
            for t in targets:
                expanded_targets.extend([t] * args.num_samples)

            # save results to cache file
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

        # evaluate other rounds (if rounds > 1)
        if args.rounds > 1:
            round_scores = {}
            for r in range(1, args.rounds):
                col = f"translation_round{r}"
                if col in df.columns:
                    print(f"Evaluating translations for round {r}...")
                    
                    comet_scores_r = comet22_eval(curr_sources, df[col].tolist(), curr_targets)
                    df[f"comet22_score_round{r}"] = comet_scores_r.scores

                    metricx_scores_r = metricx24_eval(curr_sources, df[col].tolist())
                    df[f"metricx24_score_round{r}"] = metricx_scores_r

                    round_scores[r] = (comet_scores_r.system_score, sum(metricx_scores_r) / len(metricx_scores_r))

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

            # optionally write other round results if rounds > 1
            if args.rounds > 1:
                for r in range(1, args.rounds):
                    if r in round_scores:
                        comet_r, metricx_r = round_scores[r]
                        f.write(f"\nRound {r} Results:\n")
                        f.write(f"Average COMET-22: {comet_r:.4f}\n")
                        f.write(f"Average MetricX-24: {metricx_r:.4f}\n")
            f.write(f"Total Candidate Count: {len(curr_translations)}\n")

        print(f"\nResults for {lang_key}:")
        print(f"Average COMET-22: {comet_results.system_score:.4f}")
        print(f"Average MetricX-24: {avg_metricx:.4f}")
        print(f"Results saved to: {results_file}")

        # if num_samples > 1: analyze results for translate-again
        if args.num_samples > 1 and args.rounds == 1:
            print(f"Analyzing hypotheses for translate-again data...")
            analyze_hypos(results_file, lang_key, remove_canary=(args.dataset=="wmt"))

if __name__ == "__main__":
    main()