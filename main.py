from src import (
    load_model_and_processor,
    load_wmt_data,
    batch_translate,
    comet22_eval
)

def main():
    # load model
    print("Loading model...")
    model, processor = load_model_and_processor()

    # load data
    print("Loading data...")
    sources, targets = load_wmt_data(lang="en-nl_NL", limit=50)

    # translate
    print(f"Translating {len(sources)} sentences...")
    translations = batch_translate(model, processor, sources)

    # evaluate on comet22
    print("Running COMET evaluation...")
    results = comet22_eval(sources, translations, targets)
    
    # show results
    print(f"Total scores captured: {len(results.scores)}")
    print(f"Average COMET-22 Score: {results.system_score:.4f}")

if __name__ == "__main__":
    main()