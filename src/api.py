import time
import json
import pandas as pd
import os

from google import genai
import typing_extensions as typing

# setup for api
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("VERTEX_API_KEY")
client = genai.Client(
    #api_key=api_key,
    vertexai=True,
    project=os.getenv("VERTEX_PROJECT"),
    location="europe-west4"
)

# create json template for LLM response
class DetailedDifference(typing.TypedDict):
    difference_id: int
    location: typing.Literal["first_half", "second_half"]
    t1_version: str
    t2_version: str
    reason_for_change: str

class Metrics(typing.TypedDict):
    total_major_differences: int
    differences_in_first_half: int
    differences_in_second_half: int

class ComparativeAnalysis(typing.TypedDict):
    modification_density: typing.Literal["first_half_heavy", "second_half_heavy", "balanced"]

# combine previous 3 classes into final json object template
class TranslationAnalysis(typing.TypedDict):
    metrics: Metrics
    detailed_differences: list[DetailedDifference]
    comparative_analysis: ComparativeAnalysis

# custom instruction for model
system_instruction = """
Analyze English to Chinese translations by comparing T1 (initial) and T2 (refined) against the source.

Rules:
- Difference Detection: Identify major changes in wording, syntax, and semantics. Ignore minor punctuation or capitalization.
- Positional Logic: Divide the source sentence into two equal halves based on its length. Assign differences to 'first_half' or 'second_half' based on where the change occurs in the source string.
- Null Case: If T1 and T2 are identical strings, return 0 for all metrics and an empty list [] for detailed_differences.
"""

# load data and initialize empty results and output file
df = pd.read_csv('results_zh.csv')
total_rows = len(df)
all_results = []
out_name = "analysis_differences_zh.json"
print(f"Starting analysis for {len(df)} rows...")

# loop through datapoints in df
for index, row in df.iterrows():

    # enter input data
    user_msg = f"""
    TGT LANG: Chinese
    Source: {row['source']}
    T1: {row['translation_round1']}
    T2: {row['translation']}
    """

    try:
        # send prompt and force json template response
        # put system instruction here
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=user_msg,
            config=genai.types.GenerateContentConfig(
                system_instruction=system_instruction,
                response_mime_type="application/json",
                response_schema=TranslationAnalysis,
                temperature=0
            )
        )

        # add response to results array. also add source
        analysis = response.parsed
        analysis['source'] = row['source']
        analysis['sentence_id'] = index
        all_results.append(analysis)

        # intermediate save results
        if (index + 1) % 50 == 0:
            with open(out_name, "w", encoding="utf-8") as f:
                json.dump(all_results, f, indent=4, ensure_ascii=False)
            print(f"Results saved until row {index + 1}")

    # don't break entire loop if one datapoint has an error
    except Exception as e:
        print(f"Error on row {index}: {e}")
        with open(out_name, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=4, ensure_ascii=False)
        time.sleep(20)
        continue

    # timeout for api limit
    time.sleep(1.1)

# (final) save all results to json
with open(out_name, "w", encoding="utf-8") as f:
    json.dump(all_results, f, indent=4, ensure_ascii=False)

print(f"Done! Results saved to {out_name}")