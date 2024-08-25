import pandas as pd
import json
import glob

def compute_statistics(pattern:str):
    print(pattern)
    directory_path = f"evaluation_output/{pattern}"
    all_data = []
    for filename in glob.glob(directory_path):
        with open(filename, "r") as file:
            data = json.load(file)
            all_data.append(data)

    df = pd.DataFrame(all_data)
    table_summary = df.describe()
    description = table_summary.to_csv('weiner.csv', index=True)


compute_statistics("*_weiner.json")
#compute_statistics("*_rnnoise_denoising.json")
