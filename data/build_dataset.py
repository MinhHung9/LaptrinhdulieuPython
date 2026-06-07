import os
import pandas as pd

def load_folder(folder_path, split_name):
    records = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            label = filename.replace('.txt', '')
            filepath = os.path.join(folder_path, filename)
            with open(filepath, encoding='utf-8', errors='ignore') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append({
                            'review_text': line,
                            'label': label,
                            'split': split_name
                        })
    return records

# Load train (1a3) và test (1a6)
train_records = load_folder('data/1a3', 'train')
test_records  = load_folder('data/1a6', 'test')

df_train = pd.DataFrame(train_records)
df_test  = pd.DataFrame(test_records)
df_all   = pd.concat([df_train, df_test], ignore_index=True)

# Lưu file
df_train.to_csv('data/train.csv', index=False, encoding='utf-8-sig')
df_test.to_csv('data/test.csv',   index=False, encoding='utf-8-sig')
df_all.to_csv('data/tiki_reviews.csv', index=False, encoding='utf-8-sig')

print(f" Train: {len(df_train):,} dòng")
print(f" Test : {len(df_test):,} dòng")
print(f"\nPhân phối nhãn (train):")
print(df_train['label'].value_counts())