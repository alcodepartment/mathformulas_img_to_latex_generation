import os
import csv

def clean_csvs(
  input_csvs=['train.csv', 'test.csv', 'val.csv'],
  images_dir='../data/images/',
  output_csvs=['cleaned_train.csv', 'cleaned_test.csv', 'cleaned_val.csv'],
  root_dir='../data/'
):
  """
  Remove images that doesn't exist from dfs.
  """

  for in_csv, out_csv in zip(input_csvs, output_csvs):

    in_path = os.path.join(root_dir, in_csv)
    out_path = os.path.join(root_dir, out_csv)

    print(f"\nCleaning: {in_path}")

    good = 0
    bad = 0

    with open(in_path, newline='', encoding='utf-8') as f_in, \
         open(out_path, "w", newline='', encoding='utf-8') as f_out:

      reader = csv.DictReader(f_in)
      writer = csv.DictWriter(f_out, fieldnames=reader.fieldnames)
      writer.writeheader()

      for row in reader:
        fname = row["image"]
        img_path = os.path.join(images_dir, fname)

        if os.path.isfile(img_path):
          writer.writerow(row)
          good += 1
        else:
          bad += 1

    print(f"  → Kept:    {good}")
    print(f"  → Removed: {bad}")
    print(f"  Saved cleaned CSV to: {out_path}")
