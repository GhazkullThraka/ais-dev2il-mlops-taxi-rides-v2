# AIS DEV2IL 😈 MLOps: Taxi Rides

Welcome to the MLOps Taxi Rides exercises! You are going to explore real-world MLOps practices — 
data management, model training, experiment tracking, and serving predictions — all using a dataset 
of [New York City taxi rides](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page).

---

## 🛫 Getting Started

1. Fork [this repository](https://github.com/peterrietzler/ais-dev2il-mlops-taxi-rides-v2), clone your fork and open it in PyCharm.
2. Make sure you are working on the `main` branch.
3. Install dependencies:
   ```bash
   uv sync
   ```

You're ready to go! 🚀

---

## 📦 Exercise 1: Feel the Pain — CSV vs Parquet

Before we talk about why Parquet is great, let's experience what life looks like without it.

In the `example-data` folder you'll find the same dataset in two formats:
- `2025-01-01.taxi-rides.csv`
- `2025-01-01.taxi-rides.parquet`

Create a new Python file called `compare.py` in the root of the repository and work through the steps below together with your pair.

### Step 1: Load the CSV and inspect the schema

```python
import pandas as pd

df_csv = pd.read_csv('example-data/2025-01-01.taxi-rides.csv')
print(df_csv.dtypes)
```

Run it with:

```bash
uv run compare.py
```

🤔 Look at the column types carefully:
- What type are `tpep_pickup_datetime` and `tpep_dropoff_datetime`? Are those really the right types for timestamps?
- What type is `ride_time`? It should be a number — but is it?

Open the CSV file in PyCharm and look at the first row. Notice anything odd about `ride_time`?

Someone put `"unknown"` in that field. Because CSV has no schema, pandas silently loaded the entire
column as `object` (string) — **without any warning**. Your data is now silently broken. 😬

### Step 2: Load the Parquet and inspect the schema

```python
df_pq = pd.read_parquet('example-data/2025-01-01.taxi-rides.parquet')
print(df_pq.dtypes)
```

✅ The datetime columns have the correct type — no extra code needed. The schema is embedded in the file itself.

And that `"unknown"` value? It could never end up in a Parquet file in the first place.
Parquet enforces the column type on **write** — if the data doesn't match, it fails loudly instead of silently corrupting your dataset.

### Step 3: Compare file sizes

```python
import os

csv_size = os.path.getsize('example-data/2025-01-01.taxi-rides.csv')
pq_size  = os.path.getsize('example-data/2025-01-01.taxi-rides.parquet')

print(f'CSV:     {csv_size / 1024 / 1024:.1f} MB')
print(f'Parquet: {pq_size  / 1024 / 1024:.1f} MB')
print(f'Parquet is {csv_size / pq_size:.1f}x smaller')
```

💡 How much disk space would you save if you stored a full year of taxi data as Parquet instead of CSV?

### Step 4: Time the loads

```python
import time

t = time.time()
pd.read_csv('example-data/2025-01-01.taxi-rides.csv')
csv_time = time.time() - t

t = time.time()
pd.read_parquet('example-data/2025-01-01.taxi-rides.parquet')
pq_time = time.time() - t

print(f'CSV:     {csv_time:.3f}s')
print(f'Parquet: {pq_time:.3f}s')
```

🗣️ **Discuss with your pair:** You are building an ML pipeline that retrains a model every day.
What problems would these differences cause over time?

### 🚀 Level Up

#### Challenge 1: Chuck Norris Never Needs `pd.to_datetime()`

> *Chuck Norris doesn't parse dates. Dates parse themselves in his presence.*

When you loaded the CSV, the datetime columns came in as plain strings (`object`). Fix them manually to match the Parquet schema:

```python
df_csv = pd.read_csv('example-data/2025-01-01.taxi-rides.csv')
df_csv['tpep_pickup_datetime']  = pd.to_datetime(df_csv['tpep_pickup_datetime'])
df_csv['tpep_dropoff_datetime'] = pd.to_datetime(df_csv['tpep_dropoff_datetime'])
print(df_csv.dtypes)
```

Now try to fix `ride_time` the same way:

```python
df_csv['ride_time'] = pd.to_numeric(df_csv['ride_time'])
print(df_csv.dtypes)
```

💥 It crashes with a `ValueError`. And that's actually the *right* reaction — you want to train your model on 
valid data. A `ride_time` of `"unknown"` is useless for training and silently coercing it to `NaN` would just hide the problem deeper.

But here's the thing: **you should never have had to deal with this in the first place.** Parquet enforces the schema on 
write — `"unknown"` could never have ended up in a float column. The problem is caught at the source, not discovered halfway 
through your training pipeline.

#### Challenge 2: A Year Worth of Taxi Rides

In ML training, you rarely load just one day of data — you load months or even a full year.
Simulate this by loading the same file 365 times in a loop for both formats:

```python
import time
import pandas as pd

t = time.time()
for _ in range(365):
    pd.read_csv('example-data/2025-01-01.taxi-rides.csv')
csv_time = time.time() - t

t = time.time()
for _ in range(365):
    pd.read_parquet('example-data/2025-01-01.taxi-rides.parquet')
pq_time = time.time() - t

print(f'CSV:     {csv_time:.1f}s')
print(f'Parquet: {pq_time:.2f}s')
print(f'Parquet is {csv_time / pq_time:.0f}x faster')
```

⏱️ Every time you retrain your model, you'd be waiting those extra seconds just reading data.
At scale, this adds up fast.

#### Challenge 3: No Code? No Problem!

Explore the Parquet file without writing a single line of Python:

1. In PyCharm's **Project** view, open `example-data/2025-01-01.taxi-rides.parquet`
2. Click **"Edit in Data Wrangler"**
3. Try some transformations:
   - Filter out rows where `outlier` is `True`
   - Sort by `trip_distance` descending
   - Drop the `ride_time` column
4. Click **"Export"** to generate a Python script from your actions
5. Run the generated script with `uv run <script_name>.py`

> Data Wrangler lets you explore visually and then hands you the Python code for free —
> great for getting started with a new dataset fast.

---

## 🗄️ Exercise 2: Data Management with DVC

You now have Parquet files — great format, right size. But here's the next problem: **how do you share large data files with your team?**

Git is designed for code, not data. Try to `git add` a folder with 30 Parquet files and Git will dutifully track every byte forever, bloating your repository. There has to be a better way.

Enter **DVC** (Data Version Control) — it works alongside Git. Git tracks your code and tiny pointer files. DVC tracks your actual data files in a separate storage. Same workflow, sane repository.

### Step 1: Get the data

The file `example-data/data.zip` is already in your repository. Extract it into a `data` folder in the root of your repository.
If you are using PyCharm, make sure to **not** add the files to Git.

Your folder structure should look like this:
```
data/
  2025-01-01.taxi-rides.parquet
  2025-01-02.taxi-rides.parquet
  ...
```

### Step 2: Create your DagsHub repository

[DagsHub](https://dagshub.com) is a collaboration platform built for data scientists and ML engineers. 
Think of it as GitHub — but with built-in support for large data files, experiment tracking, 
and model registries. For now we'll use it purely as a **DVC remote storage** — a place to store and share our Parquet files.

1. Go to [https://dagshub.com](https://dagshub.com) and sign in or up with your GitHub account
2. Click **"Create Repository"** → **"Connect a repository"** → select your GitHub fork
3. In your new DagsHub repo, go to **Your Settings** (in your profile menu in the upper right corner) 
   → *Tokens** and copy the default access token. You will need it a bit later.

### Step 3: Initialise DVC

```bash
uv run dvc init
uv run dvc config core.autostage true
```

This creates a `.dvc` folder (similar to `.git`). The `autostage` setting tells DVC to automatically stage `.dvc` pointer files in Git when you run `dvc add` — one less thing to remember.

### Step 4: Configure the DVC remote

Run these commands. Don't forget to replace the placeholders with your actual values using your username, repository name and the token
that you copied above.  Alternatively you can go to your DagsHub repository **Remote** → **DVC** and copy the exact commands from there.

```bash
uv run dvc remote add origin s3://dvc
uv run dvc remote modify origin endpointurl https://dagshub.com/<YOUR USERNAME>/<YOUR REPO>.s3
uv run dvc remote modify origin --local access_key_id <YOUR TOKEN>
uv run dvc remote modify origin --local secret_access_key <YOUR TOKEN>
uv run dvc remote default origin
```

Here's what each command does:

- **`dvc remote add origin s3://dvc`** — registers a new DVC remote called `origin` using the S3 protocol.
- **`dvc remote modify origin endpointurl ...`** — tells DVC where the S3 storage actually lives. DagsHub provides an S3-compatible HTTP endpoint for every repository.
- **`dvc remote modify origin --local access_key_id ...`** — sets your DagsHub token as the S3 access key. The `--local` flag stores this in `.dvc/config.local`, which is gitignored and **never committed**.
- **`dvc remote modify origin --local secret_access_key ...`** — same token again, used as the S3 secret key. DagsHub uses one token for both.
- **`dvc remote default origin`** — marks `origin` as the default remote. Without this, you'd have to specify `-r origin` on a lot of commands

### Step 5: Track Parquet Files

```bash
uv run dvc add data/*.parquet
```

DVC creates one `.dvc` pointer file per Parquet file (e.g. `data/2025-01-01.taxi-rides.parquet.dvc`) and adds `data/*.parquet` to `data/.gitignore` automatically.
Look them up in the data folder. Also have a look at the contents of `data/.gitignore`

Add the data folder to Git:

```bash
git add data
```

Run `git status` — Git now tracks all the `.dvc` pointer files and the updated `.gitignore`, but **not** the raw Parquet files. 
Each pointer file is very small and tells DVC where to find the actual data. The Parquet files are still there on your disk, but Git is 
blissfully unaware of them and will not store them.

### Step 6: Check what DVC wants to push

Before pushing, check which files DVC needs to upload to the remote:

```bash
uv run dvc status --cloud
```

You should see all 30 Parquet files listed as `new file` — they exist locally but haven't been pushed yet. 
After `dvc push` in the next step, running this again will show nothing, meaning everything is in sync.

### Step 7: Push the data

```bash
uv run dvc push
```

Your Parquet files are now stored in your DagsHub remote. DVC uploads them directly to the remote storage without going through Git at all.

### Step 8: Commit the pointer files to Git

```bash
git add data/
git commit -m "Track data files with DVC"
git push
```

### Step 9: "Gone but not Forgotten"

Delete all the Parquet files from your `data` folder:

```bash
rm data/*.parquet
```

Run `dvc pull` to restore them:

```bash
uv run dvc pull
```

🎉 The files are back. Your data is safe, versioned and shareable.

> **💡 The DVC Workflow — every time you pull or switch branches**
>
> Whenever you run `git pull` or `git checkout <branch>`, Git updates the `.dvc` pointer files — but **not** the actual data files. You need to follow up with:
>
> ```bash
> git pull          # updates .dvc pointer files
> uv run dvc pull   # downloads the matching data from the remote
> ```
>
> Think of `.dvc` files as Git's way of saying *"the data should look like this"* — and `dvc pull` as the step that makes it actually happen.

### 🚀 Level Up

#### Challenge 1: Gone but not Forgotten (Remote Edition)

On your **pair's machine**, clone your GitHub fork, configure the DVC remote credentials and run:

```bash
uv run dvc pull
```

Does it work? Your pair should now have the exact same data as you — pulled from *your* DagsHub remote. This is how teams share data without emailing zip files around.

#### Challenge 2: Catch the Change

Let's simulate updating your dataset. Create a small Python script that:
1. Loads `data/2025-01-01.taxi-rides.parquet`
2. Filters out all rows where `outlier` is `True`
3. Saves it back to the same file

Now use DVC to detect what changed:

```bash
uv run dvc status
```

You should now see that the file `data/2025-01-01.taxi-rides.parquet` has changed locally. 
DVC detects this because the hash of the file content has changed. In order to get Git and 
the DVC remote back in sync, you need to add the updated file to DVC, push it to the remote, and commit the pointer file to Git.

Then make sure everything lands correctly in both DVC and Git:

```bash
uv run dvc add data/2025-01-01.taxi-rides.parquet
uv run dvc push
git add data/2025-01-01.taxi-rides.parquet.dvc
git commit -m "Update data: filter outliers, add February data"
git push
```

🤔 **What happens if you forget to `dvc push` before `git push`?**
Your teammate runs `dvc pull` and gets an error — the pointer in Git points to 
data that doesn't exist in the remote yet. Always push DVC before Git!