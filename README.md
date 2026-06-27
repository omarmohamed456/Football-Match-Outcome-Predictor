# Football Match Outcome Predictor

An end-to-end machine learning project covering the collection, processing, and prediction of football match outcomes across 37 competitions, spanning 16,688 matches scraped from us.soccerway.com across 37 domestic leagues, cups, and international competitions, covering the 2024–2025 and 2025–2026 seasons.  

The project is structured in three stages: a multi-stage web scraper that collects detailed per-match data, an exploratory analysis of the resulting 109-feature dataset, and a machine learning pipeline that predicts match outcomes — Home Win, Draw, or Away Win — across two distinct modelling approaches.  

The core modelling challenge is a practical one: in-match statistics such as shots, possession, and xG are only available after a game has started, making them useless for real pre-match prediction.  
To address this, the pipeline engineers rolling features from each team's last 5 matches as a proxy for form, and compares seven classifiers — Logistic Regression, Linear SVM, K-Nearest Neighbours, Gaussian Naive Bayes, XGBoost, and an Artificial Neural Network — across both an in-match benchmark and a pre-match model (the genuine real-world use case). Models are evaluated using per-class F1 score across all three outcome classes.

**Dataset:** [kaggle.com/datasets/omarameen99/football-matches-data-from-soccerway](https://www.kaggle.com/datasets/omarameen99/football-matches-data-from-soccerway)

---

## Table of Contents

- [Project Structure](#project-structure)
- [1. Web Scraping](#1-web-scraping)
  - [Requirements](#requirements)
  - [Link Scraper](#link-scraper)
  - [Match Scraper](#match-scraper)
  - [Full Scraper](#full-scraper)
  - [Retry Low Fields](#retry-low-fields)
  - [Combine CSVs](#combine-csvs)
- [2. Exploratory Data Analysis](#2-exploratory-data-analysis)
- [3. Match Outcome Modelling](#3-match-outcome-modelling)
  - [In-Match Model (Benchmark)](#in-match-model-benchmark)
  - [Pre-Match Model (Real-World Use Case)](#pre-match-model-real-world-use-case)

---

## Project Structure

```
project/
├── scraper/
│   ├── link_scraper.py          # Stage 1: collect match URLs
│   ├── match_scraper.py         # Stage 2: scrape match data
│   ├── full_scraper.py          # Orchestrator: runs link_scraper → match_scraper
│   ├── retry_low_fields.py      # Re-scrape incomplete or failed rows
│   ├── combine_csv.py           # Merge multiple data CSVs into one
│   │
│   ├── links/                   # .txt files produced by link_scraper.py
│   ├── scraped_links/           # .txt files moved here after scraping
│   ├── scraped_data/            # match data CSVs
│   └── scraped_logs/            # per-file scrape log CSVs
│
└── link_scraping_log/           # text files tracking which leagues were scraped
```

---

## 1. Web Scraping

**Source:** [us.soccerway.com](https://us.soccerway.com)

Scraping is split into two stages. The link scraper collects all match page URLs for a given league and season and stores them in plain text files. The match scraper then reads those files and extracts detailed match data from each URL, saving the results to CSV.

Both scripts share a set of design principles to reduce the risk of being blocked:

- A single Chrome driver instance is opened once and reused for the entire run, avoiding the overhead and fingerprinting risk of repeated browser launches.
- Randomised sleep intervals are applied between page loads, button clicks, and consecutive matches to simulate human browsing behaviour.
- The match scraper applies an additional random delay of 8–15 seconds between consecutive matches.

---

### Requirements

```bash
pip install selenium webdriver-manager pandas beautifulsoup4
```

Google Chrome must be installed on the system. `webdriver-manager` downloads the matching ChromeDriver automatically.

---

### Link Scraper

**Script:** `scraper/link_scraper.py`

Navigates to a league's results page, clicks "Show more matches" repeatedly until all matches are loaded, then extracts and saves every match URL. Only links from within match-row containers are collected; navigation links, team profile pages, and other incidental links on the page are excluded.

Output files are named after the league and season they correspond to, for example:

```
egypt_premier_league_2024-2025.txt
```

The `egypt_` prefix is used to distinguish the Egyptian Premier League from the English one. All 28 configured leagues follow the same naming convention.

**Configured leagues:**

| Region | Leagues |
|---|---|
| Egypt | Premier League, Egypt Cup |
| England | Premier League, Championship |
| France | Ligue 1, Ligue 2 |
| Germany | Bundesliga, 2. Bundesliga |
| Italy | Serie A, Serie B |
| Spain | LaLiga, LaLiga 2 |
| Netherlands | Eredivisie, Eerste Divisie |
| Europe | Champions League, Europa League, Conference League, Nations League |
| World | World Cup, Club World Cup |
| South America | Copa Libertadores, Copa Sudamericana |
| Mexico | Liga MX |
| Japan | J1 League, J2 League |
| Middle East | Saudi Pro League, Süper Lig |
| Africa | CAF Champions League |

**Modes and flags:**

| Flag | Description |
|---|---|
| `--link URL` | Scrape a single results page URL |
| `--file FILE` | Read one results-page URL per line from a `.txt` file and scrape each one |
| `--league KEY` | Scrape all configured seasons for the league whose key contains `KEY` (substring match) |
| `--prefix PREFIX` | Scrape all leagues whose filename prefix contains `PREFIX` (e.g. `germany_`) |
| `--output-dir DIR` | Directory to save output `.txt` files (default: `./links`) |
| *(no flags)* | Scrape all 28 configured leagues (current season + 2024–2025) |

**Examples:**

```bash
cd scraper

# Test a single results page
python link_scraper.py --link "https://us.soccerway.com/germany/bundesliga/results/"

# Scrape from a custom list of results URLs
python link_scraper.py --file my_urls.txt --output-dir ../links

# Scrape one configured league (both seasons)
python link_scraper.py --league bundesliga

# Scrape all German leagues
python link_scraper.py --prefix germany_

# Scrape all configured leagues
python link_scraper.py --output-dir ../links
```

---

### Match Scraper

**Script:** `scraper/match_scraper.py`

For each match URL, the scraper visits three tabs on the match page — Summary, Stats, and Lineups — and consolidates the data into a single row. Results are written to a CSV file incrementally after each match so that progress is not lost if the run is interrupted. A separate log CSV is written alongside the data file to record timing information, success status, and failure reasons for every URL.

**Data collected:**

- **Identity:** match ID, league/division, round, date, kick-off time, attendance, capacity, stadium, city, home team, away team, score, result
- **Tactical:** home/away formation, home/away team rating
- **Shots:** total, on/off target, inside/outside box, headed goals, hit woodwork
- **Set pieces:** corners, free kicks, throw-ins
- **Discipline:** fouls, offsides, yellow/red cards
- **Goalkeeper:** saves
- **Possession/Pressure:** possession %, touches in opposition box
- **Passing:** overall, long balls, final third passes (%, successful, total for each)
- **Crossing:** crosses (%, successful, total)
- **Other attacking:** big chances, duels won, errors leading to shot/goal, accurate through-passes
- **Defending:** tackles (%, successful, total), shots blocked, clearances, interceptions
- **Expected stats:** xG, xGOT, xA, xGOT faced, goals prevented

**File management:** after a `.txt` file has been fully processed, the scraper moves it from `links/` to `scraped_links/`. Data CSVs are saved to `scraped_data/` and log CSVs to `scraped_logs/`, each named after the source `.txt` file, for example:
```
egypt_premier_league_2024-2025_match_data.csv
egypt_premier_league_2024-2025_scrape_log.csv
```

**Log columns:**

`scraped_at`, `url`, `success`, `reason`, `non_empty_fields`, `t_summary_s`, `sleep_after_summary_s`, `t_stats_s`, `sleep_after_stats_s`, `t_lineups_s`, `t_scrape_only_s`, `t_total_s`, `sleep_before_next_match_s`, `cumulative_total_s`

The `reason` field records why a match was not scraped successfully, for example `no_data`, `403`, `404`, `no_stats`, `no_lineups`, or a truncated exception message.

**Flags:**

| Flag | Description |
|---|---|
| `--url URL` | Scrape a single match page URL |
| `--file FILE [FILE ...]` | One or more `.txt` files of match URLs; each file gets its own data and log CSV |
| `--dir DIR` | Directory to scan for `.txt` files; every `.txt` found is processed |
| `--output FILE` | Output data CSV name (only used with `--url`; default: `match_data.csv`) |
| `--log FILE` | Log CSV name (only used with `--url`; default: `scrape_log.csv`) |
| `--debug` | Save raw HTML to `debug_*.html` files for inspection |

**Examples:**

```bash
cd scraper

# Scrape a single match (useful for testing)
python match_scraper.py --url "https://us.soccerway.com/game/..."

# Scrape all matches in one file
python match_scraper.py --file ../links/germany_bundesliga_2024-2025.txt

# Scrape multiple files in one run
python match_scraper.py --file ../links/england_premier_league_2024-2025.txt \
                                   ../links/england_championship_2024-2025.txt

# Scrape every .txt file in a directory
python match_scraper.py --dir ../links

# Save a single match to a named file
python match_scraper.py --url "https://us.soccerway.com/game/..." \
                           --output bvb_vs_fcb.csv --log bvb_vs_fcb_log.csv

# Debug mode
python match_scraper.py --file ../links/test.txt --debug
```

---

### Full Scraper

**Script:** `scraper/full_scraper.py`

Orchestrates the full pipeline by running `link_scraper.py` followed by `match_scraper.py` in a single command. Useful when you want to go directly from a results page URL (or a list of them) to finished match data CSVs without running each stage separately.

**Modes:**

- **URL mode** — takes a single results-page URL, runs `link_scraper.py --link` to produce a `.txt` file, then immediately runs `match_scraper.py --file` on it.
- **File mode** — takes a `.txt` file containing one results-page URL per line, runs `link_scraper.py --file` on it, then runs `match_scraper.py --dir` on the links directory.

**Flags:**

| Flag | Description |
|---|---|
| `URL` | Results-page URL to scrape (URL mode) |
| `--file FILE` / `-f FILE` | Text file with one results-page URL per line (file mode) |
| `--links-dir DIR` | Folder used for intermediate `.txt` link files (default: `./links`) |

**Examples:**

```bash
cd scraper

# URL mode: scrape one results page end-to-end
python full_scraper.py "https://us.soccerway.com/germany/bundesliga/results/"

# File mode: scrape all URLs listed in a file
python full_scraper.py --file ../my_results_pages.txt

# Custom links directory
python full_scraper.py "https://us.soccerway.com/spain/laliga/results/" --links-dir ../links
```

---

### Retry Low Fields

**Script:** `scraper/retry_low_fields.py`

After a scraping run, some matches may have succeeded but returned an unusually low number of populated fields (e.g. stats or lineups tabs failed to load). This script reads one or more `*_scrape_log.csv` files, identifies rows that need retrying, re-scrapes those URLs, and patches the data CSV in place if the new scrape yields more data.

A row is retried if any of the following apply:
- `non_empty_fields` is below the threshold (default: 30)
- `reason` is `no_stats` or `no_lineups` (regardless of field count)

A unified retry log (`scraping_failed_urls_log.csv`) is written to the script directory and appended to on every run.

**Retry log columns:**

All original log columns plus: `file_name`, `fields_before`, `fields_after`, `row_updated`, `retry_trigger`, `data_csv_found`

**Flags:**

| Flag | Description |
|---|---|
| `--log LOG_CSV [...]` | One or more `*_scrape_log.csv` files to process |
| `--dir DIR` | Directory to scan for `*_scrape_log.csv` files (existing `*_retry_log*.csv` files are skipped) |
| `--threshold N` | Re-scrape rows with `non_empty_fields` < N (default: 30). `no_stats` and `no_lineups` rows are always retried regardless |
| `--debug` | Save raw HTML debug files for each re-scrape |

**Examples:**

```bash
cd scraper

# Retry a single log file
python retry_low_fields.py --log ../scraped_logs/germany_bundesliga_2024-2025_scrape_log.csv

# Retry all log files in a directory
python retry_low_fields.py --dir ../scraped_logs

# Custom threshold
python retry_low_fields.py --dir ../scraped_logs --threshold 25

# Retry multiple specific log files
python retry_low_fields.py --log ../scraped_logs/england_premier_league_2024-2025_scrape_log.csv \
                                  ../scraped_logs/spain_laliga_2024-2025_scrape_log.csv
```

---

### Combine CSVs

**Script:** `scraper/combine_csv.py`

Concatenates all CSV files in a folder into a single output file. Optionally extracts the season from each filename and inserts it as a column.

**Flags:**

| Flag | Description |
|---|---|
| `--folder DIR` | Path to folder containing CSV files (required) |
| `--output FILE` | Output CSV file name (default: `combined.csv`) |
| `--season` | Add a `season` column extracted from each source filename |

**Examples:**

```bash
cd scraper

# Combine all CSVs in scraped_data into one file
python combine_csv.py --folder ../scraped_data --output ../merged.csv

# Same but add a season column derived from each filename
python combine_csv.py --folder ../scraped_data --output ../merged.csv --season
```

---

## 2. Exploratory Data Analysis

**Notebook:** `football_eda.ipynb`
 
A structured analysis of the full dataset — 16,688 matches, 109 columns, 37 competitions, 1,353 teams — covering data quality, match patterns, team and formation statistics, and feature relationships.
 
**Dataset Overview** — shape, column inventory, data types, and duplicate check, followed by derived features (total goals, goal differential, total shots, total xG) and a numeric summary of core match statistics.
 
**Missing Data Analysis** — column-level and row-level audit of missingness. Columns are categorised by missing rate (complete / low / moderate / high / critical) and visualised as bar and circular bar plots. Home/away column pairs were confirmed to have symmetric missingness. A specific investigation into `headed_goals` found it was not scraped when both teams scored zero headed goals — structural missingness, not random noise. The row-level analysis showed the majority of rows have their core and advanced features fully populated.
 
**Outlier Check** — IQR-based detection across key numeric columns (goals, xG, shots, possession, cards, fouls), reporting outlier counts and percentages per column to confirm the data is within realistic footballing ranges.
 
**League & Season Distribution** — match counts broken down by league and season.
 
**Match Outcomes & Scoring** — distribution of Home Win / Draw / Away Win results; goals per match distribution; score frequency heatmap. Home wins account for the plurality of outcomes (~44.7%), with draws the least common (~24.7%), and (~30.6%) away wins.
 
**Home Advantage Analysis** — home win rate examined per league to assess whether the home advantage effect is consistent or varies meaningfully across competitions and regions.
 
**Shooting & xG Analysis** — shots on target vs goals relationship; xG vs actual goals to identify over- and under-performing teams; shot conversion rate distributions.
 
**Possession & Passing** — possession percentage distributions split by match result; passing accuracy compared between winning and losing sides.
 
**Discipline** — yellow and red card distributions; foul counts by result; league-level discipline comparisons.
 
**Team Formations** — most common formations across the dataset; win rate and goals per match by formation, smoothed with a Bayesian prior and filtered to formations with at least 700 appearances to avoid small-sample distortion.
 
**Correlation Heatmap** — lower-triangle Pearson correlation matrix across 20 key features including goals, xG, shots, possession, passing accuracy, corners, cards, and big chances.
 
**Top Teams Analysis** — top 15 teams by total goals scored and by Bayesian-smoothed win rate (minimum 10 games); bubble chart of goals scored vs conceded with win rate as colour and matches played as bubble size.  

---

## 3. Match Outcome Modelling

**Notebooks:** `in_match_data_based_models.ipynb` · `sliding_window_based_models.ipynb`

Both notebooks share the same preprocessing pipeline (steps 1–9): row filtering (dropping rows with >40% missing values), removal of outcome and identifier columns, column-level missingness re-check, median/mode imputation, datetime and kickoff time feature extraction, categorical label encoding, and final validation checks. They diverge in how they construct the feature set used for training.

### In-Match Model (Benchmark)

**Notebook:** `in_match_data_based_models.ipynb`

After the shared preprocessing, this pipeline retains all in-match statistics — shots, xG, possession, corners, cards, fouls, passing accuracy, and so on — and engineers differential features (home minus away) for each stat pair. These are combined with contextual features (league, round, formation, attendance, kickoff time) to produce the final feature matrix `df_preprocessed`.

This is a **benchmark**, not a deployable predictor. Because in-match stats are only available after kickoff, a model trained on them cannot be used to predict results before a match starts. Its purpose is to establish an upper-bound accuracy ceiling: how well can the outcome be predicted when all match information is available?

### Pre-Match Model (Real-World Use Case)

**Notebook:** `sliding_window_based_models.ipynb`

This pipeline is the real-world prediction scenario. After the shared preprocessing steps, all in-match statistics are dropped entirely. Instead, the pipeline reconstructs each team's recent form using a **sliding window of their last 5 matches** (regardless of home or away). For each match, rolling features are computed for both the home and the away team, capturing metrics such as recent points, goals scored and conceded, xG, shot volume, and passing accuracy over the preceding five appearances. These rolling features are combined with the pre-match contextual features (league, round, formation, kick-off time, attendance, team ratings) to produce `df_sliding`.

Because no information from the match itself is used, this model can genuinely be applied before a game kicks off.

**Both notebooks** train and evaluate the same six classifiers — Logistic Regression, Linear SVM, K-Nearest Neighbours, Gaussian Naive Bayes, XGBoost, and an Artificial Neural Network — using a strict chronological 80/20 train/test split with no random shuffling, to preserve temporal order and prevent future data leaking into training. Models are compared across accuracy, macro precision, macro recall, macro F1, ROC-AUC (one-vs-rest), and per-class F1 for Home Win, Draw, and Away Win.