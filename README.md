# NLP-Driven Evaluation of Abstract Quality and Citation Counts

A runnable Streamlit dashboard prototype with a SQLite database and CSV dataset.

## Important data note
`data/articles.csv` and `data/articles.db` contain **synthetic demonstration records**, not real Scopus records. Journal names and citation counts are illustrative. Do not use these rows as research evidence or claim that they came from Scopus. Replace them with a lawful, verified Scopus export before reporting results.

## Features
- Abstract word count, sentence count, average sentence length
- Transparent heuristic abstract-quality score based on structural cues and length
- Filters by year, subject area, quartile, and citation count
- Scatter plot of score vs citations, word-count distribution, year trend
- Pearson correlation with log(1 + citations), Spearman correlation with raw citations
- Searchable article table and CSV export
- SQLite database (`data/articles.db`)

## Run on Windows
1. Install Python 3.10 or newer.
2. Extract the ZIP.
3. Open Command Prompt in the extracted folder.
4. Create a virtual environment (recommended):
   ```bat
   py -m venv .venv
   .venv\Scripts\activate
   ```
5. Install packages:
   ```bat
   pip install -r requirements.txt
   ```
6. Start the dashboard:
   ```bat
   streamlit run app.py
   ```
7. Open the local URL shown in the terminal (usually http://localhost:8501).

## Expected CSV columns
Required: `title`, `abstract`, `citations`
Recommended: `article_id`, `year`, `journal`, `quartile`, `subject_area`, `source`

You can export records from Scopus if you have appropriate institutional access and permissions. Keep DOI/EID identifiers where available, record the export date, and document citation-count retrieval date. Respect Scopus terms and your institution's data-use rules.

## Methodology cautions
- The quality score is a transparent heuristic prototype, not a validated NLP scale. It does not assess factual accuracy, novelty, or scientific rigor.
- Citation counts depend strongly on time since publication, field, journal, document type, access status, and other factors.
- Correlation is not causation. For a stronger study, model citations with negative-binomial regression (often suitable for overdispersed counts) and control for publication year/age, subject area, journal, and document type.
- The app currently displays unadjusted Pearson and Spearman correlations. It is intended as a starting point for research, not a validated analysis pipeline.
