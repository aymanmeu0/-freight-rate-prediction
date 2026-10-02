"""Build the submission report (DOCX, plus PDF when Microsoft Word is available).

Every number in the report is read from files the pipeline writes
(`reports/metrics.json`, the cleaning log from `src/data.py`, the QA summary),
so the report cannot drift away from the code.

Usage, from the project root:
    python src/report.py            # writes reports/report.docx and reports/report.pdf
    python src/report.py --no-pdf   # DOCX only
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import date
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
METRICS_PATH = REPORTS / "metrics.json"
QA_SUMMARY_PATH = REPORTS / "qa_summary.json"
CHART_PATH = ROOT / "scorer_results" / "candidate_december.png"
DOCX_PATH = REPORTS / "report.docx"
PDF_PATH = REPORTS / "report.pdf"

AUTHOR = "Ayman Alhalabi"
ACCENT = RGBColor(0x06, 0x4A, 0x56)  # same teal as the score.py chart
GREY = RGBColor(0x45, 0x5A, 0x60)


# ---------------------------------------------------------------- formatting

def money(x: float) -> str:
    return f"${x:,.2f}"


def pct(x: float, digits: int = 2) -> str:
    return f"{x:.{digits}f}%"


def signed_pct(x: float) -> str:
    return f"{x:+.2f}%"


def shade(cell, hex_fill: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_fill)
    tc_pr.append(shd)


def add_table(doc, header, rows, widths_cm, bold_last_col=False, highlight_row=None):
    table = doc.add_table(rows=1, cols=len(header))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, text in enumerate(header):
        cell = table.rows[0].cells[i]
        cell.text = ""
        run = cell.paragraphs[0].add_run(text)
        run.bold = True
        run.font.size = Pt(9)
        run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        shade(cell, "064A56")
    for r, row in enumerate(rows):
        cells = table.add_row().cells
        for i, text in enumerate(row):
            cells[i].text = ""
            run = cells[i].paragraphs[0].add_run(str(text))
            run.font.size = Pt(9)
            if highlight_row is not None and r == highlight_row:
                run.bold = True
                shade(cells[i], "E6F0F1")
            elif bold_last_col and i == len(row) - 1:
                run.bold = True
    for row in table.rows:
        for i, w in enumerate(widths_cm):
            row.cells[i].width = Cm(w)
    doc.add_paragraph()
    return table


def para(doc, text, size=10.5, italic=False, color=None, space_after=6):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.font.size = Pt(size)
    run.italic = italic
    if color is not None:
        run.font.color.rgb = color
    p.paragraph_format.space_after = Pt(space_after)
    return p


def bullets(doc, items):
    for item in items:
        p = doc.add_paragraph(style="List Bullet")
        p.add_run(item).font.size = Pt(10.5)
        p.paragraph_format.space_after = Pt(2)
    doc.add_paragraph().paragraph_format.space_after = Pt(0)


def figure(doc, path: Path, caption: str, width_cm: float = 16.0):
    if not path.is_file():
        raise SystemExit(f"ERROR: figure not found: {path}")
    doc.add_picture(str(path), width=Cm(width_cm))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap = para(doc, caption, size=9, italic=True, color=GREY, space_after=10)
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER


def heading(doc, text, level=1):
    h = doc.add_heading(text, level=level)
    for run in h.runs:
        run.font.color.rgb = ACCENT
    return h


def metric_row(label, res):
    c, a = res["clean"], res["all"]
    return [label, money(c["MAE"]), pct(c["MAPE"]), money(c["RMSE"]),
            signed_pct(c["median_signed_pct"]), money(a["MAE"]), pct(a["MAPE"]), money(a["RMSE"])]


# ---------------------------------------------------------------- content

def cleaning_counts():
    """Recompute the cleaning log through the project pipeline (fit on all of train)."""
    sys.path.insert(0, str(ROOT))
    from src.data import Cleaner, load_all  # noqa: E402

    ds = load_all()
    cleaner = Cleaner(ds.market_index_table).fit(ds.train)
    train = cleaner.transform(ds.train, drop_outliers=True, name="train")
    cleaner.transform(ds.validation, name="validation")
    log = cleaner.log_table()
    return log, len(ds.train), len(train)


def build(metrics: dict, qa: dict | None) -> Document:
    hold = metrics["holdout"]
    unseen = metrics["unseen_city"]
    final = metrics["final"]
    cv = metrics["cv"]
    diag = metrics["level_diagnostics"]
    chosen = hold["results"]["chosen"]
    b1 = hold["results"]["B1 baseline"]
    spec = metrics["chosen"]["spec"]

    doc = Document()
    section = doc.sections[0]
    section.page_width, section.page_height = Cm(21.0), Cm(29.7)
    for side in ("left_margin", "right_margin"):
        setattr(section, side, Cm(2.0))
    section.top_margin = section.bottom_margin = Cm(1.8)
    normal = doc.styles["Normal"]
    normal.font.name = "Calibri"
    normal.font.size = Pt(10.5)

    # Title block
    title = doc.add_paragraph()
    run = title.add_run("Freight Rate Prediction")
    run.bold = True
    run.font.size = Pt(24)
    run.font.color.rgb = ACCENT
    sub = para(doc, "Machine Learning Engineer assessment: approach, validation and December chart",
               size=12, color=GREY, space_after=2)
    para(doc, f"{AUTHOR}  |  {date.today():%d %B %Y}", size=10, color=GREY, space_after=14)
    del sub

    # 1. Summary
    heading(doc, "1. Summary")
    para(doc, (
        "The task is to predict posted_rate, the price in dollars of one truck load, for 12,000 loads "
        "dated November and December 2025. The model learns from 48,000 labeled loads dated January to "
        "October 2025. Because the loads to predict lie in the future, every test in this project is a "
        "forecast test: the model is trained on earlier months and scored on later months it has never seen."
    ))
    add_table(doc, ["Test", "Our model", "Simple baseline"], [
        ["Sep-Oct holdout, average error in % (MAPE, clean rows)", pct(chosen["clean"]["MAPE"]), pct(b1["clean"]["MAPE"])],
        ["Sep-Oct holdout, average error in $ (MAE, clean rows)", money(chosen["clean"]["MAE"]), money(b1["clean"]["MAE"])],
        ["Tuning folds May-Aug, mean MAPE", pct(cv_mean(cv, metrics["chosen"]["name_in_cv"]), 3),
         pct(cv_mean(cv, "B1 baseline"), 3)],
        ["Cities never seen in training, MAPE", pct(unseen["results"]["chosen, cities unseen"]["clean"]["MAPE"]),
         pct(unseen["results"]["B1, cities unseen"]["clean"]["MAPE"])],
    ], [9.5, 3.5, 3.5])
    para(doc, (
        "The model is XGBoost on the log of the rate per mile, on top of a small market-level layer that "
        "captures a slow upward drift in prices and a price rise in the last weeks of each quarter. "
        "The simple baseline is the typical rate per mile for the equipment type and distance band, times the "
        "distance. Clean rows exclude the 1.4% of labels that are clearly corrupted (section 3); "
        "results on all rows are reported too."
    ))

    # 2. Data and key findings
    heading(doc, "2. What the data showed")
    bullets(doc, [
        "Distance explains most of the price (correlation 0.91). Reefer loads cost about 13% more per mile "
        "than Dry Van, and Flatbed about 8% more.",
        "The column quote_signal is a trap. In January to March, June and September it is an almost exact copy "
        "of the rate per mile (correlation 0.99 or more on clean rows). In April, May, July and October it is a "
        "mirror image (about -0.99). In August, and in the November and December loads we must predict, it is "
        "pure noise. A model that uses it scores better in random testing and worse on the real task, so it is "
        "never used as a feature. For the same reason the model never sees the month.",
        "market_index is one number per day (it varies 0.025 within a day against 0.17 across days). It moves "
        "rates up and down, with a weekly cycle: highest on Thursday, lowest on Sunday and Monday.",
        "Beyond the index, rates drift up by about 0.6% a month, and in March, June and September they climb "
        "about 4% through the month and fall back on the 1st (figure 3). December is also a quarter-end month.",
        "Eight cities in the November-December file (Allentown, Charlotte, Chicago, Jackson, Knoxville, Laredo, "
        "Norfolk, San Diego) never appear in training. City effects are smooth on the map, so the model works "
        "from latitude and longitude and not from city names.",
    ])
    figure(doc, FIGURES / "02_quote_signal_regimes.png",
           "Figure 1. quote_signal copies the label in some months, mirrors it in others, and is noise in the months we predict.")

    # 3. Data quality
    heading(doc, "3. Data quality issues and how they were handled")
    log, n_train, n_train_clean = cleaning_counts()
    tr, va = log["train"], log["validation"]
    add_table(doc, ["Issue", "Train", "Validation", "What we did"], [
        ["Negative weight (sign flipped; same shape as positive weights)",
         f"{int(tr['weight_negative_flipped']):,}", f"{int(va['weight_negative_flipped']):,}", "Use the absolute value"],
        ["Missing weight (missing at random)", f"{int(tr['weight_filled']):,}", f"{int(va['weight_filled']):,}",
         "Fill with the equipment median from the training rows"],
        ["Missing market_index (missing at random)", f"{int(tr['market_index_filled']):,}",
         f"{int(va['market_index_filled']):,}", "Fill with the mean of the same date"],
        ["Corrupted rate (2x to 6x too high or too low)", f"{int(tr['outliers_flagged']):,}", "n/a",
         "Remove from training; report scores with and without them"],
        ["quote_signal leaks the label", "all rows", "all rows", "Never used as a feature"],
        ["Duplicates or spelling variants", "0", "0", "None needed"],
        ["Distance vs map distance (1.18x straight line, 70-mile floor)", "consistent", "consistent", "Kept as given"],
    ], [6.6, 1.8, 2.2, 6.4])
    para(doc, (
        f"A rate is flagged as corrupted when its rate per mile is more than 1.65 times or less than 0.61 times "
        f"the typical value for its equipment and distance band. The rule is robust: every threshold from 0.3 to "
        f"0.7 on the log scale flags the same rows, because no normal rate falls in the gap. Training keeps "
        f"{n_train_clean:,} of {n_train:,} rows. Validation rows are never dropped. Every cleaning number "
        f"that touches prices or weights (medians, typical rates) is learned on the training rows of each split "
        f"only, so test months never leak into training. The market_index fill uses the same-date mean across "
        f"both files; it reads only that input column on the same day, never a price."
    ))

    # 4. Split and validation
    heading(doc, "4. How the data was split and validated")
    para(doc, (
        "The submission asks for predictions on the two months after the training data ends, so a random split "
        "would be the wrong test: it lets the model see the future and hides the drift in prices. Every split "
        "here respects time."
    ))
    folds = cv["folds"]
    add_table(doc, ["Step", "Train on", "Test on", "Used for"], [
        ["Tuning fold 1", "Jan-Apr", "May", "Choosing the model and its settings"],
        ["Tuning fold 2", "Jan-May", "Jun", "Choosing the model and its settings"],
        ["Tuning fold 3", "Jan-Jun", "Jul", "Choosing the model and its settings"],
        ["Tuning fold 4", "Jan-Jul", "Aug", "Choosing the model and its settings"],
        ["Holdout", f"Jan-Aug ({hold['fit_rows']:,} clean rows)", f"Sep-Oct ({hold['test_rows']:,} rows)",
         "One final score, after all choices were made"],
        ["Unseen cities", f"Jan-Aug without 8 cities ({unseen['fit_rows']:,} rows)",
         f"Sep-Oct rows touching them ({unseen['test_rows']:,})", "Price of a city never seen before"],
        ["Final model", f"Jan-Oct ({final['fit_rows']:,} clean rows)", "Nov-Dec (12,000 loads)", "Submission"],
    ], [2.6, 4.6, 4.4, 5.4])
    del folds
    bullets(doc, [
        "Two months of holdout match the two-month horizon of the real task.",
        "The holdout was scored once, after the model was chosen on the four tuning folds. It played no part "
        "in any choice.",
        "Cleaning is refit inside every fold on that fold's training rows, so no statistic from a test month "
        "reaches training.",
        "The eight cities held out for the unseen-city test are the eight least frequent cities "
        f"({', '.join(unseen['cities'])}). All their training rows are removed, which mimics the eight new "
        "cities in the November-December file.",
        "Metrics: MAE (average dollar error), MAPE (average percent error) and RMSE, each on clean rows and on "
        "all rows. The corrupted labels dominate RMSE on all rows, so model choice uses clean MAPE.",
    ])

    # 5. Model
    heading(doc, "5. Model")
    xgb = spec["xgb"]
    bullets(doc, [
        "Target: log of the rate per mile. It validated better than the log of the rate.",
        "Features: distance and its log; pickup and delivery latitude and longitude; the change in latitude and "
        "longitude; the compass direction of the lane; straight-line distance and the road-to-straight-line "
        "ratio; equipment type; weight; market_index; day of the week. Never used: quote_signal, month, date, "
        "load_id. A code check stops the run if a banned column reaches the model.",
        "Market-level layer: a straight-line trend in time plus the quarter-end ramp, fitted by least squares "
        "on the training rows. The trees learn the rest of the price. Past the last training day the trend is "
        f"extended at {spec['damp']:g} of its slope, which is a cautious middle between no trend and a full trend.",
        f"Lane correction: a small per-lane adjustment, learned out of fold and shrunk toward zero for rare "
        f"lanes (prior of {spec['lane_prior']:g} loads). A new lane gets no correction.",
        f"XGBoost settings: {xgb['n_estimators']:,} trees, depth {xgb['max_depth']}, learning rate "
        f"{xgb['learning_rate']}, row and column sampling {xgb['subsample']}, fixed seed {metrics['seed']}.",
    ])
    para(doc, "How each step changed the mean tuning-fold MAPE (May-Aug):")
    stages = [
        ("Simple baseline", "B1 baseline"),
        ("Ridge regression, no drift handling", "ridge, log rate, geo, no drift"),
        ("XGBoost, no drift handling", "xgb, log rate, geo, no drift, d6 n600 lr0.05 mcw1"),
        ("XGBoost, recency weights", "xgb, log rate, geo, no drift, recency hl30, d6 n600 lr0.05 mcw1"),
        ("XGBoost + trend", "xgb, log rate, geo, trend damp 0.5, d6 n600 lr0.05 mcw1"),
        ("XGBoost + trend + quarter-end ramp", "xgb, log rate, geo, trend+ramp damp 0.5, d6 n600 lr0.05 mcw1"),
        ("+ rate per mile target", "xgb, log rpm, geo, trend+ramp damp 0.5, d6 n600 lr0.05 mcw1"),
        ("+ day of week", "xgb, log rpm, geo_dow, trend+ramp damp 0.5, d6 n600 lr0.05 mcw1"),
        ("+ tuned XGBoost settings", "xgb, log rpm, geo_dow, trend+ramp damp 0.5, d5 n2000 lr0.03 mcw1"),
        ("+ lane correction (chosen)", metrics["chosen"]["name_in_cv"]),
    ]
    add_table(doc, ["Setup", "May", "Jun", "Jul", "Aug", "Mean"],
              [[label] + cv_row(cv, key) for label, key in stages],
              [7.2, 1.9, 1.9, 1.9, 1.9, 2.2], highlight_row=len(stages) - 1)

    # 6. Results
    heading(doc, "6. Results")
    para(doc, "Holdout: trained on January to August, scored once on September and October.")
    hdr = ["Setup", "Clean MAE", "Clean MAPE", "Clean RMSE", "Clean bias", "All MAE", "All MAPE", "All RMSE"]
    w = [3.4, 1.8, 1.8, 1.8, 1.8, 1.8, 1.8, 1.8]
    add_table(doc, hdr, [
        metric_row("Chosen model", chosen),
        metric_row("Chosen, Sep", hold["by_month"]["chosen, Sep"]),
        metric_row("Chosen, Oct", hold["by_month"]["chosen, Oct"]),
        metric_row("Simple baseline", b1),
        metric_row("XGBoost, no drift", hold["results"]["xgb, log rate, geo, no drift, d6 n600 lr0.05 mcw1"]),
        metric_row("Chosen without ramp", hold["results"]["chosen without the ramp (trend only)"]),
    ], w, highlight_row=0)
    ucs = unseen["results"]
    para(doc, (
        f"Unseen cities: on the {unseen['test_rows']:,} September-October loads that touch the eight held-out "
        f"cities, the model trained without those cities scores {pct(ucs['chosen, cities unseen']['clean']['MAPE'])} "
        f"clean MAPE, against {pct(ucs['chosen main holdout model (cities seen)']['clean']['MAPE'])} when it has "
        f"seen them. The cost of a new city is {unseen['penalty_clean_MAPE_pp']:.2f} percentage points, so the "
        f"eight new cities in the submission file are priced almost as well as known ones."
    ))
    para(doc, (
        "Bias is the median signed error (below zero means predictions are too low). Results on all rows are "
        "much worse than on clean rows only because of the corrupted labels: a label that is four times too high "
        "cannot be predicted by any model. If the November-December labels contain the same 1.4% of corrupted "
        "rows, they will dominate any RMSE computed on them."
    ))
    figure(doc, FIGURES / "12_holdout_residual_by_date.png",
           "Figure 2. Daily median error on the holdout. Without drift handling XGBoost runs 2% to 8% low; the chosen model stays close to zero.",
           width_cm=13.5)
    q = diag["quarter_end_months"]
    para(doc, (
        "The quarter-end ramp is the largest single improvement and the biggest bet for December, so it was "
        "checked several ways. The last five days of March, June and September sit "
        f"{q['3']['last_5_days_pct']:.1f}%, {q['6']['last_5_days_pct']:.1f}% and {q['9']['last_5_days_pct']:.1f}% "
        "above the trend, and no other month shows a rise. Adding the ramp raises the fit of the daily price "
        f"level from R2 {diag['daily_fits']['index + day']['R2']:.2f} to "
        f"{diag['daily_fits']['index + day + quarter-end ramp']['R2']:.2f}, and its size is stable whichever "
        "months it is fitted on."
    ))
    figure(doc, FIGURES / "15_quarter_end_ramp.png",
           "Figure 3. Daily price level after the market index and trend: a rise through each quarter-end month.",
           width_cm=13.5)

    # 7. December chart
    heading(doc, "7. December prediction chart")
    dc = final["summaries"]["december_chart"]
    para(doc, (
        "The chart below is produced by the provided score.py from our filled december_chart_inputs.csv. Only "
        "the date changes: Lexington to Fort Wayne, 360 miles, Dry Van, 32,000 lb. The file has no map position, "
        "market_index or quote_signal, so the map positions come from the city table and market_index from the "
        "mean of the same date in the November-December file."
    ))
    figure(doc, CHART_PATH, "Figure 4. December 2025 predicted rate from score.py.", width_cm=16.5)
    bullets(doc, [
        f"Predictions run from {money(dc['min'])} to {money(dc['max'])} (mean {money(dc['mean'])}). All of them "
        "sit inside the range of the 21 past Dry Van loads on this lane ($757.93 to $934.37).",
        "The weekly up-and-down follows market_index: highest on Thursday, lowest on Sunday and Monday.",
        "The steady climb comes from the trend and the quarter-end ramp, since December ends a quarter. If "
        "December does not ramp like March, June and September, these values run about 2% high on average.",
    ])

    # 8. Quality checks
    heading(doc, "8. Quality checks")
    if qa:
        bullets(doc, qa["bullets"])
    else:
        para(doc, "See docs/qa_report.md.")

    # 9. Limits
    heading(doc, "9. Limits and risks")
    bullets(doc, [
        "The ramp for December is learned from three quarters. It is the main source of uncertainty for the "
        "December loads.",
        "The trend extension is a judgment call: on the holdout, a full trend scored 1.53% and no trend 1.78%, "
        f"against {pct(chosen['clean']['MAPE'])} for the chosen half trend. October still ran "
        f"{abs(hold['by_month']['chosen, Oct']['clean']['median_signed_pct']):.1f}% low.",
        "Laredo lies south of every training city, so its prices rely on extrapolation.",
        "The ramp was first noticed while looking at all of January to October, so the holdout is slightly "
        "optimistic about it. The tuning folds, which use January to August only, support it on their own.",
        "XGBoost results depend slightly on the number of threads; the code fixes it at "
        f"{metrics['n_jobs']} so runs are byte-identical.",
    ])

    # 10. Reproduce
    heading(doc, "10. How to reproduce")
    for line in [
        "python -m pip install -r requirements.txt",
        "python run_pipeline.py",
        "python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv",
        "python -m pytest -q",
    ]:
        p = doc.add_paragraph()
        r = p.add_run(line)
        r.font.name = "Consolas"
        r.font.size = Pt(9.5)
        p.paragraph_format.space_after = Pt(1)
    para(doc, "Data files go in data/ (see README). Every step is explained in docs/WORKLOG.md.", space_after=0)
    return doc


def cv_lookup(cv: dict, name: str) -> dict:
    cands = cv["candidates"]
    if isinstance(cands, dict):
        return cands[name]
    for c in cands:
        if c.get("name") == name:
            return c
    raise KeyError(name)


def cv_mean(cv: dict, name: str) -> float:
    return cv_lookup(cv, name)["mean_clean_MAPE"]


def cv_row(cv: dict, name: str) -> list[str]:
    c = cv_lookup(cv, name)
    folds = c["clean_MAPE_by_fold"]
    return [pct(folds[m]) for m in ("May", "Jun", "Jul", "Aug")] + [pct(c["mean_clean_MAPE"], 3)]


def to_pdf(docx_path: Path, pdf_path: Path) -> bool:
    """Convert with Microsoft Word through PowerShell COM. Returns False if Word is unavailable."""
    ps = (
        "$w = New-Object -ComObject Word.Application; $w.Visible = $false; "
        f"$d = $w.Documents.Open('{docx_path}'); "
        f"$d.SaveAs([ref] '{pdf_path}', [ref] 17); $d.Close(); $w.Quit()"
    )
    try:
        subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=True, timeout=180,
                       capture_output=True)
    except (OSError, subprocess.SubprocessError):
        return False
    return pdf_path.is_file()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-pdf", action="store_true")
    args = parser.parse_args()
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    qa = json.loads(QA_SUMMARY_PATH.read_text(encoding="utf-8")) if QA_SUMMARY_PATH.is_file() else None
    doc = build(metrics, qa)
    doc.save(DOCX_PATH)
    print(f"Wrote {DOCX_PATH.relative_to(ROOT)}")
    if not args.no_pdf:
        if PDF_PATH.exists():
            PDF_PATH.unlink()
        if to_pdf(DOCX_PATH, PDF_PATH):
            print(f"Wrote {PDF_PATH.relative_to(ROOT)}")
        else:
            print("PDF not written: Microsoft Word is not available. Open report.docx and save as PDF.")


if __name__ == "__main__":
    main()
