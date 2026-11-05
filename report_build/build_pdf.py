"""Build the project report PDF from the executed notebook, the saved results and the figures."""
import re, json, sys, os
import numpy as np, pandas as pd
import matplotlib
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER, TA_LEFT
from reportlab.platypus import (BaseDocTemplate, PageTemplate, Frame, Paragraph, Spacer, Table, TableStyle, Image,
                                PageBreak, KeepTogether, NextPageTemplate, ListFlowable, ListItem, CondPageBreak)
from reportlab.platypus.tableofcontents import TableOfContents
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from PIL import Image as PI
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import simulation_core as sc
from narrative import narrative          # the interpretation text, written after reading the results

# ------------------------------------------------------------------ fonts and styles
fd = matplotlib.get_data_path() + "/fonts/ttf/"
for name, f in [("Serif", "DejaVuSerif.ttf"), ("Serif-Bold", "DejaVuSerif-Bold.ttf"), ("Serif-Italic", "DejaVuSerif-Italic.ttf"), ("Serif-BoldItalic", "DejaVuSerif-BoldItalic.ttf"),
                ("Sans", "DejaVuSans.ttf"), ("Sans-Bold", "DejaVuSans-Bold.ttf"), ("Sans-Oblique", "DejaVuSans-Oblique.ttf")]:
    pdfmetrics.registerFont(TTFont(name, fd + f))
pdfmetrics.registerFontFamily("Serif", normal="Serif", bold="Serif-Bold", italic="Serif-Italic", boldItalic="Serif-BoldItalic")
pdfmetrics.registerFontFamily("Sans", normal="Sans", bold="Sans-Bold", italic="Sans-Oblique", boldItalic="Sans-Bold")
NAVY, LIGHT, ZEBRA, BOX = colors.HexColor("#1f3a5f"), colors.HexColor("#dfe8f3"), colors.HexColor("#f4f7fb"), colors.HexColor("#eaf3ea")
W = A4[0] - 4.4 * cm
body = ParagraphStyle("body", fontName="Serif", fontSize=10, leading=15, alignment=TA_JUSTIFY, spaceAfter=7)
small = ParagraphStyle("small", parent=body, fontSize=8.5, leading=12, alignment=TA_LEFT)
cap = ParagraphStyle("cap", parent=body, fontName="Serif-Italic", fontSize=8.8, leading=12, alignment=TA_CENTER, spaceBefore=3, spaceAfter=12)
h1 = ParagraphStyle("h1", fontName="Sans-Bold", fontSize=16, leading=20, textColor=NAVY, spaceBefore=14, spaceAfter=8, keepWithNext=1)
h1x = ParagraphStyle("h1x", parent=h1)
h2 = ParagraphStyle("h2", fontName="Sans-Bold", fontSize=12, leading=15, textColor=NAVY, spaceBefore=10, spaceAfter=5, keepWithNext=1)
h2n = ParagraphStyle("h2n", parent=h2)
cell = ParagraphStyle("cell", fontName="Sans", fontSize=8.2, leading=10.5, alignment=TA_LEFT)
cellb = ParagraphStyle("cellb", parent=cell, fontName="Sans-Bold")
boxst = ParagraphStyle("box", parent=body, fontSize=9.5, leading=13.5, spaceAfter=0, alignment=TA_LEFT)
bullet = ParagraphStyle("bullet", parent=body, spaceAfter=3)
B1 = "β<sub>1</sub>"

# ------------------------------------------------------------------ data
R = f"{ROOT}/results"
mc = pd.read_csv(f"{R}/monte_carlo_results.csv")
est = pd.read_csv(f"{R}/summary_estimates.csv").set_index(["Mechanism", "Method"])
cov = pd.read_csv(f"{R}/summary_coverage.csv").set_index(["Mechanism", "Method"])
wid = pd.read_csv(f"{R}/summary_interval_width.csv").set_index(["Mechanism", "Method"])
hon = pd.read_csv(f"{R}/summary_se_honesty.csv").set_index(["Mechanism", "Method"])
walk = pd.read_csv(f"{R}/walkthrough_estimates.csv", index_col=0)
settings = json.load(open(f"{R}/monte_carlo_settings.json"))
N_REPS, B_BOOT, M_IMP = settings["N_REPS"], settings["B_BOOT"], settings["M_IMPUTATIONS"]
MC_ERR = 100 * 1.96 * np.sqrt(0.95 * 0.05 / N_REPS); MC_SE = 100 * np.sqrt(0.95 * 0.05 / N_REPS)
MECHS = ["MCAR", "MAR", "MNAR"]
METHODS = ["Complete Case", "Mean Imputation", "Hot-Deck", "Regression Imputation", "Stochastic Regression", "MICE (normal)", "MICE (PMM)", "EM"]
ROWS = ["Full data"] + METHODS
KINDS = ["Classical", "Bootstrap standard", "Bootstrap percentile", "Rubin"]
def E(me, m, c): return float(est.loc[(me, m), c])
def C(me, m, k): return float(cov.loc[(me, m), k])
def best(me, m): return C(me, m, "Rubin" if m.startswith("MICE") else "Bootstrap percentile")
nb = json.load(open(f"{ROOT}/notebook/missing_data_simulation.ipynb"))
def out_text(prefix):
    for c in nb["cells"]:
        if c["cell_type"] == "code" and "".join(c["source"]).startswith(prefix):
            return "\n".join("".join(o["text"]) if o.get("output_type") == "stream" else "".join(o.get("data", {}).get("text/plain", "")) for o in c["outputs"])
    raise KeyError(prefix)
full = sc.make_data(np.random.default_rng(sc.SEED)); bfull, sefull = sc.ols(full); FULL_B1 = bfull[1]
counts_txt = out_text("# ===== Step 4h")
def count_row(label): return [int(x) for x in re.search(rf"^{re.escape(label)}\s+(\d+)\s+(\d+)\s+(\d+)\s*$", counts_txt, re.M).groups()]
cnt_x1, cnt_y, cnt_any = count_row("X1"), count_row("Y"), count_row("rows with any gap")
chk_txt = out_text("# ===== Step 4j")
chk = {m: [float(x) for x in re.search(rf"^{m}\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s*$", chk_txt, re.M).groups()] for m in MECHS}
mice = {}
for mech, pre in zip(MECHS, ["# ===== Step 5.8e", "# ===== Step 5.8f", "# ===== Step 5.8g"]):
    t = out_text(pre); betas = [float(x) for x in re.findall(r"imputation \d+\s+([\d.]+)\s+[\d.]+\s*$", t, re.M)]
    w, b = re.search(r"within-variance = ([\d.]+)\s+between-variance = ([\d.]+)", t).groups(); mice[mech] = (betas, float(w), float(b))
boot_demo = json.load(open(f"{HERE}/_boot_demo.json")); slopes = json.load(open(f"{HERE}/_scatter_slopes.json"))
txt6c = out_text("# ===== Step 6c")
D = dict(M_IMP_TXT=str(M_IMP), E=E, C=C, best=best, MC_ERR=MC_ERR, MC_SE=MC_SE, N_REPS=N_REPS, B_BOOT=B_BOOT, METHODS=METHODS, MECHS=MECHS, hon=hon, wid=wid, FULL_B1=FULL_B1, walk=walk, slopes=slopes, boot_demo=boot_demo)
NARR = narrative(D)

# ------------------------------------------------------------------ helpers
story = []
def P(text, st=body): story.append(Paragraph(text, st))
def H1(text): story.append(CondPageBreak(4.5 * cm)); story.append(Paragraph(text, h1))
def H2(text): story.append(CondPageBreak(3.5 * cm)); story.append(Paragraph(text, h2))
def H2n(text): story.append(CondPageBreak(3.5 * cm)); story.append(Paragraph(text, h2n))
def bullets(items): story.append(ListFlowable([ListItem(Paragraph(i, bullet), leftIndent=14) for i in items], bulletType="bullet", start="•", leftIndent=14, bulletFontSize=9))
def simple_box(text, title="In simple words"):
    t = Table([[Paragraph(f"<b>{title}.</b> {text}", boxst)]], colWidths=[W])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), BOX), ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#8fbf8f")), ("LEFTPADDING", (0, 0), (-1, -1), 9),
                           ("RIGHTPADDING", (0, 0), (-1, -1), 9), ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 7)]))
    story.extend([t, Spacer(1, 9)])
def figure(path, caption, width=W):
    w, h = PI.open(path).size
    story.append(KeepTogether([Image(path, width=width, height=width * h / w), Paragraph(caption, cap)]))
def table(rows, widths, caption=None, text_cols=(), align_right_from=1, header_rows=1):
    data = [[Paragraph(str(c), cellb if ri < header_rows else cell) if (ri < header_rows or ci in text_cols) else str(c) for ci, c in enumerate(r)] for ri, r in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=header_rows)
    st = [("BACKGROUND", (0, 0), (-1, header_rows - 1), LIGHT), ("FONTNAME", (0, header_rows), (-1, -1), "Sans"), ("FONTSIZE", (0, header_rows), (-1, -1), 8.2), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
          ("LINEBELOW", (0, 0), (-1, 0), 0.8, NAVY), ("LINEBELOW", (0, -1), (-1, -1), 0.8, NAVY), ("TOPPADDING", (0, 0), (-1, -1), 3.2), ("BOTTOMPADDING", (0, 0), (-1, -1), 3.2),
          ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4)]
    for ri in range(header_rows, len(rows)):
        if (ri - header_rows) % 2 == 1: st.append(("BACKGROUND", (0, ri), (-1, ri), ZEBRA))
    for ci in range(len(rows[0])):
        if ci >= align_right_from and ci not in text_cols: st.append(("ALIGN", (ci, header_rows), (ci, -1), "RIGHT"))
    t.setStyle(TableStyle(st)); story.append(KeepTogether([t] + ([Paragraph(caption, cap)] if caption else [Spacer(1, 8)])))
def f(x, d=3, sign=False):
    if x is None or (isinstance(x, float) and np.isnan(x)): return "–"
    return f"{x:+.{d}f}" if sign else f"{x:.{d}f}"

class Doc(BaseDocTemplate):
    def afterFlowable(self, fl):
        cands = [fl] + (list(fl._content[:1]) if isinstance(fl, KeepTogether) else [])
        for c in cands:
            if isinstance(c, Paragraph) and c.style.name in ("h1", "h2"):
                self.notify("TOCEntry", (0 if c.style.name == "h1" else 1, c.getPlainText(), self.page - 1))
def header_footer(canv, doc):
    canv.saveState(); canv.setFont("Sans", 8); canv.setFillColor(colors.HexColor("#555555"))
    canv.drawString(2.2 * cm, A4[1] - 1.5 * cm, "Missing Data & Statistical Inference"); canv.drawRightString(A4[0] - 2.2 * cm, A4[1] - 1.5 * cm, "Project Report")
    canv.setStrokeColor(colors.HexColor("#999999")); canv.setLineWidth(0.5); canv.line(2.2 * cm, A4[1] - 1.7 * cm, A4[0] - 2.2 * cm, A4[1] - 1.7 * cm)
    canv.drawCentredString(A4[0] / 2, 1.3 * cm, str(doc.page - 1)); canv.restoreState()
doc = Doc(f"{ROOT}/report/Missing_Data_Statistical_Inference_Report.pdf", pagesize=A4, leftMargin=2.2 * cm, rightMargin=2.2 * cm, topMargin=2.4 * cm, bottomMargin=2.0 * cm,
          title="Missing Data and Statistical Inference", author="")
frame = Frame(doc.leftMargin, doc.bottomMargin, W, A4[1] - doc.topMargin - doc.bottomMargin, id="f")
doc.addPageTemplates([PageTemplate("cover", [frame], onPage=lambda c, d: None), PageTemplate("main", [frame], onPage=header_footer)])

# ================================================================== COVER
story.append(Spacer(1, 5.5 * cm))
story.append(Paragraph("MISSING DATA &amp;<br/>STATISTICAL INFERENCE", ParagraphStyle("t", fontName="Sans-Bold", fontSize=30, leading=38, textColor=NAVY, alignment=TA_CENTER)))
story.append(Spacer(1, 0.8 * cm))
story.append(Paragraph("A Monte Carlo Study of MCAR, MAR and MNAR", ParagraphStyle("t2", fontName="Sans", fontSize=15, leading=20, alignment=TA_CENTER, textColor=colors.HexColor("#333333"))))
story.append(Spacer(1, 0.4 * cm))
story.append(Paragraph("Eight Ways of Handling Missing Values, and the Coverage of<br/>Classical, Bootstrap and Rubin Confidence Intervals", ParagraphStyle("t3", fontName="Sans", fontSize=12, leading=17, alignment=TA_CENTER, textColor=colors.HexColor("#555555"))))
story.append(Spacer(1, 4.2 * cm))
story.append(Paragraph("Project Report<br/>M.Sc. Statistics<br/>November 2026", ParagraphStyle("t4", fontName="Sans", fontSize=12, leading=19, alignment=TA_CENTER, textColor=colors.HexColor("#333333"))))
story.append(NextPageTemplate("main")); story.append(PageBreak())

# ================================================================== SUMMARY + TOC
story.append(Paragraph("Summary", h1x))
P("Real datasets very often have missing values. This project asks: <b>if some values are missing, does the way we deal with them change our results, and can we still trust the confidence interval that comes with them?</b>")
P(f"We created data from a model we know, <i>Y</i> = 5 + 2<i>X</i><sub>1</sub> − 1.5<i>X</i><sub>2</sub> + <i>X</i><sub>3</sub> + error, with 200 observations, so the true slope is {B1} = 2. About 20% of the values of <i>Y</i> and of <i>X</i><sub>1</sub> were deleted in three ways: completely at random (MCAR), depending on another observed variable (MAR), and depending on the value that disappears (MNAR). Eight methods were applied to every incomplete dataset: complete-case analysis, mean imputation, hot-deck, regression imputation, stochastic regression, two versions of multiple imputation (MICE) and the EM algorithm.")
P(f"For every method we built the confidence intervals an analyst would actually use: the <b>classical</b> interval (from the usual formula), two <b>bootstrap</b> intervals (standard and percentile) and, for MICE, the <b>Rubin</b> interval. To find out whether these intervals can be trusted we ran a <b>Monte Carlo study</b>: {N_REPS} new datasets were drawn from the true model, and inside each one every method was bootstrapped {B_BOOT} times (N × B replication). The <b>coverage</b> of an interval is the share of the {N_REPS} intervals that contain the true value; it should be about 95%. All comparisons are made <b>within</b> each mechanism.")
P("<b>Main findings</b>", ParagraphStyle("mf", parent=body, spaceAfter=3))
bullets(NARR["summary"])
story.append(PageBreak())
story.append(Paragraph("Contents", h1x))
toc = TableOfContents()
toc.levelStyles = [ParagraphStyle("t0", fontName="Sans-Bold", fontSize=9.5, leading=12.5, spaceBefore=2.5), ParagraphStyle("t1", fontName="Sans", fontSize=8.6, leading=11, leftIndent=18)]
story.append(toc); story.append(PageBreak())

# ================================================================== 1-2
H1("1  Introduction")
P("Missing data are very common: a survey respondent skips a question, a machine fails to record a reading, a patient leaves a study early. When values are missing, the analysis changes for two reasons. There is less information, and the way we <i>handle</i> the missing values can change the relationships we find and how sure we are about them.")
P("How serious the problem is depends on <i>why</i> the values are missing. Statisticians describe three situations:")
bullets(["<b>MCAR (Missing Completely At Random).</b> Whether a value is missing has nothing to do with the data. Example: a sensor fails at random moments.",
         "<b>MAR (Missing At Random).</b> Whether a value is missing depends on something we <i>did</i> observe. Example: older people skip a question more often, and age is recorded for everyone.",
         "<b>MNAR (Missing Not At Random).</b> Whether a value is missing depends on the missing value itself. Example: people with very high incomes refuse to report their income."])
P("These form a ladder: MCAR ⊂ MAR ⊂ MNAR. There are many ways to deal with missing values, and they can lead to different estimates <i>and</i> to different confidence intervals. This project compares them in a controlled experiment where the true answer is known, and checks not only the estimates but also whether the confidence intervals really contain the truth as often as they claim.")
H1("2  Aim and Research Question")
P(f"The aim is to find out how different missing-data methods affect a regression estimate and its uncertainty, when every method is given the same incomplete data from the same missingness mechanism. We focus on {B1}, the slope of <i>X</i><sub>1</sub>, whose true value is 2.")
simple_box("<i>Within each type of missingness, which way of handling the missing values gives the most accurate estimate, and which confidence intervals (classical, bootstrap or Rubin) contain the true value about 95% of the time, as a 95% interval should?</i>", "The question")
P("The question is answered three times: once for MCAR, once for MAR and once for MNAR. The three mechanisms are not ranked against each other; each is a separate setting in which the methods are compared.")

# ================================================================== 3-4 data and deletion
H1("3  How the Data Were Created")
P("Each complete dataset follows the linear model")
P("<i>Y</i> = 5 + 2<i>X</i><sub>1</sub> − 1.5<i>X</i><sub>2</sub> + 1<i>X</i><sub>3</sub> + ε,", ParagraphStyle("eq", parent=body, alignment=TA_CENTER))
P("where <i>X</i><sub>1</sub>, <i>X</i><sub>2</sub>, <i>X</i><sub>3</sub> and the error ε are independent standard normal variables, with <b>n = 200</b> observations. We also create an <b>auxiliary variable</b> <i>Z</i> = <i>Y</i> + noise (correlation with <i>Y</i> about 0.94). <i>Z</i> is not in the regression and is never missing, but it can help to fill in the gaps, like a cheap rough measurement next to an exact one.")
table([["Parameter", "True value", "Estimate on the walkthrough dataset (nothing missing)", "Standard error"]] +
      [[n, f"{t:.3f}", f"{e:.4f}", f"{se:.4f}"] for n, t, e, se in zip(["β0 (intercept)", "β1", "β2", "β3"], [5, 2, -1.5, 1], bfull, sefull)],
      [3.4 * cm, 2.6 * cm, 7.2 * cm, 3.4 * cm], "Table 1: The regression on one complete dataset (seed 42), before any value is deleted.")
P(f"The estimate of {B1} is {FULL_B1:.3f}, not exactly 2: a dataset is a sample, so even with nothing missing the estimate has random error. This is why one dataset can never tell us whether a method works.")
H1("4  How Values Were Deleted")
P("About <b>20% of the values of <i>Y</i> and of <i>X</i><sub>1</sub></b> were deleted, each separately, so roughly a third of the rows have at least one gap. The mechanisms differ only in <i>how likely</i> a value is to be deleted (Figure 1):")
bullets(["<b>MCAR:</b> every value has the same 20% chance.", "<b>MAR:</b> the chance depends on <i>Z</i>, which is always observed; rows with a large <i>Z</i> lose values more often.",
         "<b>MNAR:</b> the chance depends on the value being deleted; large <i>Y</i> and large <i>X</i><sub>1</sub> vanish more often."])
P("For MAR and MNAR the chance is expit(−2.23 + 2 × <i>d</i>), where <i>d</i> is the standardised driver and expit(<i>z</i>) = 1/(1 + e<super>−<i>z</i></super>) turns any number into a probability. The −2.23 keeps the overall rate near 20% in every mechanism.")
figure(f"{ROOT}/figures/fig1_design.png", "Figure 1: (a) The complete data. (b) The chance that a value is deleted under each mechanism.")
table([["Mechanism", "X1 missing", "Y missing", "Rows with at least one gap", "Complete rows left"]] + [[m, cnt_x1[i], cnt_y[i], f"{cnt_any[i]} ({cnt_any[i] / 2:.0f}%)", 200 - cnt_any[i]] for i, m in enumerate(MECHS)],
      [3 * cm, 2.8 * cm, 2.8 * cm, 4.6 * cm, 3.4 * cm], "Table 2: Missing values in the walkthrough dataset (out of 200 rows).")
table([["Mechanism", "Y missing (%)", "X1 missing (%)", "Rows with a gap (%)", "Z: missing − observed", "Y: missing − observed"]] + [[m] + [f"{x:.2f}" for x in chk[m]] for m in MECHS],
      [2.4 * cm, 2.6 * cm, 2.6 * cm, 3 * cm, 3 * cm, 3 * cm], "Table 3: Check of the deletion rules on 20,000 rows. Under MCAR the rows that lose Y look like the others; under MAR and MNAR they have clearly higher Z and Y.")
simple_box("In real data we can see <i>Z</i> but not the deleted <i>Y</i> values, so MAR and MNAR cannot be told apart from the observed data alone. We can tell them apart here only because we deleted the values ourselves.")

# ================================================================== 5 methods
H1("5  The Eight Methods")
P("Every method ends with the same regression of <i>Y</i> on <i>X</i><sub>1</sub>, <i>X</i><sub>2</sub>, <i>X</i><sub>3</sub>. They differ only in what they do with the gaps first.")
table([["Method", "How it works", "Good", "Not so good"],
       ["Complete case", "Delete every row with a gap.", "Simple. Fine under MCAR.", "Throws away rows; can be biased otherwise."],
       ["Mean imputation", "Fill each gap with the column average.", "Very simple.", "Shrinks the spread and weakens relationships."],
       ["Hot-deck", "Copy the value from the most similar row.", "Uses real values.", "One guess per gap."],
       ["Regression imputation", "Fill each gap with its regression prediction.", "Uses the other variables.", "Filled values sit exactly on the line."],
       ["Stochastic regression", "Prediction plus a random residual.", "Restores natural scatter.", "Still one guess per gap."],
       ["MICE (normal)", f"Make {M_IMP} random completed datasets, analyse each, combine.", "Honest uncertainty.", "More work."],
       ["MICE (PMM)", "Same, but borrow real values from similar rows.", "Real values; honest uncertainty.", "Needs enough similar donors."],
       ["EM", "Maximum likelihood using every row, even partly observed ones.", "Uses all information.", "Assumes normality."]],
      [3.1 * cm, 5.6 * cm, 3.6 * cm, 4.3 * cm], "Table 4: The eight methods at a glance.", text_cols=(0, 1, 2, 3), align_right_from=9)
P("<b>Regression imputation, stochastic regression and both MICE versions</b> share one engine: each incomplete variable is regressed on all the others (<i>Z</i> included), and the variables are filled in one at a time, cycling through them five times. Regression imputation uses the prediction; stochastic regression adds a random residual; MICE first draws new regression coefficients (so the uncertainty of the model itself is included), fills the gaps either from the regression model (normal) or by borrowing the real value of one of the five rows with the closest prediction (PMM), and repeats this "
  f"{M_IMP} times. <b>EM</b> estimates the means and covariances of all five variables by maximum likelihood, alternating an expectation step (fill each gap with its expected value) and a maximization step (re-estimate), and computes the regression coefficients from them.")
figure(f"{ROOT}/figures/fig2_what_methods_do_MAR.png", "Figure 2: Y against X1 in the MAR walkthrough dataset after each method has handled the gaps. Orange = rows where a value was missing; the black line is the fitted slope of Y on X1.")
P(f"Figure 2 shows what the methods do. Complete case loses the top-right part of the cloud (under MAR, rows with a large <i>Z</i> lose values), so its slope flattens to {slopes['Complete Case']:.2f}. Mean imputation puts the filled points on a cross through the average, flattening the slope to {slopes['Mean Imputation']:.2f}. Regression imputation places them exactly on the line. The other methods scatter them around the line like real data. (These slopes are of <i>Y</i> on <i>X</i><sub>1</sub> alone; for the full data it is {slopes['Full data (truth)']:.2f}.)")

# ================================================================== 6 intervals and how they are checked
H1("6  Confidence Intervals and How They Are Checked")
H2("6.1  Four kinds of interval")
table([["Interval", "How it is built", "Used for"],
       ["Classical", "estimate ± 1.96 × the usual regression formula SE", "complete case, mean, hot-deck, regression, stochastic"],
       ["Bootstrap standard", "estimate ± 1.96 × the bootstrap SE", "complete case, mean, hot-deck, regression, stochastic, EM"],
       ["Bootstrap percentile", "the 2.5% and 97.5% points of the bootstrap estimates", "the same six methods"],
       ["Rubin", "pooled estimate ± 1.96 × the Rubin’s-rules SE", "MICE (normal) and MICE (PMM)"]],
      [3.4 * cm, 6.6 * cm, 6.6 * cm], "Table 5: The confidence intervals compared.", text_cols=(0, 1, 2), align_right_from=9)
P("For the imputation methods the classical interval is <i>naive</i>: it treats the filled-in values as if they were real observations. EM has no simple standard-error formula, so it gets only the bootstrap intervals. MICE needs no bootstrap, because Rubin’s rules (total variance = within + (1 + 1/<i>m</i>) × between) already include the extra uncertainty caused by the missing data.")
H2("6.2  The bootstrap")
P(f"The bootstrap does not know the true population, so it treats the sample as if it were the population: draw 200 rows <i>with replacement</i>, <b>redo the whole method</b> (including filling in the gaps), and record {B1}. Repeating this {B_BOOT} times gives {B_BOOT} estimates. Their standard deviation is the <b>bootstrap SE</b> and their middle 95% is the <b>percentile interval</b>. Of the four common bootstrap intervals (standard, percentile, bootstrap-t and BCa), only the standard and bootstrap-t intervals use the bootstrap SE; we use the standard and percentile intervals.")
figure(f"{ROOT}/figures/fig3_one_bootstrap.png", f"Figure 3: One bootstrap: {B_BOOT} estimates of β1 for stochastic regression on the MAR walkthrough dataset.")
bd = boot_demo
P(f"Figure 3 shows one bootstrap. The classical interval of stochastic regression on this dataset is ({bd['classical'][0]:.3f}, {bd['classical'][1]:.3f}); the percentile interval is ({bd['pct'][0]:.3f}, {bd['pct'][1]:.3f}). Both happen to contain 2 here — but that is one dataset.")
H2("6.3  Why Monte Carlo is needed: bootstrap and Monte Carlo are different")
bullets(["The <b>bootstrap</b> resamples the rows of <i>one</i> dataset and produces <i>one</i> confidence interval. It cannot say how often such intervals contain the truth.",
         "<b>Monte Carlo</b> draws <i>many new datasets from the true population</i>. Building the interval on each and counting how many contain the true value gives the <b>empirical coverage probability</b>. Without Monte Carlo the coverage of an interval cannot be found."])
P(f"So the bootstrap runs <b>inside</b> the Monte Carlo. For each of {N_REPS} repetitions: (1) generate a new complete dataset from the true model; (2) for each mechanism, delete values with the same deletion function; (3) apply all eight methods, and for each build its classical interval, bootstrap it {B_BOOT} times for the two bootstrap intervals, and build the Rubin interval for MICE; (4) also analyse the complete data as a benchmark. This is <b>N × B replication</b>: about one million imputations and regressions in total.")
H2("6.4  What is measured")
table([["Measure", "What it tells us"],
       ["Bias", f"Average of (estimate − 2) over the {N_REPS} datasets. Zero means right on average."],
       ["Spread (SD)", "How much the estimate varies from one dataset to the next: the true uncertainty."],
       ["RMSE", "Overall accuracy, combining bias and spread. Smaller is better."],
       ["Coverage (%)", f"Share of the {N_REPS} intervals that contain 2. It should be about 95."],
       ["Width", "Average length of the interval. Narrower is better, but only if coverage is right."],
       ["SE ÷ SD", "Reported standard error divided by the true spread. 1 = honest; below 1 = over-confident."]],
      [3.2 * cm, 13.4 * cm], "Table 6: Measures used in the Monte Carlo study.", text_cols=(0, 1), align_right_from=9)
P(f"With {N_REPS} repetitions, a coverage near 95% has a standard error of about {MC_SE:.1f} percentage points, so the 95% band of random error is about ±{MC_ERR:.1f} points. Differences smaller than that should not be over-read.")

# ================================================================== 7 walkthrough results (brief)
H1("7  One Dataset: An Illustration")
P(f"Before the Monte Carlo study, every method was applied to one dataset (seed 42) under each mechanism, step by step, to see what it does. Table 7 shows the estimates. The complete-data estimate is {FULL_B1:.3f}.")
table([["Method"] + MECHS] + [[m] + [f"{walk.loc[m, me]:.3f}" for me in MECHS] for m in METHODS], [5.2 * cm, 3.6 * cm, 3.6 * cm, 3.6 * cm],
      "Table 7: The β1 estimate of every method on the one walkthrough dataset.")
P(NARR["walk"])

# ================================================================== 8 Monte Carlo results
H1("8  Results of the Monte Carlo Study")
P(f"For each mechanism we show a table of the estimates (ranked by RMSE), a table of the coverage of every interval, and a figure with three panels: (a) the spread of the {N_REPS} estimates, (b) the coverage of each interval type, and (c) the reported SE divided by the true spread. Colours in panel (a): grey = complete case, orange = single imputation, blue = MICE and EM; light grey = the benchmark with no missing data.")
def est_table(me, num):
    t = est.loc[me].loc[ROWS]; rest = t.drop("Full data").sort_values("RMSE")
    rows = [["Rank", "Method", "Bias", "Spread (SD)", "RMSE"], ["–", "Full data (benchmark)", f(t.loc["Full data", "Bias"], sign=True), f(t.loc["Full data", "SD"]), f(t.loc["Full data", "RMSE"])]]
    for i, (m, r) in enumerate(rest.iterrows(), 1): rows.append([i, m, f(r.Bias, sign=True), f(r.SD), f(r.RMSE)])
    table(rows, [1.4 * cm, 6 * cm, 3 * cm, 3 * cm, 3.2 * cm], f"Table {num}: {me}, the estimates ({N_REPS} repetitions), ranked by RMSE.", align_right_from=2)
def cov_table(me, num):
    rows = [["Method", "Classical", "Bootstrap standard", "Bootstrap percentile", "Rubin", "Width of best-suited interval"]]
    for m in ROWS:
        k = "Rubin" if m.startswith("MICE") else ("Classical" if m == "Full data" else "Bootstrap percentile")
        rows.append([m if m != "Full data" else "Full data (benchmark)"] + [f(C(me, m, kk), 1) for kk in KINDS] + [f(float(wid.loc[(me, m), k]))])
    table(rows, [4.6 * cm, 2.2 * cm, 2.4 * cm, 2.5 * cm, 1.8 * cm, 3.1 * cm], f"Table {num}: {me}, coverage (%) of every 95% interval (target 95; random error about ±{MC_ERR:.1f}). Width: percentile interval (Rubin for MICE, classical for the benchmark).", align_right_from=1)
nums = iter(range(8, 20)); fignums = iter([4, 5, 6])
for me, title in [("MCAR", "8.1  MCAR: the gaps are pure luck"), ("MAR", "8.2  MAR: the gaps depend on an observed variable"), ("MNAR", "8.3  MNAR: the gaps depend on the missing value itself")]:
    H2(title); est_table(me, next(nums)); cov_table(me, next(nums))
    fn = next(fignums)
    figure(f"{ROOT}/figures/fig{fn}_simulation_{me}.png", f"Figure {fn}: {me}. (a) Spread of the estimates; (b) coverage of each interval type; (c) reported SE ÷ true spread.")
    bullets(NARR[me])
H2("8.4  Classical versus bootstrap intervals, one by one")
figure(f"{ROOT}/figures/fig7_intervals_MAR.png", "Figure 7: MAR. The first 40 intervals of complete case and stochastic regression: classical (left) and bootstrap percentile (right). Green = contains the truth, red = misses.")
P(NARR["fig7"])
H2("8.5  Before and after the bootstrap")
figure(f"{ROOT}/figures/fig8_coverage_before_after.png", "Figure 8: Coverage of every method under every mechanism, without the bootstrap (left) and with the bootstrap percentile interval (right). MICE uses the Rubin interval in both.")
P(NARR["fig8"])
wins = []
for me in MECHS:
    t = est.loc[me].drop("Full data"); bc = pd.Series({m: best(me, m) for m in METHODS}); honest = bc[bc >= 90].index
    wins.append([me, t.RMSE.idxmin(), t.Bias.abs().idxmin(), (bc - 95).abs().idxmin(), t.loc[honest, "RMSE"].idxmin() if len(honest) else "none"])
table([["Mechanism", "Lowest RMSE", "Smallest bias", "Coverage closest to 95%*", "Lowest RMSE among methods with coverage ≥ 90%*"]] + wins,
      [2.4 * cm, 3.3 * cm, 3.3 * cm, 3.5 * cm, 4.1 * cm], "Table 14: The best method under each mechanism. *Coverage of the best-suited interval: Rubin for MICE, bootstrap percentile otherwise.", text_cols=(0, 1, 2, 3, 4), align_right_from=9)
P("Where the leading methods are within random error of each other, the “best” in this table should not be read as a real difference.", small)

# ================================================================== 9-11
story.append(CondPageBreak(8 * cm)); H1("9  Discussion")
for head, text in NARR["discussion"]:
    H2(head); P(text)
    if head.startswith("9.3"):
        table([["Mechanism", "Method", "Bias ÷ SD", "Bootstrap SE ÷ SD", "Predicted coverage", "Observed coverage"]] +
              [[me, m, f"{k:+.2f}", f"{r:.2f}", f"{pp:.1f}%", f"{o:.1f}%"] for me, m, k, r, pp, o in D["pred_table"]],
              [2.3 * cm, 4.4 * cm, 2.2 * cm, 2.9 * cm, 2.4 * cm, 2.4 * cm], "Table 16: Coverage of the bootstrap-standard interval predicted from bias and SD alone, and the coverage observed in the Monte Carlo study.", align_right_from=2)
H1("10  Limitations")
bullets(NARR["limitations"])
H1("11  Conclusion")
P("How missing data are handled is part of the statistical analysis, not a detail to be skipped. It changes the estimate and it changes whether the confidence interval can be trusted.")
table([["Mechanism", "What works", "What to avoid"]] + NARR["recommend"], [2.6 * cm, 7.4 * cm, 6.6 * cm], "Table 15: Recommendations by mechanism.", text_cols=(0, 1, 2), align_right_from=9)
bullets(NARR["conclusion"])

# ================================================================== APPENDICES
story.append(PageBreak())
H1("Appendix A: A Worked MICE Example")
P(f"The numbers behind MICE (normal) on the three walkthrough datasets: the {M_IMP} estimates of β<sub>1</sub> and Rubin’s rules.")
table([["Mechanism"] + [f"m{i}" for i in range(1, M_IMP + 1)]] + [[m] + [f"{x:.3f}" for x in mice[m][0]] for m in MECHS], [2.4 * cm] + [1.42 * cm] * M_IMP, "Table A1: The estimates of β1 from the completed datasets.")
rows = [["Mechanism", "Pooled β1", "Within (W)", "Between (B)", f"Total = W + {1 + 1 / M_IMP:.1f} B", "Pooled SE"]]
for m in MECHS:
    betas, w, b = mice[m]; tot = w + (1 + 1 / M_IMP) * b; rows.append([m, f"{np.mean(betas):.3f}", f"{w:.5f}", f"{b:.5f}", f"{tot:.5f}", f"{np.sqrt(tot):.3f}"])
table(rows, [2.4 * cm, 2.4 * cm, 2.8 * cm, 2.8 * cm, 3.4 * cm, 2.4 * cm], "Table A2: Rubin’s rules.")
H1("Appendix B: Settings and How to Reproduce")
table([["Setting", "Value"], ["Observations per dataset", "200"], ["True model", "Y = 5 + 2·X1 − 1.5·X2 + 1·X3 + ε,  Z = Y + noise"], ["Missing values", "about 20% of Y and about 20% of X1"],
       ["Walkthrough", "one dataset, seed 42"], ["Monte Carlo", f"{N_REPS} repetitions, seed {settings['MC_SEED']} (one independent seed per repetition)"],
       ["Bootstrap", f"{B_BOOT} resamples inside every dataset, for complete case, mean, hot-deck, regression, stochastic and EM"],
       ["MICE", f"{M_IMP} imputations, 5 cycles, PMM with 5 donors"], ["Software", "Python: numpy, pandas, scipy, scikit-learn, statsmodels, matplotlib, joblib (parallel computing)"],
       ["Run time", "the Monte Carlo study takes about 40 minutes on one core and is spread over all cores; results are saved and reloaded"]],
      [5 * cm, 11.6 * cm], None, text_cols=(0, 1), align_right_from=9)
H1("References")
for r in ["Dempster, A. P., Laird, N. M., and Rubin, D. B. (1977). Maximum likelihood from incomplete data via the EM algorithm. <i>Journal of the Royal Statistical Society: Series B</i>, 39(1), 1–38.",
          "Efron, B., and Tibshirani, R. J. (1993). <i>An Introduction to the Bootstrap</i>. Chapman &amp; Hall/CRC.",
          "Rubin, D. B. (1987). <i>Multiple Imputation for Nonresponse in Surveys</i>. Wiley.",
          "van Buuren, S. (2018). <i>Flexible Imputation of Missing Data</i>, 2nd ed. Chapman &amp; Hall/CRC.",
          "Koley, D. Missing Data – Unit I: Rigorous Treatment. Course material, Department of Statistics, St. Xavier’s University, Kolkata.",
          "Koley, D. Missing Data – Unit II: Imputation Methods and Algorithmic Approaches. Course material, Department of Statistics, St. Xavier’s University, Kolkata."]:
    P(r, ParagraphStyle("ref", parent=body, leftIndent=14, firstLineIndent=-14, alignment=TA_LEFT))

def attach_headings(st):
    out, i = [], 0
    while i < len(st):
        fl = st[i]
        if isinstance(fl, Paragraph) and fl.style.name in ("h1", "h2") and i + 1 < len(st) and not isinstance(st[i + 1], (Paragraph, CondPageBreak, PageBreak)):
            nxt = st[i + 1]; out.append(KeepTogether([fl] + (list(nxt._content) if isinstance(nxt, KeepTogether) else [nxt]))); i += 2
        else: out.append(fl); i += 1
    return out
story[:] = attach_headings(story)
doc.multiBuild(story)
print("PDF built")
