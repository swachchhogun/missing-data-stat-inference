import re, json, sys
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
import os
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from simulation_core import make_data, ols

# ------------------------------------------------------------------ fonts (DejaVu has Greek letters)
fd = matplotlib.get_data_path() + "/fonts/ttf/"
for name, f in [("Serif", "DejaVuSerif.ttf"), ("Serif-Bold", "DejaVuSerif-Bold.ttf"), ("Serif-Italic", "DejaVuSerif-Italic.ttf"),
                ("Serif-BoldItalic", "DejaVuSerif-BoldItalic.ttf"), ("Sans", "DejaVuSans.ttf"), ("Sans-Bold", "DejaVuSans-Bold.ttf"),
                ("Sans-Oblique", "DejaVuSans-Oblique.ttf")]:
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
cell = ParagraphStyle("cell", fontName="Sans", fontSize=8.2, leading=10.5, alignment=TA_LEFT)
cellb = ParagraphStyle("cellb", parent=cell, fontName="Sans-Bold")
boxst = ParagraphStyle("box", parent=body, fontSize=9.5, leading=13.5, spaceAfter=0, alignment=TA_LEFT)
bullet = ParagraphStyle("bullet", parent=body, spaceAfter=3)
h2n = ParagraphStyle("h2n", parent=h2)          # same look as h2, but not listed in the contents
B1 = "β<sub>1</sub>"

# ------------------------------------------------------------------ data
s = pd.read_csv(f"{ROOT}/results/simulation_summary.csv"); a = pd.read_csv(f"{ROOT}/results/part_a_results.csv")
MECHS = ["MCAR", "MAR", "MNAR"]
METHODS = ["Complete Case", "Mean Imputation", "Hot-Deck", "Regression Imputation", "Stochastic Regression", "MICE (normal)", "MICE (PMM)", "EM"]
def S(mech, method, col): return float(s[(s.Mechanism == mech) & (s.Method == method)][col].iloc[0])
def A(mech, method, col): return float(a[(a.Mechanism == mech) & (a.Method == method)][col].iloc[0])
nb = json.load(open(f"{ROOT}/notebook/missing_data_simulation.ipynb"))
def out_text(prefix):
    for c in nb["cells"]:
        if c["cell_type"] == "code" and "".join(c["source"]).startswith(prefix):
            t = []
            for o in c["outputs"]:
                if o.get("output_type") == "stream": t.append("".join(o["text"]))
                elif "data" in o and "text/plain" in o["data"]: t.append("".join(o["data"]["text/plain"]))
            return "\n".join(t)
    raise KeyError(prefix)

full = make_data(np.random.default_rng(42)); bfull, sefull = ols(full)
FULL_B1 = bfull[1]
counts_txt = out_text("# ===== Step 4h")
def count_row(label):
    m = re.search(rf"^{re.escape(label)}\s+(\d+)\s+(\d+)\s+(\d+)\s*$", counts_txt, re.M); return [int(x) for x in m.groups()]
cnt_x1, cnt_y, cnt_any = count_row("X1"), count_row("Y"), count_row("rows with any gap")
chk_txt = out_text("# ===== Step 4j")
chk = {m: [float(x) for x in re.search(rf"^{m}\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s*$", chk_txt, re.M).groups()] for m in MECHS}
mice = {}
for mech, pre in zip(MECHS, ["# ===== Step 5.8e", "# ===== Step 5.8f", "# ===== Step 5.8g"]):
    t = out_text(pre)
    betas = [float(x) for x in re.findall(r"imputation \d+\s+([\d.]+)\s+[\d.]+\s*$", t, re.M)]
    w, b = re.search(r"within-variance = ([\d.]+)\s+between-variance = ([\d.]+)", t).groups()
    mice[mech] = (betas, float(w), float(b))
assert all(len(v[0]) == 10 for v in mice.values())

# ------------------------------------------------------------------ helpers
story = []
def P(text, st=body): story.append(Paragraph(text, st))
def H1(text):
    story.append(CondPageBreak(4.5 * cm)); story.append(Paragraph(text, h1))
def H2(text):
    story.append(CondPageBreak(3.5 * cm)); story.append(Paragraph(text, h2))
def H2n(text):
    story.append(CondPageBreak(3.5 * cm)); story.append(Paragraph(text, h2n))
def bullets(items): story.append(ListFlowable([ListItem(Paragraph(i, bullet), leftIndent=14) for i in items], bulletType="bullet", start="•", leftIndent=14, bulletFontSize=9))
def simple_box(text, title="In simple words"):
    t = Table([[Paragraph(f"<b>{title}.</b> {text}", boxst)]], colWidths=[W])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), BOX), ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#8fbf8f")),
                           ("LEFTPADDING", (0, 0), (-1, -1), 9), ("RIGHTPADDING", (0, 0), (-1, -1), 9), ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 7)]))
    story.extend([t, Spacer(1, 9)])
def figure(path, caption, width=W):
    from PIL import Image as PI
    w, h = PI.open(path).size
    story.append(KeepTogether([Image(path, width=width, height=width * h / w), Paragraph(caption, cap)]))
def table(rows, widths, caption=None, align_right_from=1, text_cols=(), header_rows=1):
    data = []
    for ri, r in enumerate(rows):
        data.append([Paragraph(str(c), cellb if ri < header_rows else cell) if (ri < header_rows or ci in text_cols or ci == 0 and align_right_from > 0 and False) else str(c) for ci, c in enumerate(r)])
    t = Table(data, colWidths=widths, repeatRows=header_rows)
    st = [("BACKGROUND", (0, 0), (-1, header_rows - 1), LIGHT), ("FONTNAME", (0, header_rows), (-1, -1), "Sans"), ("FONTSIZE", (0, header_rows), (-1, -1), 8.2),
          ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("LINEBELOW", (0, 0), (-1, 0), 0.8, NAVY), ("LINEBELOW", (0, -1), (-1, -1), 0.8, NAVY),
          ("TOPPADDING", (0, 0), (-1, -1), 3.2), ("BOTTOMPADDING", (0, 0), (-1, -1), 3.2), ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4)]
    for ri in range(header_rows, len(rows)):
        if (ri - header_rows) % 2 == 1: st.append(("BACKGROUND", (0, ri), (-1, ri), ZEBRA))
    for ci in range(len(rows[0])):
        if ci >= align_right_from and ci not in text_cols: st.append(("ALIGN", (ci, header_rows), (ci, -1), "RIGHT"))
    t.setStyle(TableStyle(st))
    items = [t] + ([Paragraph(caption, cap)] if caption else [Spacer(1, 8)])
    story.append(KeepTogether(items))

# ------------------------------------------------------------------ page templates
class Doc(BaseDocTemplate):
    def afterFlowable(self, fl):
        if isinstance(fl, Paragraph) and fl.style.name in ("h1", "h2"):
            text = re.sub(r"<[^>]+>", "", fl.getPlainText()) if False else fl.getPlainText()
            self.notify("TOCEntry", (0 if fl.style.name == "h1" else 1, text, self.page - 1))
def header_footer(canv, doc):
    canv.saveState(); canv.setFont("Sans", 8); canv.setFillColor(colors.HexColor("#555555"))
    canv.drawString(2.2 * cm, A4[1] - 1.5 * cm, "Missing Data & Statistical Inference"); canv.drawRightString(A4[0] - 2.2 * cm, A4[1] - 1.5 * cm, "Project Report")
    canv.setStrokeColor(colors.HexColor("#999999")); canv.setLineWidth(0.5); canv.line(2.2 * cm, A4[1] - 1.7 * cm, A4[0] - 2.2 * cm, A4[1] - 1.7 * cm)
    canv.drawCentredString(A4[0] / 2, 1.3 * cm, str(doc.page - 1)); canv.restoreState()
def cover_page(canv, doc): pass

doc = Doc(f"{ROOT}/report/Missing_Data_Statistical_Inference_Report.pdf", pagesize=A4, leftMargin=2.2 * cm, rightMargin=2.2 * cm, topMargin=2.4 * cm, bottomMargin=2.0 * cm,
          title="Missing Data and Statistical Inference", author="")
frame = Frame(doc.leftMargin, doc.bottomMargin, W, A4[1] - doc.topMargin - doc.bottomMargin, id="f")
doc.addPageTemplates([PageTemplate("cover", [frame], onPage=cover_page), PageTemplate("main", [frame], onPage=header_footer)])

# ================================================================== COVER
story.append(Spacer(1, 5.5 * cm))
story.append(Paragraph("MISSING DATA &amp;<br/>STATISTICAL INFERENCE", ParagraphStyle("t", fontName="Sans-Bold", fontSize=30, leading=38, textColor=NAVY, alignment=TA_CENTER)))
story.append(Spacer(1, 0.8 * cm))
story.append(Paragraph("A Simulation Study of MCAR, MAR and MNAR", ParagraphStyle("t2", fontName="Sans", fontSize=15, leading=20, alignment=TA_CENTER, textColor=colors.HexColor("#333333"))))
story.append(Spacer(1, 0.4 * cm))
story.append(Paragraph("Comparing Eight Ways of Handling Missing Values<br/>Within Each Missingness Mechanism", ParagraphStyle("t3", fontName="Sans", fontSize=12, leading=17, alignment=TA_CENTER, textColor=colors.HexColor("#555555"))))
story.append(Spacer(1, 4.5 * cm))
story.append(Paragraph("Project Report<br/>M.Sc. Statistics<br/>October 2026", ParagraphStyle("t4", fontName="Sans", fontSize=12, leading=19, alignment=TA_CENTER, textColor=colors.HexColor("#333333"))))
story.append(NextPageTemplate("main")); story.append(PageBreak())

# ================================================================== SUMMARY + TOC
story.append(Paragraph("Summary", h1))
P("Real datasets very often have missing values. This project asks a simple question: <b>if some values are missing, does the way we deal with them change our results, and can we still trust the confidence interval that comes with them?</b>")
P(f"To answer it, we created data from a model we know, <i>Y</i> = 5 + 2<i>X</i><sub>1</sub> − 1.5<i>X</i><sub>2</sub> + <i>X</i><sub>3</sub> + error, with 200 observations. Because we know the model, we know the true value of the slope we care about, {B1} = 2. We then deleted about 20% of the values of <i>Y</i> and of <i>X</i><sub>1</sub> in three different ways: completely at random (MCAR), in a way that depends on another variable that is still observed (MAR), and in a way that depends on the value that disappears (MNAR). To every incomplete dataset we applied eight methods: complete-case analysis, mean imputation, hot-deck imputation, regression imputation, stochastic regression imputation, two versions of multiple imputation (MICE), and the EM algorithm.")
P("The study has two parts. In <b>Part A</b> one dataset is analysed step by step. In <b>Part B</b> the whole experiment is repeated 300 times with fresh data, so that we can see how each method behaves <i>on average</i>. All comparisons are made <b>within</b> each mechanism: for MCAR, then MAR, then MNAR, we ask which method works best.")
P("<b>Main findings</b>", ParagraphStyle("mf", parent=body, spaceAfter=3))
bullets([
    f"<b>Mean imputation is the worst method in every setting.</b> Its bias is {S('MCAR','Mean Imputation','Bias'):+.2f}, {S('MAR','Mean Imputation','Bias'):+.2f} and {S('MNAR','Mean Imputation','Bias'):+.2f} (MCAR, MAR, MNAR), and its 95% intervals contain the true value in only {S('MCAR','Mean Imputation','Coverage (%)'):.0f}%, {S('MAR','Mean Imputation','Coverage (%)'):.0f}% and {S('MNAR','Mean Imputation','Coverage (%)'):.0f}% of the runs.",
    "<b>Hot-deck, regression imputation and stochastic regression report standard errors that are too small</b> (roughly 70–85% of the true spread), so their intervals are too narrow. They treat made-up values as if they were real data.",
    f"<b>MCAR:</b> complete-case analysis works well; MICE and EM are slightly more precise.",
    f"<b>MAR:</b> complete-case analysis is biased and its intervals cover the truth only {S('MAR','Complete Case','Coverage (%)'):.0f}% of the time. EM and MICE (normal) are almost unbiased with about 95% coverage, because they use an extra observed variable.",
    f"<b>MNAR:</b> no method fixes the problem. Even the best intervals cover the truth only {min(S('MNAR',m,'Coverage (%)') for m in ['Complete Case','MICE (normal)','MICE (PMM)','EM']):.0f}–{max(S('MNAR',m,'Coverage (%)') for m in ['Complete Case','MICE (normal)','MICE (PMM)','EM']):.0f}% of the time.",
])
story.append(PageBreak())
story.append(Paragraph("Contents", h1x))
toc = TableOfContents()
toc.levelStyles = [ParagraphStyle("t0", fontName="Sans-Bold", fontSize=9, leading=11, leftIndent=0, spaceBefore=2.5),
                   ParagraphStyle("t1", fontName="Sans", fontSize=8, leading=9.6, leftIndent=16)]
story.append(toc); story.append(PageBreak())

# ================================================================== 1 INTRODUCTION
H1("1  Introduction")
P("Missing data are very common. A survey respondent skips a question, a machine fails to record a reading, a patient leaves a study early. When values are missing, the analysis changes for two reasons. First, there is less information. Second, the way we <i>handle</i> the missing values can change the relationships we find and how sure we are about them.")
P("How serious the problem is depends on <i>why</i> the values are missing. Statisticians describe three situations:")
bullets([
    "<b>MCAR (Missing Completely At Random).</b> Whether a value is missing has nothing to do with the data. Example: a sensor fails at random moments.",
    "<b>MAR (Missing At Random).</b> Whether a value is missing depends on something we <i>did</i> observe. Example: older people skip a question more often, and age is recorded for everyone.",
    "<b>MNAR (Missing Not At Random).</b> Whether a value is missing depends on the missing value itself. Example: people with very high incomes refuse to report their income.",
])
P("These three form a ladder: MCAR is the simplest case, MAR is more general, and MNAR is the most general (MCAR ⊂ MAR ⊂ MNAR).")
P("There are many ways to deal with missing values. We can delete the incomplete rows, fill the gaps with an average, copy values from similar rows, predict them with a regression, create several completed datasets, or estimate the model directly with likelihood methods. These choices can lead to different estimates <i>and</i> to different confidence intervals. This project compares them in a controlled experiment where the true answer is known.")

# ================================================================== 2 AIM
H1("2  Aim and Research Question")
H2n("2.1  Aim")
P(f"The aim is to find out how different missing-data methods affect a regression estimate and its uncertainty, when all methods are given the same incomplete data from the same missingness mechanism. We look at both the <b>estimate</b> itself and the <b>confidence interval</b> around it. The regression has four coefficients, but we focus on {B1}, the slope of <i>X</i><sub>1</sub>, whose true value is known to be {B1} = 2.")
H2n("2.2  Research question")
simple_box("<i>Within each type of missingness, which way of handling the missing values gives the most accurate estimate and the most honest confidence interval?</i> An “honest” interval is one that contains the true value about 95% of the time, as a 95% interval should.", "The question")
P("The question is answered three times: once for MCAR, once for MAR and once for MNAR. We do not rank the three mechanisms against each other. Each one is a separate setting in which the eight methods are compared.")

# ================================================================== 3 DATA
H1("3  How the Data Were Created")
P(f"A complete dataset was generated first, before any value was deleted. The outcome follows the linear model")
P("<i>Y</i> = 5 + 2<i>X</i><sub>1</sub> − 1.5<i>X</i><sub>2</sub> + 1<i>X</i><sub>3</sub> + ε,", ParagraphStyle("eq", parent=body, alignment=TA_CENTER))
P("where <i>X</i><sub>1</sub>, <i>X</i><sub>2</sub>, <i>X</i><sub>3</sub> are independent standard normal variables and the error ε is standard normal. Every dataset has <b>n = 200</b> observations. Because we chose the coefficients, the true values are known: 5, 2, −1.5 and 1.")
P(f"We also create an <b>auxiliary variable</b> <i>Z</i> = <i>Y</i> + noise. It is a noisy copy of <i>Y</i> (their correlation is about 0.94). <i>Z</i> is <b>not</b> part of the regression and is never missing, but it can help to fill in the gaps. Think of it as a cheap, rough measurement of the same thing, such as a self-reported value next to an exactly measured one.")
H2n("3.1  Baseline: the regression with no missing data")
P("Before deleting anything, the regression was fitted to the complete data. This is the best any method could hope to match.")
table([["Parameter", "True value", "Estimate from the complete data", "Standard error"]] +
      [[n, f"{t:.3f}", f"{e:.4f}", f"{se:.4f}"] for n, t, e, se in zip(["β0 (intercept)", "β1", "β2", "β3"], [5, 2, -1.5, 1], bfull, sefull)],
      [3.6 * cm, 3 * cm, 5.6 * cm, 3.5 * cm], "Table 1: Baseline regression on the complete data (no missing values).")
P(f"The complete-data estimate of {B1} is {FULL_B1:.3f}, close to but not exactly 2. This is normal: a dataset is only a sample, so even with no missing data the estimate has some random error. It is important to keep this in mind when reading the results: not every difference from 2 is caused by missing data.")

# ================================================================== 4 MISSINGNESS
H1("4  How Values Were Deleted")
P("About <b>20% of the values of <i>Y</i> and about 20% of the values of <i>X</i><sub>1</sub></b> were deleted, each one separately. Because two variables lose values, roughly a third of the rows have at least one gap. The three mechanisms differ only in <i>how likely</i> a value is to be deleted (Figure 1).")
bullets([
    "<b>MCAR:</b> every value has the same 20% chance of being deleted. It is pure luck.",
    "<b>MAR:</b> the chance of deletion depends on <i>Z</i>, which is always observed. Rows with a large <i>Z</i> lose their values more often.",
    "<b>MNAR:</b> the chance of deletion depends on the value that gets deleted. Large <i>Y</i> values and large <i>X</i><sub>1</sub> values vanish more often.",
])
P("For MAR and MNAR the chance of deletion is expit(−2.23 + 2 × <i>d</i>), where <i>d</i> is the standardised “driver” (<i>Z</i> for MAR, the value itself for MNAR) and expit is the S-shaped curve that turns any number into a probability between 0 and 1. The number −2.23 was tuned so that about 20% of values are deleted in every mechanism.")
figure(f"{ROOT}/figures/fig1_design.png", "Figure 1: (a) The complete data: Y increases with X1. (b) The chance that a value is deleted under each mechanism.")
P(f"Table 2 shows how many values were actually deleted in the one dataset used in Part A. The numbers are not exactly 20% because deletion is random.")
table([["Mechanism", "X1 missing", "Y missing", "Rows with at least one gap", "Complete rows left"]] +
      [[m, cnt_x1[i], cnt_y[i], f"{cnt_any[i]} ({cnt_any[i] / 2:.0f}%)", 200 - cnt_any[i]] for i, m in enumerate(MECHS)],
      [3 * cm, 2.8 * cm, 2.8 * cm, 4.6 * cm, 3.5 * cm], "Table 2: Missing values in the dataset used in Part A (out of 200 rows).")
P("We also checked that the rules behave as intended, using a very large dataset (20,000 rows) so that random noise does not hide the pattern (Table 3). Under MCAR, rows with a missing <i>Y</i> look just like the others. Under MAR and MNAR, rows with a missing <i>Y</i> have clearly higher <i>Z</i> and higher <i>Y</i>.")
table([["Mechanism", "Y missing (%)", "X1 missing (%)", "Rows with a gap (%)", "Z: missing minus observed", "Y: missing minus observed"]] +
      [[m] + [f"{x:.2f}" for x in chk[m]] for m in MECHS], [2.4 * cm, 2.6 * cm, 2.6 * cm, 3 * cm, 3 * cm, 3 * cm],
      "Table 3: Check of the three deletion rules on 20,000 rows.")
simple_box("In real data we can see <i>Z</i> but we cannot see the deleted <i>Y</i> values, so MAR and MNAR cannot be told apart from the observed data alone. We can only tell them apart here because we deleted the values ourselves.")

# ================================================================== 5 METHODS
H1("5  The Eight Methods")
P("Every method ends in the same step: a regression of <i>Y</i> on <i>X</i><sub>1</sub>, <i>X</i><sub>2</sub> and <i>X</i><sub>3</sub>. The methods differ only in what they do with the gaps before that regression.")
table([["Method", "How it works", "Good", "Not so good"],
       ["Complete case", "Delete every row with a gap.", "Simple. Fine under MCAR.", "Throws away rows; can be biased otherwise."],
       ["Mean imputation", "Fill each gap with the column average.", "Very simple.", "Shrinks the spread and weakens relationships."],
       ["Hot-deck", "Copy the value from the most similar row.", "Uses real values.", "One guess per gap; treated as if real."],
       ["Regression imputation", "Fill each gap with the value predicted by a regression.", "Uses the other variables.", "Filled values sit exactly on the line."],
       ["Stochastic regression", "Predicted value plus a random residual.", "Restores natural scatter.", "Still one guess per gap."],
       ["MICE (normal)", "Make 10 random completed datasets, analyse each, combine.", "Honest uncertainty.", "More work."],
       ["MICE (PMM)", "Same, but borrow real values from similar rows.", "Honest uncertainty; real values.", "Needs enough similar donors."],
       ["EM", "Maximum likelihood using every row, even partly observed ones.", "Uses all information.", "Assumes a normal distribution."]],
      [3.1 * cm, 5.6 * cm, 3.6 * cm, 4.3 * cm], "Table 4: The eight methods at a glance.", align_right_from=9, text_cols=(0, 1, 2, 3))
H2("5.1  Complete-case analysis")
P("Rows with any missing value are removed and the regression is fitted to the rest. No made-up values are created, but information is thrown away: in this study between 120 and 140 of the 200 rows remain.")
H2("5.2  Mean imputation")
P("Each gap is filled with the average of the observed values in that column. All filled values are identical, so the spread of the data shrinks and the link between <i>Y</i> and <i>X</i><sub>1</sub> becomes weaker.")
H2("5.3  Hot-deck imputation")
P("For each row with a gap, the most similar row (the nearest neighbour, judged on the variables that are observed in both rows, after putting them on the same scale) is found, and its value is copied. The filled values are real observed values, but each gap gets only one guess.")
H2("5.4  Regression imputation and stochastic regression")
P("Each incomplete variable is regressed on all the other variables, <i>Z</i> included, and the variables are filled in one at a time, cycling through them five times. In <b>regression imputation</b> the gap receives the predicted value. In <b>stochastic regression</b> a randomly chosen residual from the fitted regression is added, so the filled values scatter around the line the way real data do.")
H2("5.5  MICE: multiple imputation")
P("MICE (Multiple Imputation by Chained Equations) does not fill each gap just once. It creates <b>m = 10</b> different completed datasets. Each time, new regression coefficients are drawn first, so the uncertainty about the model itself is included. In the <b>normal</b> version the gap is filled with the prediction plus a random draw. In the <b>PMM</b> version (predictive mean matching) the five rows whose predicted values are closest are found, and the real observed value of one of them is copied.")
P("The regression is then run on every completed dataset and the 10 results are combined with <b>Rubin’s rules</b>. The estimate is the average of the 10 estimates. The total variance is")
P("total = within + (1 + 1/<i>m</i>) × between,", ParagraphStyle("eq2", parent=body, alignment=TA_CENTER))
P("where <i>within</i> is the average variance inside each completed dataset and <i>between</i> measures how much the 10 estimates disagree. The <i>between</i> part captures the extra uncertainty caused by the missing data. A method that fills each gap only once has no such part.")
H2("5.6  EM algorithm")
P("EM estimates the means and covariances of all five variables by maximum likelihood, using every row, including partly observed ones. It repeats two steps until nothing changes: the <b>E-step</b> fills each gap with its expected value given what is observed, and the <b>M-step</b> re-estimates the means and covariances. The regression coefficients are then calculated directly from them. EM has no simple formula for its standard error, so a bootstrap is used.")
H2("5.7  What the methods do to the data")
P("Figure 2 shows <i>Y</i> against <i>X</i><sub>1</sub> in the MAR dataset after each method has handled the gaps. Orange points are rows where a value was missing. The black line is the fitted line, with its slope printed above each panel. (This is the slope of <i>Y</i> on <i>X</i><sub>1</sub> alone, so for the full data it is 1.89 rather than the 2.037 of the full regression; compare the panels with each other.)")
figure(f"{ROOT}/figures/fig3_what_methods_do_MAR.png", "Figure 2: What each method did to the data (MAR dataset). Orange = rows where a value was missing.", width=W)
P("Complete case loses the top-right part of the cloud (rows with large <i>Z</i> are the ones that lose values under MAR), so its slope is too flat. Mean imputation puts the filled points on a horizontal and a vertical line through the average, which flattens the slope a lot. Regression imputation places the filled points exactly on the fitted line. The remaining methods scatter the filled points around the line like real data.")

# ================================================================== 6 EVALUATION
H1("6  How the Methods Are Judged")
H2n("6.1  Part A: one dataset")
P(f"In Part A each method is applied once to each of the three incomplete datasets. For {B1} we report the estimate, its <b>bias</b> (estimate − 2), a standard error and a 95% confidence interval. Two kinds of difference are shown:")
bullets(["<b>Bias vs truth</b> = estimate − 2. This mixes the damage done by missing data with ordinary random error.",
         f"<b>Difference from the full-data estimate</b> = estimate − {FULL_B1:.3f}. This isolates what the missing data did."])
P("The standard error is found in one of two ways. For every method except MICE we use the <b>bootstrap</b>: the rows are resampled with replacement 500 times, the <i>whole</i> method (including the filling-in) is repeated each time, and the spread of the 500 estimates gives the standard error. The middle 95% of the 500 estimates gives the interval. For MICE we use Rubin’s rules. We also show the <b>naive SE</b>, which is what you get by running the regression on the completed data and forgetting that some values were made up.")
H2n("6.2  Part B: 300 repetitions")
P("One dataset can mislead, because one dataset is one random draw. In Part B the whole experiment is repeated <b>300 times</b>: each time a new complete dataset is generated, damaged under each mechanism, and analysed with every method. We also analyse the same data with <i>no</i> missing values as a benchmark. The measures are in Table 5.")
table([["Measure", "What it tells us"],
       ["Bias", "Average of (estimate − 2). Zero means right on average."],
       ["Spread (SD)", "How much the estimate varies from one dataset to the next: the true uncertainty."],
       ["RMSE", "Overall accuracy; combines bias and spread. Smaller is better."],
       ["Reported SE", "The standard error the method itself reports. If it is clearly smaller than the spread, the method is over-confident."],
       ["Coverage (%)", "Share of the 300 intervals (estimate ± 1.96 × SE) that contain the true value 2. It should be about 95."]],
      [3.2 * cm, 13.4 * cm], "Table 5: Measures used in the simulation study.", align_right_from=9, text_cols=(0, 1))
P("With 300 repetitions, a coverage figure near 95% has a random error of about 1.3 percentage points, so differences of 2–3 points or less should not be over-read. How each method gets its standard error inside the simulation is listed in Appendix B.")
bench = s[(s.Mechanism == "MCAR") & (s.Method == "Full data")].iloc[0]
simple_box(f"With no missing data, the spread of the estimate is {bench['SD']:.3f} and the intervals cover the truth {bench['Coverage (%)']:.0f}% of the time. No method can beat this benchmark. It also shows that the simulation itself works as it should.", "Benchmark")

# ================================================================== 7 PART A RESULTS
H1("7  Results: Part A (One Dataset)")
P(f"The complete-data estimate for this dataset is {FULL_B1:.3f}. Tables 6 to 8 show every method under each mechanism, and Figure 3 shows the same intervals.")
for k, mech in enumerate(MECHANISMS if False else MECHS):
    rows = [["Method", "β1 estimate", "Naive SE", "Final SE", "95% interval", "Bias vs truth", "Diff. from full data"]]
    for m in METHODS:
        rows.append([m, f"{A(mech,m,'Beta1'):.3f}", "–" if np.isnan(A(mech, m, 'Naive SE')) else f"{A(mech,m,'Naive SE'):.3f}", f"{A(mech,m,'Final SE'):.3f}",
                     f"({A(mech,m,'CI_Low'):.3f}, {A(mech,m,'CI_High'):.3f})", f"{A(mech,m,'Bias vs truth'):+.3f}", f"{A(mech,m,'Diff vs full data'):+.3f}"])
    table(rows, [3.8 * cm, 2.0 * cm, 1.8 * cm, 1.8 * cm, 3.0 * cm, 2.1 * cm, 2.3 * cm], f"Table {6 + k}: Part A, {mech}: all eight methods on the same dataset.")
figure(f"{ROOT}/figures/fig2_partA_intervals.png", "Figure 3: Part A. β1 estimate with 95% interval for each method. Red dashed line = true value 2; dotted line = full-data estimate.")
H2n("7.1  What Part A shows")
mi = [A(m, "Mean Imputation", "Beta1") for m in MECHS]
bullets([
    f"<b>Mean imputation is far below the full-data estimate</b> in all three settings ({mi[0]:.2f}, {mi[1]:.2f} and {mi[2]:.2f}, compared with {FULL_B1:.2f}).",
    f"<b>The naive SE is smaller than the final SE for hot-deck, regression imputation and stochastic regression</b> in all three settings. For example, hot-deck under MCAR has a naive SE of {A('MCAR','Hot-Deck','Naive SE'):.3f} but a bootstrap SE of {A('MCAR','Hot-Deck','Final SE'):.3f}. Ignoring that values were made up makes the method look more certain than it is.",
    f"<b>For MICE the naive and final SEs are the same</b>, because Rubin’s rules already include the extra uncertainty.",
    f"<b>One dataset can mislead.</b> Complete case looks fine under MNAR here ({A('MNAR','Complete Case','Beta1'):.3f}), yet Part B shows it is biased on average. This is why a simulation study is needed.",
])

# ================================================================== 8 PART B RESULTS
H1("8  Results: Part B (300 Repetitions)")
P("For each mechanism we show a table ranked by RMSE (best first) and a figure with three panels: (a) the spread of the 300 estimates, (b) the coverage of the 95% interval, and (c) the reported SE divided by the true spread. The colours group the methods: grey = delete incomplete rows, orange = single imputation, blue = multiple imputation and EM.")
def partb_table(mech, num):
    t = s[(s.Mechanism == mech) & (s.Method != "Full data")].sort_values("RMSE")
    rows = [["Rank", "Method", "Bias", "Spread (SD)", "RMSE", "Reported SE", "Coverage (%)"]]
    for i, (_, r) in enumerate(t.iterrows(), 1): rows.append([i, r.Method, f"{r.Bias:+.3f}", f"{r.SD:.3f}", f"{r.RMSE:.3f}", f"{r.Avg_SE:.3f}", f"{r['Coverage (%)']:.1f}"])
    table(rows, [1.2 * cm, 4.2 * cm, 2.0 * cm, 2.4 * cm, 2.0 * cm, 2.5 * cm, 2.3 * cm], f"Table {num}: Part B, {mech}: methods ranked by RMSE (300 repetitions).", align_right_from=2)
def g(mech, m, c): return S(mech, m, c)

H2("8.1  MCAR: the gaps are pure luck")
partb_table("MCAR", 9)
figure(f"{ROOT}/figures/fig4_simulation_MCAR.png", "Figure 4: MCAR. Spread of the estimates, coverage of the 95% interval, and honesty of the reported SE.")
bullets([
    f"<b>Complete case is fine.</b> It is right on average (bias {g('MCAR','Complete Case','Bias'):+.3f}) and its intervals cover the truth {g('MCAR','Complete Case','Coverage (%)'):.0f}% of the time. Its only cost is precision, because it uses fewer rows.",
    f"<b>MICE (normal), MICE (PMM) and EM</b> have about the same accuracy as each other (RMSE {g('MCAR','MICE (normal)','RMSE'):.3f}, {g('MCAR','MICE (PMM)','RMSE'):.3f} and {g('MCAR','EM','RMSE'):.3f}), slightly better than complete case ({g('MCAR','Complete Case','RMSE'):.3f}), with coverage close to 95%.",
    f"<b>Single imputation fails even though the data are MCAR.</b> Mean imputation is strongly biased ({g('MCAR','Mean Imputation','Bias'):+.2f}), regression imputation is biased upward ({g('MCAR','Regression Imputation','Bias'):+.2f}) and hot-deck downward ({g('MCAR','Hot-Deck','Bias'):+.2f}). Stochastic regression is almost unbiased ({g('MCAR','Stochastic Regression','Bias'):+.3f}), but its coverage is only {g('MCAR','Stochastic Regression','Coverage (%)'):.0f}%, because its reported SE is too small.",
])
H2("8.2  MAR: the gaps depend on an observed variable")
partb_table("MAR", 10)
figure(f"{ROOT}/figures/fig5_simulation_MAR.png", "Figure 5: MAR. Spread of the estimates, coverage of the 95% interval, and honesty of the reported SE.")
bullets([
    f"<b>Complete case is no longer safe.</b> It is biased downward ({g('MAR','Complete Case','Bias'):+.3f}) and its coverage falls to {g('MAR','Complete Case','Coverage (%)'):.0f}%. The rows with a missing <i>Y</i> are not a random sample: they tend to have large <i>Z</i>, which goes with large <i>Y</i>.",
    f"<b>EM and MICE (normal) fix this.</b> They use <i>Z</i>, are almost unbiased ({g('MAR','EM','Bias'):+.3f} and {g('MAR','MICE (normal)','Bias'):+.3f}), cover {g('MAR','EM','Coverage (%)'):.0f}% and {g('MAR','MICE (normal)','Coverage (%)'):.0f}% of the time, and have much lower RMSE than complete case ({g('MAR','EM','RMSE'):.3f} and {g('MAR','MICE (normal)','RMSE'):.3f} against {g('MAR','Complete Case','RMSE'):.3f}).",
    f"<b>MICE (PMM)</b> has a small downward bias ({g('MAR','MICE (PMM)','Bias'):+.3f}) and lower coverage ({g('MAR','MICE (PMM)','Coverage (%)'):.0f}%).",
    f"<b>Single imputation again under-covers.</b> Stochastic regression is unbiased ({g('MAR','Stochastic Regression','Bias'):+.3f}) but covers only {g('MAR','Stochastic Regression','Coverage (%)'):.0f}%. Hot-deck ({g('MAR','Hot-Deck','Coverage (%)'):.0f}%) and regression imputation ({g('MAR','Regression Imputation','Coverage (%)'):.0f}%) are biased as well. Mean imputation is by far the worst ({g('MAR','Mean Imputation','Coverage (%)'):.0f}%).",
])
P("Figure 6 makes coverage concrete. It shows the first 40 of the 300 intervals for four methods. Green intervals contain the true value 2 and red ones miss it. Complete case misses it too often because its intervals are centred too low. Stochastic regression is centred correctly but its intervals are too short.")
figure(f"{ROOT}/figures/fig7_intervals_MAR.png", "Figure 6: MAR. The first 40 confidence intervals of four methods (green = contains the truth, red = misses it).")
H2("8.3  MNAR: the gaps depend on the missing value itself")
partb_table("MNAR", 11)
figure(f"{ROOT}/figures/fig6_simulation_MNAR.png", "Figure 7: MNAR. Spread of the estimates, coverage of the 95% interval, and honesty of the reported SE.")
bullets([
    f"<b>No method is reliable.</b> Complete case is biased downward ({g('MNAR','Complete Case','Bias'):+.3f}), while EM, MICE (normal) and stochastic regression, which all assume the data are MAR, overshoot upward ({g('MNAR','EM','Bias'):+.3f}, {g('MNAR','MICE (normal)','Bias'):+.3f}, {g('MNAR','Stochastic Regression','Bias'):+.3f}). Even the best intervals cover only {min(g('MNAR',m,'Coverage (%)') for m in ['Complete Case','MICE (normal)','MICE (PMM)','EM']):.0f}–{max(g('MNAR',m,'Coverage (%)') for m in ['Complete Case','MICE (normal)','MICE (PMM)','EM']):.0f}% of the time.",
    f"<b>The leading methods are hard to separate.</b> Complete case, MICE (normal), MICE (PMM) and EM have similar RMSE ({min(g('MNAR',m,'RMSE') for m in ['Complete Case','MICE (normal)','MICE (PMM)','EM']):.3f}–{max(g('MNAR',m,'RMSE') for m in ['Complete Case','MICE (normal)','MICE (PMM)','EM']):.3f}), within random error of each other. The small bias of MICE (PMM) is probably two errors that happen to cancel, not a sign that it handles MNAR.",
    f"<b>Regression imputation</b> (coverage {g('MNAR','Regression Imputation','Coverage (%)'):.0f}%) and <b>mean imputation</b> (coverage {g('MNAR','Mean Imputation','Coverage (%)'):.0f}%) are clearly the worst.",
])
H2("8.4  At a glance")
P("Figure 8 puts all the results on one page. Each row is a method and each column a mechanism, so you can read down a column to compare the methods under one mechanism. Table 12 lists the best method for each mechanism according to several criteria.")
figure(f"{ROOT}/figures/fig8_at_a_glance.png", "Figure 8: Bias (left) and coverage (right) of every method under every mechanism.")
wins = []
for mech in MECHS:
    t = s[(s.Mechanism == mech) & (s.Method != "Full data")].set_index("Method")
    honest = t[t["Coverage (%)"] >= 90]
    wins.append([mech, t.RMSE.idxmin(), t.Bias.abs().idxmin(), (t["Coverage (%)"] - 95).abs().idxmin(), honest.RMSE.idxmin() if len(honest) else "none"])
table([["Mechanism", "Lowest RMSE", "Smallest bias", "Coverage closest to 95%", "Lowest RMSE among methods with coverage ≥ 90%"]] + wins,
      [2.4 * cm, 3.3 * cm, 3.3 * cm, 3.5 * cm, 4.1 * cm], "Table 12: The best method under each mechanism, by four criteria.", align_right_from=9, text_cols=(0, 1, 2, 3, 4))
P("Where the leading methods are within random error of each other (see the notes above), the “best” in this table should not be taken as a real difference.", small)

# ================================================================== 9 DISCUSSION
H1("9  Discussion")
H2("9.1  Mean imputation hurts everywhere")
P("Filling every gap with the same number removes the natural variation in <i>Y</i> and <i>X</i><sub>1</sub>, and with it part of the relationship between them (Figure 2). The slope is pulled towards zero. This is not a small effect: the bias is between −0.40 and −0.55 in all three settings, and the interval almost never contains the truth. It does not matter whether the data are MCAR, MAR or MNAR.")
H2("9.2  Why single imputation is over-confident")
P(f"Hot-deck, regression imputation and stochastic regression fill each gap once and then treat the result as real data. But about a third of the rows contain made-up values, so the data look more certain than they are. The reported SE is only about 70–85% of the true spread (panel (c) of Figures 4, 5 and 7), and the intervals are too narrow. Stochastic regression is the clearest example: under MCAR and MAR it has almost no bias, yet its 95% intervals contain the truth only about {S('MCAR','Stochastic Regression','Coverage (%)'):.0f}% of the time.")
simple_box("A good estimate is not enough. A method can get the right answer on average and still give you a confidence interval you should not believe.")
H2("9.3  What multiple imputation and EM do differently")
P("MICE (normal) uses almost the same filling-in model as stochastic regression. It adds only three things: new coefficients are drawn each time, the filling-in is repeated 10 times, and Rubin’s rules combine the results. These changes are enough to bring the coverage from clearly below 95% up to about 95% under MCAR and MAR. EM reaches the same accuracy by a different route, by using the likelihood of all the observed data.")
H2("9.4  Why complete case fails under MAR here, and the role of Z")
P("Under MAR the rows that lose values are those with a large <i>Z</i>. Since <i>Z</i> is closely tied to <i>Y</i>, the complete rows are a biased selection of the data. Methods that use <i>Z</i> in the filling-in step (EM and MICE) can correct for this. Complete case cannot, because it ignores <i>Z</i>. The lesson: <b>variables that explain why data are missing belong in the imputation model, even if they are not in the analysis model.</b> In this study <i>Z</i> is a very good stand-in for <i>Y</i>; in real data the auxiliary variables are usually weaker, so the advantage would be smaller.")
H2("9.5  PMM versus the normal version of MICE")
P("Under MAR, MICE (PMM) shows a small downward bias and lower coverage than MICE (normal). Here the data really are linear and normal, so the normal model is exactly right and PMM’s usual strength, being robust to unusual distributions, is not needed. A likely reason for its weakness is that PMM can only borrow values that were observed, and the rows that MAR targets (those with extreme predicted values) have few similar donors. PMM would be expected to do better with skewed or non-linear data, which this study does not contain.")
H2("9.6  MNAR is different in kind")
P("Under MNAR the chance of being missing depends on the missing value itself, so the observed data cannot reveal the missingness process. The methods that assume MAR cannot correct for it, and none of the methods tried here does. When data may be MNAR, the honest approach is a <b>sensitivity analysis</b>: repeat the analysis under several plausible assumptions about the missing values and report how the conclusions change.")
H2("9.7  Why the results differ from an earlier one-variable version of this project")
P("In an earlier version of this project only <i>Y</i> had missing values and the predictors were complete. In that setting complete case, regression imputation and EM give exactly the same coefficients, which hides their differences. In this version two variables (<i>Y</i> and <i>X</i><sub>1</sub>) have gaps and an auxiliary variable <i>Z</i> is available, so the methods really do behave differently. The sample size is also larger (200 instead of 100), and the experiment is repeated 300 times instead of being based on a single dataset.")

# ================================================================== 10 LIMITATIONS
H1("10  Limitations")
bullets([
    "<b>One simple model.</b> The data are linear and normal, the predictors are independent, n = 200 and about 20% of two variables are missing. Results can change with other distributions, sample sizes or amounts of missing data. In particular, PMM is designed for non-normal data, which this study does not contain.",
    "<b>Chosen strengths.</b> The strength of the MAR and MNAR rules (the number 2 in the formula) was chosen by us, and MNAR is only one of many possible forms.",
    "<b>A very helpful extra variable.</b> <i>Z</i> is a close stand-in for <i>Y</i>. Real auxiliary variables are usually weaker, which would shrink the advantage of MICE and EM over complete case under MAR.",
    "<b>Simplified methods.</b> MICE uses 5 cycles, 10 imputations, 5 donors and a normal (±1.96) interval rather than Rubin’s t-based interval. EM assumes a normal distribution and, inside Part B, takes its standard error from only 40 bootstrap resamples. For real work use a tested package such as <i>mice</i> in R, <i>statsmodels</i> or scikit-learn’s <i>IterativeImputer</i>.",
    "<b>Random error.</b> With 300 repetitions, coverage is accurate to about ±1.3 points. Part A is a single dataset and should not be used to rank methods.",
    f"<b>Only {B1} is examined</b>, not the other coefficients or predictions.",
])

# ================================================================== 11 CONCLUSION
H1("11  Conclusion")
P("How missing data are handled is part of the statistical analysis, not a detail to be skipped. It changes the estimate and it changes whether the confidence interval can be trusted. Table 13 summarises what works under each mechanism.")
table([["Mechanism", "What works", "What to avoid"],
       ["MCAR", "Complete case is acceptable. MICE or EM are slightly more precise.", "Mean imputation and single imputation."],
       ["MAR", "EM or MICE (normal), with the helpful extra variable in the imputation model.", "Complete case (biased) and single imputation (over-confident)."],
       ["MNAR", "Nothing is reliable. Report a sensitivity analysis.", "Claiming that any of these methods fixes MNAR."]],
      [2.6 * cm, 7.4 * cm, 6.6 * cm], "Table 13: Recommendations by mechanism.", align_right_from=9, text_cols=(0, 1, 2))
P("In every case:")
bullets(["<b>Never use mean imputation</b> for inference.",
         "<b>A good point estimate is not enough.</b> Single imputation can be nearly unbiased and still badly over-confident. Carry the uncertainty through with multiple imputation (Rubin’s rules), EM, or a bootstrap that repeats the filling-in.",
         "<b>Use all the information you have.</b> Include variables that explain the missingness or the outcome in the imputation model."])

# ================================================================== APPENDICES
story.append(PageBreak())
H1("Appendix A: A Worked MICE Example")
P("This appendix shows the numbers behind MICE (normal) for the three datasets of Part A. Each row lists the 10 estimates of β<sub>1</sub> (one per completed dataset), then the pooled estimate and the variances used by Rubin’s rules.")
rows = [["Mechanism"] + [f"m{i}" for i in range(1, 11)]]
for m in MECHS: rows.append([m] + [f"{x:.3f}" for x in mice[m][0]])
table(rows, [2.4 * cm] + [1.42 * cm] * 10, "Table A1: The 10 estimates of β1 from the 10 completed datasets (MICE, normal version).")
rows = [["Mechanism", "Pooled β1", "Within (W)", "Between (B)", "Total = W + 1.1 B", "Pooled SE"]]
for m in MECHS:
    betas, w, b = mice[m]; tot = w + (1 + 1 / 10) * b
    rows.append([m, f"{np.mean(betas):.3f}", f"{w:.5f}", f"{b:.5f}", f"{tot:.5f}", f"{np.sqrt(tot):.3f}"])
table(rows, [2.4 * cm, 2.4 * cm, 2.8 * cm, 2.8 * cm, 3.4 * cm, 2.4 * cm], "Table A2: Rubin’s rules applied to the 10 estimates (m = 10, so 1 + 1/m = 1.1).")
P("The within-imputation variance is the ordinary regression uncertainty inside one completed dataset. The between-imputation variance shows that different plausible fillings of the gaps lead to slightly different estimates. Single imputation has only the first part, which is why its standard error is too small.")

H1("Appendix B: Settings and How to Reproduce")
table([["Setting", "Value"],
       ["Observations per dataset", "200"],
       ["True model", "Y = 5 + 2·X1 − 1.5·X2 + 1·X3 + ε,  Z = Y + noise"],
       ["Missing values", "about 20% of Y and about 20% of X1"],
       ["Part A", "seed 42; 500 bootstrap samples per method"],
       ["Part B", "seed 2024; 300 repetitions; 40 bootstrap samples for EM’s SE"],
       ["Standard errors in Part B", "naive OLS SE (complete case and single imputation); Rubin’s rules (MICE); bootstrap (EM)"],
       ["MICE", "10 imputations, 5 cycles, PMM with 5 donors"],
       ["Software", "Python with numpy, pandas, scipy, scikit-learn, statsmodels, matplotlib"],
       ["Run time", "about 8 minutes for the full notebook"]],
      [5 * cm, 11.6 * cm], None, align_right_from=9, text_cols=(0, 1))
P("All code, the executed notebook, the figures and the result tables are in the project repository (see the README). Every code block in the notebook starts from its own random seed, so any block can be re-run on its own and gives the same answer.")

# ================================================================== REFERENCES
H1("References")
for r in ["Dempster, A. P., Laird, N. M., and Rubin, D. B. (1977). Maximum likelihood from incomplete data via the EM algorithm. <i>Journal of the Royal Statistical Society: Series B</i>, 39(1), 1–38.",
          "Rubin, D. B. (1987). <i>Multiple Imputation for Nonresponse in Surveys</i>. Wiley.",
          "van Buuren, S. (2018). <i>Flexible Imputation of Missing Data</i>, 2nd ed. Chapman &amp; Hall/CRC.",
          "Koley, D. Missing Data – Unit I: Rigorous Treatment. Course material, Department of Statistics, St. Xavier’s University, Kolkata.",
          "Koley, D. Missing Data – Unit II: Imputation Methods and Algorithmic Approaches. Course material, Department of Statistics, St. Xavier’s University, Kolkata."]:
    P(r, ParagraphStyle("ref", parent=body, leftIndent=14, firstLineIndent=-14, alignment=TA_LEFT))

doc.multiBuild(story)
print("PDF built")
