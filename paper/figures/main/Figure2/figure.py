"""Figure 2: retained September statistical base plus new classical panels.

Run: python figure.py
The adjacent base/figure.py retains the current manuscript's six-panel source,
including supplied method/task and rank confidence intervals. Only its canvas
validation accepts the extended height through an optional argument.
"""
from pathlib import Path
import importlib.util
import json
from io import StringIO
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
from matplotlib.text import Text
import matplotlib.pyplot as plt
from new_panels import add_panels

HERE = Path(__file__).resolve().parent
BASE_PATH = HERE / "base" / "figure.py"
SOURCE_BASE = "runs/STRSC-PDF-CODE-DELIVERY-20260914/manuscript_source_workspace/source_data/rank_uncertainty/figure_source/main_source/figure.py"


def load_base():
    spec = importlib.util.spec_from_file_location("retained_figure2_base", BASE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_retained_source_data(base):
    rows = []
    for panel, task in zip("abc", base.TASKS):
        for method in base.METHODS:
            for index, patient in enumerate(base.PATIENTS):
                rows.append(dict(panel=panel, task=task, method=method, patient=patient,
                                 statistic="patient_ordering_gain", value=base.PATIENT_VALUES[task][method][index]))
            rows.append(dict(panel=panel, task=task, method=method, patient="EQUAL_PATIENT_MEAN",
                             statistic="plotted_method_mean", value=base.SUMMARY[task][method][0]))
        rows.append(dict(panel=panel, task=task, method="MEDIAN_ACROSS_11_METHODS", patient="EQUAL_PATIENT_MEAN",
                         statistic="plotted_task_median", value=base.MEDIANS[task]))
    pd.DataFrame(rows).to_csv(HERE / "SourceData_a_c.tsv", sep="\t", index=False)
    pd.read_csv(StringIO(base.METHOD_INTERVALS_TSV), sep="\t").to_csv(
        HERE / "SourceData_a_c_method_intervals.tsv", sep="\t", index=False)
    pd.read_csv(StringIO(base.TASK_INTERVALS_TSV), sep="\t").to_csv(
        HERE / "SourceData_a_c_task_intervals.tsv", sep="\t", index=False)
    ranks = []
    for panel, task in zip("def", base.TASKS):
        for method in base.METHODS:
            ranks.append(dict(panel=panel, task=task, method=method,
                              overall_pearson_rank=base.OVERALL_RANK[method],
                              task_rank=base.TASK_RANK[task][method],
                              tau=base.EXPECTED_TAU[task],
                              tau_ci_low=base.RANK_TAU_INTERVALS[task][1],
                              tau_ci_high=base.RANK_TAU_INTERVALS[task][2]))
    pd.DataFrame(ranks).to_csv(HERE / "SourceData_d_f.tsv", sep="\t", index=False)


def main():
    base = load_base()
    base.validate_data()
    plt.rcParams.update({"svg.fonttype": "none", "font.weight": "normal", "axes.labelweight": "normal"})
    fig, *artists = base.make_figure()
    width, old_height = fig.get_size_inches()
    new_height = 7.95
    extra = new_height - old_height
    # Preserve the six-panel module in physical inches, translating it upward.
    fig.set_size_inches(width, new_height)
    for ax in fig.axes:
        x, y, w, h = ax.get_position().bounds
        ax.set_position([x, (y * old_height + extra) / new_height, w, h * old_height / new_height])
    for label in fig.texts:
        x, y = label.get_position()
        label.set_position([x, (y * old_height + extra) / new_height])
        if (label.get_gid() or "").startswith("panel_letter_"):
            label.set_fontsize(10)
    new_axes, summaries, case = add_panels(fig, base)
    write_retained_source_data(base)
    # Reuse the existing direct data/CI/layout checks on the retained module.
    base.PNG_PATH = HERE / "Figure2.png"
    base.PDF_PATH = HERE / "Figure2.pdf"
    base.CHECK_PATH = HERE / "render_check.txt"
    base.INTERVAL_PATH = HERE / "SourceData_intervals_plotted.tsv"
    base.run_floor_checks(fig, *artists, expected_size_inches=(width, new_height))
    fig.savefig(HERE / "Figure2.svg", facecolor="white", bbox_inches=None, pad_inches=0)
    visible = [item for item in fig.findobj(Text) if item.get_visible() and item.get_text().strip()]
    assert min(item.get_fontsize() for item in visible) >= 7
    assert np.isclose(width * 25.4, 180, atol=.001)
    receipt = dict(status="COMPLETE", width_mm=width * 25.4, height_in=new_height,
                   dpi=300, minimum_font_pt=min(item.get_fontsize() for item in visible),
                   retained_source=SOURCE_BASE, retained_patient_points=264,
                   retained_method_means=33, retained_method_intervals=33,
                   retained_task_intervals=3, retained_rank_points=33, retained_tau_intervals=3,
                   paired_patients=8, unchanged_models=11,
                   spearman_means=summaries["spearman"].tolist(),
                   ordinary_C_means=summaries["C"].tolist(),
                   case_gene=case.gene, case_model=case.model, case_patient=case.patient,
                   case_section=case.section, case_is_exploratory=True)
    (HERE / "COMPLETE.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    plt.close(fig)
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
