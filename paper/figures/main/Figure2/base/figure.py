from pathlib import Path
import csv
from io import StringIO
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
from PIL import Image


plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"]
plt.rcParams["font.size"] = 7.0


# === DATA SECTOR (edit here) ===
TASKS = ["Across regions", "Similar cell mix", "Within regions"]
METHODS = [
    "EGNv2",
    "Hist2ST",
    "ST-Net",
    "DeepPT",
    "TCGN",
    "HisToGene",
    "DeepSpaCE",
    "THItoGene",
    "EGNv1",
    "GeneCodeR",
    "iStar",
]
PATIENTS = list("ABCDEFGH")

# Exact statistical-completion rows supplied on 13 Sep 2026.
# Raw fractions below are plotted directly; percentage columns are for text only.
# Intervals are pointwise approximate 95% patient-block percentile bootstrap CIs.
# These are not simultaneous intervals or exact-test inversions.
METHOD_INTERVALS_TSV = """method	task	source_task	estimate	ci_low	ci_high	estimate_percent	ci_low_percent	ci_high_percent	estimator	n_patients	n_methods	interval	bootstrap_draws	seed	source
DeepPT	Across regions	Across regions	0.27614028015778436	0.13869464531104692	0.4127802033689678	27.614028015778437	13.869464531104692	41.27802033689678	equal-patient mean ordering gain	8	11	95% patient-block percentile bootstrap, pointwise	20000	2026091301	output/stRSC_Source_Data_v1/tsv/F1_score_components.tsv
DeepPT	RNA context + depth matched	Similar cell mix	0.07862151293645471	0.03240782831736388	0.1265755986218377	7.862151293645471	3.2407828317363876	12.65755986218377	equal-patient mean ordering gain	8	11	95% patient-block percentile bootstrap, pointwise	20000	2026091301	output/stRSC_Source_Data_v1/tsv/F1_score_components.tsv
DeepPT	Within regions	Within regions	0.04913489821862535	0.009116094742123713	0.09285440111256647	4.913489821862535	0.9116094742123713	9.285440111256648	equal-patient mean ordering gain	8	11	95% patient-block percentile bootstrap, pointwise	20000	2026091301	output/stRSC_Source_Data_v1/tsv/F1_score_components.tsv
DeepSpaCE	Across regions	Across regions	0.2886734288423047	0.15702253408229952	0.41516676463442526	28.867342884230467	15.702253408229952	41.51667646344252	equal-patient mean ordering gain	8	11	95% patient-block percentile bootstrap, pointwise	20000	2026091301	output/stRSC_Source_Data_v1/tsv/F1_score_components.tsv
DeepSpaCE	RNA context + depth matched	Similar cell mix	0.11525359649542648	0.04882554765310433	0.17850271517947885	11.525359649542647	4.8825547653104335	17.850271517947885	equal-patient mean ordering gain	8	11	95% patient-block percentile bootstrap, pointwise	20000	2026091301	output/stRSC_Source_Data_v1/tsv/F1_score_components.tsv
DeepSpaCE	Within regions	Within regions	0.11594637846207298	0.0454868604421538	0.18266727342857897	11.594637846207299	4.54868604421538	18.266727342857898	equal-patient mean ordering gain	8	11	95% patient-block percentile bootstrap, pointwise	20000	2026091301	output/stRSC_Source_Data_v1/tsv/F1_score_components.tsv
EGNv1	Across regions	Across regions	0.2110097927776105	0.10841239746161921	0.3280751785233325	21.100979277761052	10.841239746161921	32.80751785233325	equal-patient mean ordering gain	8	11	95% patient-block percentile bootstrap, pointwise	20000	2026091301	output/stRSC_Source_Data_v1/tsv/F1_score_components.tsv
EGNv1	RNA context + depth matched	Similar cell mix	0.0610803418906581	0.01800263297067699	0.12007992213039986	6.10803418906581	1.800263297067699	12.007992213039987	equal-patient mean ordering gain	8	11	95% patient-block percentile bootstrap, pointwise	20000	2026091301	output/stRSC_Source_Data_v1/tsv/F1_score_components.tsv
EGNv1	Within regions	Within regions	0.041420457038257016	0.003191216642859825	0.0980242692149378	4.142045703825701	0.3191216642859825	9.80242692149378	equal-patient mean ordering gain	8	11	95% patient-block percentile bootstrap, pointwise	20000	2026091301	output/stRSC_Source_Data_v1/tsv/F1_score_components.tsv
EGNv2	Across regions	Across regions	0.3735169841705678	0.20796259545266263	0.5346396939758077	37.35169841705678	20.796259545266263	53.46396939758077	equal-patient mean ordering gain	8	11	95% patient-block percentile bootstrap, pointwise	20000	2026091301	output/stRSC_Source_Data_v1/tsv/F1_score_components.tsv
EGNv2	RNA context + depth matched	Similar cell mix	0.06332700052796805	0.007725684190382915	0.12083258085019363	6.332700052796805	0.7725684190382915	12.083258085019363	equal-patient mean ordering gain	8	11	95% patient-block percentile bootstrap, pointwise	20000	2026091301	output/stRSC_Source_Data_v1/tsv/F1_score_components.tsv
EGNv2	Within regions	Within regions	0.05185871346426914	0.011782206287488562	0.09434699314452542	5.1858713464269135	1.1782206287488561	9.434699314452542	equal-patient mean ordering gain	8	11	95% patient-block percentile bootstrap, pointwise	20000	2026091301	output/stRSC_Source_Data_v1/tsv/F1_score_components.tsv
GeneCodeR	Across regions	Across regions	0.2351117248507964	0.11591429052933871	0.3596673755269145	23.51117248507964	11.591429052933872	35.966737552691455	equal-patient mean ordering gain	8	11	95% patient-block percentile bootstrap, pointwise	20000	2026091301	output/stRSC_Source_Data_v1/tsv/F1_score_components.tsv
GeneCodeR	RNA context + depth matched	Similar cell mix	0.05384193087470636	0.013358090499778163	0.0985180499691584	5.384193087470636	1.3358090499778164	9.85180499691584	equal-patient mean ordering gain	8	11	95% patient-block percentile bootstrap, pointwise	20000	2026091301	output/stRSC_Source_Data_v1/tsv/F1_score_components.tsv
GeneCodeR	Within regions	Within regions	0.04859287308172071	0.0155487153191205	0.08224337451038945	4.859287308172071	1.55487153191205	8.224337451038945	equal-patient mean ordering gain	8	11	95% patient-block percentile bootstrap, pointwise	20000	2026091301	output/stRSC_Source_Data_v1/tsv/F1_score_components.tsv
HisToGene	Across regions	Across regions	0.30370330242293725	0.1459532638018391	0.44933606191545095	30.370330242293726	14.59532638018391	44.933606191545095	equal-patient mean ordering gain	8	11	95% patient-block percentile bootstrap, pointwise	20000	2026091301	output/stRSC_Source_Data_v1/tsv/F1_score_components.tsv
HisToGene	RNA context + depth matched	Similar cell mix	0.06193146454405698	0.02996059295614131	0.09410527592909079	6.1931464544056976	2.9960592956141308	9.41052759290908	equal-patient mean ordering gain	8	11	95% patient-block percentile bootstrap, pointwise	20000	2026091301	output/stRSC_Source_Data_v1/tsv/F1_score_components.tsv
HisToGene	Within regions	Within regions	0.04158845186363156	0.013372493986320995	0.07516501995905646	4.158845186363156	1.3372493986320995	7.516501995905646	equal-patient mean ordering gain	8	11	95% patient-block percentile bootstrap, pointwise	20000	2026091301	output/stRSC_Source_Data_v1/tsv/F1_score_components.tsv
Hist2ST	Across regions	Across regions	0.35811701136468926	0.23289463071964972	0.4929886564020535	35.811701136468926	23.28946307196497	49.29886564020535	equal-patient mean ordering gain	8	11	95% patient-block percentile bootstrap, pointwise	20000	2026091301	output/stRSC_Source_Data_v1/tsv/F1_score_components.tsv
Hist2ST	RNA context + depth matched	Similar cell mix	0.07331209474816722	0.0161631908849022	0.13363705008687954	7.331209474816721	1.61631908849022	13.363705008687955	equal-patient mean ordering gain	8	11	95% patient-block percentile bootstrap, pointwise	20000	2026091301	output/stRSC_Source_Data_v1/tsv/F1_score_components.tsv
Hist2ST	Within regions	Within regions	0.0658870760054012	-0.007873658811370352	0.1328552259658139	6.588707600540119	-0.7873658811370352	13.285522596581389	equal-patient mean ordering gain	8	11	95% patient-block percentile bootstrap, pointwise	20000	2026091301	output/stRSC_Source_Data_v1/tsv/F1_score_components.tsv
ST-Net	Across regions	Across regions	0.3076856627704766	0.14656133509647834	0.46067153404179517	30.76856627704766	14.656133509647834	46.067153404179514	equal-patient mean ordering gain	8	11	95% patient-block percentile bootstrap, pointwise	20000	2026091301	output/stRSC_Source_Data_v1/tsv/F1_score_components.tsv
ST-Net	RNA context + depth matched	Similar cell mix	0.09070490567295729	0.02800099039256025	0.16290977556196004	9.070490567295728	2.800099039256025	16.290977556196005	equal-patient mean ordering gain	8	11	95% patient-block percentile bootstrap, pointwise	20000	2026091301	output/stRSC_Source_Data_v1/tsv/F1_score_components.tsv
ST-Net	Within regions	Within regions	0.08493676271334796	0.023101536402164752	0.15670250498959945	8.493676271334797	2.310153640216475	15.670250498959945	equal-patient mean ordering gain	8	11	95% patient-block percentile bootstrap, pointwise	20000	2026091301	output/stRSC_Source_Data_v1/tsv/F1_score_components.tsv
TCGN	Across regions	Across regions	0.32783644226501374	0.19561487278919934	0.4455766041862208	32.78364422650137	19.561487278919934	44.55766041862208	equal-patient mean ordering gain	8	11	95% patient-block percentile bootstrap, pointwise	20000	2026091301	output/stRSC_Source_Data_v1/tsv/F1_score_components.tsv
TCGN	RNA context + depth matched	Similar cell mix	0.10795431973369435	0.058206278966997364	0.15974961334112267	10.795431973369435	5.820627896699737	15.974961334112267	equal-patient mean ordering gain	8	11	95% patient-block percentile bootstrap, pointwise	20000	2026091301	output/stRSC_Source_Data_v1/tsv/F1_score_components.tsv
TCGN	Within regions	Within regions	0.09673796842161012	0.054761342855561786	0.13946097327959894	9.673796842161012	5.476134285556179	13.946097327959894	equal-patient mean ordering gain	8	11	95% patient-block percentile bootstrap, pointwise	20000	2026091301	output/stRSC_Source_Data_v1/tsv/F1_score_components.tsv
THItoGene	Across regions	Across regions	0.2738980056765804	0.14205495548811406	0.41368218181896366	27.38980056765804	14.205495548811406	41.36821818189637	equal-patient mean ordering gain	8	11	95% patient-block percentile bootstrap, pointwise	20000	2026091301	output/stRSC_Source_Data_v1/tsv/F1_score_components.tsv
THItoGene	RNA context + depth matched	Similar cell mix	0.07821829012083872	0.027367697910512936	0.12877997817239195	7.821829012083873	2.7367697910512936	12.877997817239194	equal-patient mean ordering gain	8	11	95% patient-block percentile bootstrap, pointwise	20000	2026091301	output/stRSC_Source_Data_v1/tsv/F1_score_components.tsv
THItoGene	Within regions	Within regions	0.07762402363906523	0.028506877221599718	0.1260240185873927	7.762402363906523	2.8506877221599716	12.60240185873927	equal-patient mean ordering gain	8	11	95% patient-block percentile bootstrap, pointwise	20000	2026091301	output/stRSC_Source_Data_v1/tsv/F1_score_components.tsv
iStar	Across regions	Across regions	0.2049290503024787	0.09839994427452425	0.3122294166704451	20.492905030247872	9.839994427452424	31.222941667044513	equal-patient mean ordering gain	8	11	95% patient-block percentile bootstrap, pointwise	20000	2026091301	output/stRSC_Source_Data_v1/tsv/F1_score_components.tsv
iStar	RNA context + depth matched	Similar cell mix	0.06914443231939227	0.031319365918341696	0.11198820598472294	6.914443231939227	3.1319365918341697	11.198820598472293	equal-patient mean ordering gain	8	11	95% patient-block percentile bootstrap, pointwise	20000	2026091301	output/stRSC_Source_Data_v1/tsv/F1_score_components.tsv
iStar	Within regions	Within regions	0.06912019504773412	0.031028900950337664	0.1085625946867794	6.912019504773411	3.1028900950337666	10.856259468677939	equal-patient mean ordering gain	8	11	95% patient-block percentile bootstrap, pointwise	20000	2026091301	output/stRSC_Source_Data_v1/tsv/F1_score_components.tsv
"""
TASK_INTERVALS_TSV = """task	estimate	ci_low	ci_high	estimate_percent	ci_low_percent	ci_high_percent	estimator	n_patients	n_methods	interval	bootstrap_draws	seed	source
Across regions	0.2886734288423047	0.157067984116209	0.4081136159685129	28.867342884230467	15.7067984116209	40.81136159685129	median across fixed methods of equal-patient mean ordering gain	8	11	95% patient-block percentile bootstrap, pointwise	20000	2026091301	output/stRSC_Source_Data_v1/tsv/F1_score_components.tsv
RNA context + depth matched	0.07331209474816722	0.03455300134790459	0.12191122436388135	7.331209474816721	3.4553001347904586	12.191122436388135	median across fixed methods of equal-patient mean ordering gain	8	11	95% patient-block percentile bootstrap, pointwise	20000	2026091301	output/stRSC_Source_Data_v1/tsv/F1_score_components.tsv
Within regions	0.0658870760054012	0.028307291842337677	0.10822713380105407	6.588707600540119	2.8307291842337676	10.822713380105407	median across fixed methods of equal-patient mean ordering gain	8	11	95% patient-block percentile bootstrap, pointwise	20000	2026091301	output/stRSC_Source_Data_v1/tsv/F1_score_components.tsv
"""
METHOD_INTERVAL_ROWS = list(csv.DictReader(StringIO(METHOD_INTERVALS_TSV), delimiter="\t"))
TASK_INTERVAL_ROWS = list(csv.DictReader(StringIO(TASK_INTERVALS_TSV), delimiter="\t"))
METHOD_INTERVALS = {(r["source_task"], r["method"]): r for r in METHOD_INTERVAL_ROWS}
TASK_INTERVALS = {r["task"]: r for r in TASK_INTERVAL_ROWS}

# Selected complete records from current F1abc_task_accuracy.tsv (7 Sep 2026).
# Patient values and rounded means remain fixed; new supplied CIs are embedded above.
PATIENT_VALUES = {
    "Across regions": {
        "EGNv2": [
            0.038871,
            0.710455,
            0.236003,
            0.552385,
            0.225718,
            0.093057,
            0.515274,
            0.616372
        ],
        "Hist2ST": [
            0.199276,
            0.490022,
            0.224584,
            0.656539,
            0.24697,
            0.123546,
            0.306361,
            0.617639
        ],
        "ST-Net": [
            0.129515,
            0.639655,
            0.488311,
            0.448593,
            -0.032149,
            0.005713,
            0.429984,
            0.351863
        ],
        "DeepPT": [
            0.09104,
            0.532393,
            0.346922,
            0.333326,
            -0.032092,
            0.071016,
            0.306894,
            0.559623
        ],
        "TCGN": [
            0.127711,
            0.436746,
            0.524219,
            0.445014,
            0.066841,
            0.105346,
            0.498273,
            0.41854
        ],
        "HisToGene": [
            0.130826,
            0.360576,
            0.430923,
            0.506196,
            0.023062,
            -0.058022,
            0.501832,
            0.534234
        ],
        "DeepSpaCE": [
            0.146629,
            0.450466,
            0.462345,
            0.431799,
            0.092512,
            -0.015917,
            0.512249,
            0.229305
        ],
        "THItoGene": [
            0.033735,
            0.395602,
            0.124542,
            0.560093,
            0.183192,
            0.072059,
            0.254325,
            0.567636
        ],
        "EGNv1": [
            0.060966,
            0.480696,
            0.143263,
            0.097331,
            0.053249,
            0.087549,
            0.373673,
            0.391351
        ],
        "GeneCodeR": [
            0.071093,
            0.345785,
            0.221722,
            0.292626,
            0.031837,
            -0.005403,
            0.402403,
            0.520831
        ],
        "iStar": [
            0.071446,
            0.460879,
            0.154474,
            0.338234,
            -0.007727,
            0.029269,
            0.284548,
            0.308309
        ]
    },
    "Similar cell mix": {
        "EGNv2": [
            -0.040053,
            0.199939,
            -0.032969,
            0.090726,
            -0.003231,
            0.041952,
            0.0891,
            0.161152
        ],
        "Hist2ST": [
            -0.000569,
            0.212961,
            0.138274,
            0.065582,
            0.008276,
            0.028646,
            -0.038199,
            0.171524
        ],
        "ST-Net": [
            -0.019525,
            0.301167,
            0.09139,
            0.164996,
            0.048309,
            -0.021569,
            0.068081,
            0.09279
        ],
        "DeepPT": [
            -0.010662,
            0.093927,
            0.09049,
            0.166485,
            -0.00434,
            0.036873,
            0.06454,
            0.191659
        ],
        "TCGN": [
            0.019227,
            0.217511,
            0.097626,
            0.192975,
            0.037058,
            0.040125,
            0.073722,
            0.185391
        ],
        "HisToGene": [
            0.017974,
            0.082898,
            0.024791,
            0.098367,
            0.130462,
            -0.002114,
            0.030448,
            0.112625
        ],
        "DeepSpaCE": [
            -0.05347,
            0.261197,
            0.161075,
            0.15499,
            0.057664,
            0.026888,
            0.189643,
            0.124041
        ],
        "THItoGene": [
            -0.025275,
            0.200162,
            0.088583,
            0.111741,
            0.099737,
            0.012162,
            -0.005209,
            0.143844
        ],
        "EGNv1": [
            -0.005623,
            0.247592,
            0.035058,
            0.084011,
            0.02881,
            -0.005634,
            0.030583,
            0.073846
        ],
        "GeneCodeR": [
            -0.029996,
            0.181345,
            0.028264,
            0.031543,
            0.056819,
            0.087192,
            -0.005843,
            0.081412
        ],
        "iStar": [
            0.025345,
            0.19277,
            0.064838,
            0.103566,
            -0.010481,
            0.023411,
            0.090831,
            0.062876
        ]
    },
    "Within regions": {
        "EGNv2": [
            -0.040353,
            0.159786,
            0.028458,
            0.044275,
            0.11994,
            0.000727,
            0.035317,
            0.06672
        ],
        "Hist2ST": [
            0.003906,
            0.1868,
            0.186699,
            -0.005999,
            0.041617,
            0.046008,
            -0.115076,
            0.183141
        ],
        "ST-Net": [
            -0.023534,
            0.28345,
            0.159575,
            0.13259,
            0.035322,
            -0.023363,
            0.040082,
            0.075372
        ],
        "DeepPT": [
            -0.029706,
            0.032952,
            0.131859,
            0.151596,
            -0.010765,
            0.023971,
            0.021635,
            0.071536
        ],
        "TCGN": [
            0.007987,
            0.206633,
            0.143185,
            0.119679,
            0.076457,
            0.061526,
            0.028466,
            0.129969
        ],
        "HisToGene": [
            0.01812,
            0.04446,
            0.078438,
            0.032285,
            0.138212,
            -0.020054,
            0.015014,
            0.026233
        ],
        "DeepSpaCE": [
            -0.058739,
            0.273322,
            0.193154,
            0.140928,
            0.10501,
            0.001447,
            0.172876,
            0.099574
        ],
        "THItoGene": [
            -0.034299,
            0.194741,
            0.139649,
            0.067135,
            0.056898,
            0.048295,
            0.008933,
            0.13964
        ],
        "EGNv1": [
            -0.004003,
            0.225224,
            0.060261,
            -0.006672,
            0.043217,
            0.003905,
            -0.000277,
            0.009709
        ],
        "GeneCodeR": [
            -0.030623,
            0.139823,
            0.059905,
            0.02165,
            0.083175,
            0.063847,
            0.041064,
            0.009901
        ],
        "iStar": [
            0.026725,
            0.171268,
            0.111126,
            0.064719,
            -0.014868,
            0.021213,
            0.06593,
            0.10685
        ]
    }
}

SUMMARY = {
    "Across regions": {
        "EGNv2": [
            0.373517
        ],
        "Hist2ST": [
            0.358117
        ],
        "ST-Net": [
            0.307686
        ],
        "DeepPT": [
            0.27614
        ],
        "TCGN": [
            0.327836
        ],
        "HisToGene": [
            0.303703
        ],
        "DeepSpaCE": [
            0.288673
        ],
        "THItoGene": [
            0.273898
        ],
        "EGNv1": [
            0.21101
        ],
        "GeneCodeR": [
            0.235112
        ],
        "iStar": [
            0.204929
        ]
    },
    "Similar cell mix": {
        "EGNv2": [
            0.063327
        ],
        "Hist2ST": [
            0.073312
        ],
        "ST-Net": [
            0.090705
        ],
        "DeepPT": [
            0.078622
        ],
        "TCGN": [
            0.107954
        ],
        "HisToGene": [
            0.061931
        ],
        "DeepSpaCE": [
            0.115254
        ],
        "THItoGene": [
            0.078218
        ],
        "EGNv1": [
            0.06108
        ],
        "GeneCodeR": [
            0.053842
        ],
        "iStar": [
            0.069144
        ]
    },
    "Within regions": {
        "EGNv2": [
            0.051859
        ],
        "Hist2ST": [
            0.065887
        ],
        "ST-Net": [
            0.084937
        ],
        "DeepPT": [
            0.049135
        ],
        "TCGN": [
            0.096738
        ],
        "HisToGene": [
            0.041588
        ],
        "DeepSpaCE": [
            0.115946
        ],
        "THItoGene": [
            0.077624
        ],
        "EGNv1": [
            0.04142
        ],
        "GeneCodeR": [
            0.048593
        ],
        "iStar": [
            0.06912
        ]
    }
}

MEDIANS = {"Across regions": 0.289, "Similar cell mix": 0.073, "Within regions": 0.066}

# Overall Pearson rank and the task-specific ranks are method-level quantities.
# They are never expanded into patient-level pseudo-observations.
OVERALL_RANK = {method: rank for rank, method in enumerate(METHODS, start=1)}
TASK_RANK = {
    "Across regions": {
        "EGNv2": 1, "Hist2ST": 2, "ST-Net": 4, "DeepPT": 7, "TCGN": 3,
        "HisToGene": 5, "DeepSpaCE": 6, "THItoGene": 8, "EGNv1": 10,
        "GeneCodeR": 9, "iStar": 11,
    },
    "Similar cell mix": {
        "EGNv2": 8, "Hist2ST": 6, "ST-Net": 3, "DeepPT": 4, "TCGN": 2,
        "HisToGene": 9, "DeepSpaCE": 1, "THItoGene": 5, "EGNv1": 10,
        "GeneCodeR": 11, "iStar": 7,
    },
    "Within regions": {
        "EGNv2": 7, "Hist2ST": 6, "ST-Net": 3, "DeepPT": 8, "TCGN": 2,
        "HisToGene": 10, "DeepSpaCE": 1, "THItoGene": 4, "EGNv1": 11,
        "GeneCodeR": 9, "iStar": 5,
    },
}
EXPECTED_TAU = {
    "Across regions": 0.8181818181818182,
    "Similar cell mix": 0.2,
    "Within regions": 0.09090909090909091,
}
# === END DATA SECTOR ===


# COL_BG = #ffffff; L1-PIL: sampled white reference background at (980,650).
COL_BG = "#ffffff"
# COL_GREEN = #1f5a58; explicit v68 palette contract: deep petrol patient fill.
COL_GREEN = "#1F5A58"
# COL_BLUE = #365f78; explicit v68 palette contract: muted Prussian-blue patient fill.
COL_BLUE = "#365F78"
# COL_RED = #9b5e4c; explicit v68 palette contract: terracotta patient fill.
COL_RED = "#9B5E4C"
# COL_TEXT = #252b2d; explicit v68 palette contract: dark older-print body ink.
COL_TEXT = "#252B2D"
# COL_NEUTRAL = #85827d; explicit v68 palette contract: ordinary warm graphite.
COL_NEUTRAL = "#85827D"
# COL_SUMMARY = #aaa39b; explicit v68 palette contract: secondary warm-grey marks.
COL_SUMMARY = "#AAA39B"
# COL_SPINE = #454b4d; explicit v68 palette contract: thin dark-grey axes/spines.
COL_SPINE = "#454B4D"
# COL_GRID: author-authorized softer hierarchy; L2 warm mid-light grey.
# Kept in the measured warm-neutral family while one value-step lighter.
COL_GRID = "#E4E0DB"
# COL_ZERO = #85827d; explicit v68 palette contract: warm-graphite reference rule.
COL_ZERO = "#85827D"
# COL_EGN = #365f78; explicit v68 palette contract: principal muted Prussian blue.
COL_EGN = "#365F78"
# COL_DEEPSPACE = #9b5e4c; explicit v68 palette contract: terracotta task accent.
COL_DEEPSPACE = "#9B5E4C"

ACCENT = {"EGNv2": COL_EGN, "DeepSpaCE": COL_DEEPSPACE}

# v97 bounded hierarchy: mean ticks gain weight/length; patient style is untouched.
# L2: muted palette extension by alpha; L1 patient hues and shapes preserved.
PATIENT_FILLED_SIZE = 7.0
PATIENT_OPEN_SIZE = 9.0
PATIENT_FILLED_EDGE = 0.20
PATIENT_OPEN_EDGE = 0.42
PATIENT_ZORDER = 4
SUMMARY_LINEWIDTH = 1.65
SUMMARY_MARKER_SIZE = 9.0
SUMMARY_ZORDER = 6
# L2 visible-but-recessive hairline addition; L1 existing mean hues are reused.
MEAN_CI_LINEWIDTH = 0.60
MEAN_CI_CAP_SIZE = 3.5
MEAN_CI_ZORDER = 3
CONTROL_RANK_SIZE = 14.0
PRIMARY_RANK_NEUTRAL_SIZE = 18.0
PRIMARY_RANK_ACCENT_SIZE = 20.0
MEDIAN_FONT_SIZE = 7.0
MEDIAN_FONT_WEIGHT = "normal"
MEDIAN_HA = "right"
MEDIAN_VA = "center"
MEDIAN_ZORDER = 30
METRIC_LABEL = "Ordering gain\n(% of RNA-reference gain)"
# Display-only amendment; frozen data keys and source values above are unchanged.
TASK_DISPLAY = {task: task for task in TASKS}
TASK_DISPLAY["Similar cell mix"] = "RNA context + depth matched"

# Lower rank-row typography: a deliberate 9/8/7-pt hierarchy, with fixed
# horizontal header bands and separate tick/axis-label registers. Scientific
# encodings and the accepted upper-panel block stay fixed.
RANK_TITLE_FONT_SIZE = 8.0
RANK_TAU_FONT_SIZE = 7.2
RANK_TICK_FONT_SIZE = 7.0
RANK_AXIS_FONT_SIZE = 7.4
RANK_ANNOT_FONT_SIZE = 7.0
RANK_HEADER_OFFSET_IN = 0.155
RANK_LETTER_BAND_IN = 0.160
RANK_TAU_BAND_IN = 0.500
RANK_HEADER_EDGE_INSET_IN = 0.040
RANK_HEADER_MIN_GUTTER_PX = 24.0
RANK_HEADER_TICK_MIN_GAP_PX = 8.0
RANK_TICK_AXIS_MIN_GAP_PX = 18.0
RANK_AXIS_CANVAS_MIN_INSET_PX = 12.0

# Actual patient-block rank statistics are supplied separately. No uncertainty
# is estimated in this drawing source; a missing table stops rendering.
def load_rank_tau_intervals():
    path = Path(__file__).resolve().parents[1] / "inputs" / "TAU_INTERVALS.tsv"
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    aliases = {"RNA context + depth matched": "Similar cell mix"}
    result = {}
    for row in rows:
        task = aliases.get(row["condition"], row["condition"])
        if task not in TASKS:
            raise ValueError("Unexpected tau task: " + str(task))
        estimate, low, high = (float(row[key]) for key in ("tau_b", "ci_low", "ci_high"))
        if not np.isclose(estimate, EXPECTED_TAU[task], atol=1e-12, rtol=0):
            raise ValueError("Recovered point tau differs from frozen Figure 1: " + task)
        if not (-1 <= low <= high <= 1):
            raise ValueError("Invalid supplied tau limits: " + task)
        if task in result:
            raise ValueError("Duplicate tau task: " + task)
        result[task] = (estimate, low, high)
    if set(result) != set(TASKS):
        raise ValueError("Incomplete three-task tau interval table")
    return result

def tau_interval_label(task):
    _, low, high = RANK_TAU_INTERVALS[task]
    return f"τ 95% CI: [{low:.2f}, {high:.2f}]".replace("-", "−")

RANK_TAU_INTERVALS = load_rank_tau_intervals()

# Four fixed open/filled patient pairs. The diamond is the minimal fourth-shape
# extension required by eight patient series; all filled hues stay in the L1 palette.
PATIENT_STYLE = {
    "A": {"marker": "o", "filled": False, "color": COL_TEXT, "offset": -0.325},
    "B": {"marker": "o", "filled": True, "color": COL_GREEN, "offset": -0.275},
    "C": {"marker": "s", "filled": False, "color": COL_TEXT, "offset": -0.125},
    "D": {"marker": "s", "filled": True, "color": COL_RED, "offset": -0.075},
    "E": {"marker": "^", "filled": False, "color": COL_TEXT, "offset": 0.075},
    "F": {"marker": "^", "filled": True, "color": COL_BLUE, "offset": 0.125},
    "G": {"marker": "D", "filled": False, "color": COL_TEXT, "offset": 0.275},
    "H": {"marker": "D", "filled": True, "color": COL_GREEN, "offset": 0.325},
}

OUT_DIR = Path(__file__).resolve().parent
PNG_PATH = OUT_DIR / "img_iter0.png"
PDF_PATH = OUT_DIR / "figure_iter0.pdf"
CHECK_PATH = OUT_DIR / "floor_selfcheck_iter0.txt"
INTERVAL_PATH = OUT_DIR / "intervals_plotted.tsv"
PDF_METADATA = {
    "Creator": "FigMirror",
    "Producer": "Matplotlib",
    "CreationDate": None,
    "ModDate": None,
}

# Sep5 scientific directive: identical real y limits across a-c, containing
# every retained patient observation and mean.
COMMON_YLIM = (-0.15, 0.75)
COMMON_YTICKS = [0.0, 0.2, 0.4, 0.6]
PANEL_Y_CONFIG = {
    task: {"ylim": COMMON_YLIM, "ticks": COMMON_YTICKS} for task in TASKS
}


def validate_data():
    assert TASKS == list(PATIENT_VALUES)
    assert all(list(PATIENT_VALUES[t]) == METHODS for t in TASKS)
    assert all(list(SUMMARY[t]) == METHODS for t in TASKS)
    assert sum(len(PATIENT_VALUES[t][m]) for t in TASKS for m in METHODS) == 264
    assert all(len(PATIENT_VALUES[t][m]) == 8 for t in TASKS for m in METHODS)
    assert sum(len(SUMMARY[t]) for t in TASKS) == 33
    assert list(PATIENT_STYLE) == PATIENTS
    assert [PATIENT_STYLE[p]["marker"] for p in PATIENTS] == ["o", "o", "s", "s", "^", "^", "D", "D"]
    assert [PATIENT_STYLE[p]["filled"] for p in PATIENTS] == [False, True] * 4
    assert len(METHOD_INTERVAL_ROWS) == len(METHOD_INTERVALS) == 33
    assert set(METHOD_INTERVALS) == {(task, method) for task in TASKS for method in METHODS}
    assert len(TASK_INTERVAL_ROWS) == len(TASK_INTERVALS) == 3
    assert set(TASK_INTERVALS) == set(TASK_DISPLAY.values())
    for task in TASKS:
        for method in METHODS:
            row = METHOD_INTERVALS[(task, method)]
            assert float(f"{float(row['estimate']):.6f}") == SUMMARY[task][method][0]
            assert float(row["ci_low"]) <= float(row["estimate"]) <= float(row["ci_high"])
            assert COMMON_YLIM[0] <= float(row["ci_low"]) <= float(row["ci_high"]) <= COMMON_YLIM[1]


def median_interval_label(task):
    row = TASK_INTERVALS[TASK_DISPLAY[task]]
    return (f"median = {MEDIANS[task]:.1%} "
            f"[{float(row['ci_low_percent']):.1f}, {float(row['ci_high_percent']):.1f}]")


def write_interval_manifest():
    """Serialize the supplied values and drawing coordinates; compute no statistics."""
    fields = ["source_file", "source_key", "interval_kind", "estimate", "ci_low",
              "ci_high", "displayed_unit_factor", "panel", "axis_min", "axis_max",
              "plotted_estimate", "raw_unit", "rendering"]
    with INTERVAL_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for panel, task in zip("abc", TASKS):
            for method in METHODS:
                row = METHOD_INTERVALS[(task, method)]
                writer.writerow({
                    "source_file": "inputs/statistics/HER2ST_METHOD_INTERVALS.tsv",
                    "source_key": f"source_task={row['source_task']};method={row['method']}",
                    "interval_kind": "method_mean",
                    "estimate": row["estimate"], "ci_low": row["ci_low"], "ci_high": row["ci_high"],
                    "displayed_unit_factor": "100", "panel": panel,
                    "axis_min": "-0.15", "axis_max": "0.75",
                    "plotted_estimate": str(SUMMARY[task][method][0]),
                    "raw_unit": "normalized_ordering_gain", "rendering": "capped_vertical_whisker",
                })
            row = TASK_INTERVALS[TASK_DISPLAY[task]]
            writer.writerow({
                "source_file": "inputs/statistics/HER2ST_TASK_INTERVALS.tsv",
                "source_key": f"task={row['task']}", "interval_kind": "task_median",
                "estimate": row["estimate"], "ci_low": row["ci_low"], "ci_high": row["ci_high"],
                "displayed_unit_factor": "100", "panel": panel,
                "axis_min": "-0.15", "axis_max": "0.75", "plotted_estimate": str(MEDIANS[task]),
                "raw_unit": "normalized_ordering_gain", "rendering": "header_annotation",
            })


def kendall_tau_unique(x, y):
    concordant = 0
    discordant = 0
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            product = (x[j] - x[i]) * (y[j] - y[i])
            concordant += int(product > 0)
            discordant += int(product < 0)
    tau = (concordant - discordant) / (concordant + discordant)
    return tau, concordant, discordant


def make_figure():
    # Compact 180-mm submission plate. The method register is shared by all
    # three aligned evidence rows, so only the first row repeats method names.
    # This removes redundant label bands rather than shrinking any typography.
    fig_width_in = 7.0866
    fig_height_in = 151.0 / 25.4
    left, right = 0.108, 0.985
    rank_gutter = 0.024
    rank_width = ((right - left) - 2.0 * rank_gutter) / 3.0
    rank_target_aspect = 1.85
    rank_height_in = (rank_width * fig_width_in) / rank_target_aspect

    fig = plt.figure(figsize=(fig_width_in, fig_height_in), dpi=300, facecolor=COL_BG)

    width = right - left
    detail_height = 0.118
    detail_bottoms = [0.748, 0.570, 0.392]
    detail_header_y = [0.970, 0.714, 0.536]
    detail_letters = ["a", "b", "c"]
    x = np.arange(len(METHODS), dtype=float)

    detail_axes = []
    patient_handles = []
    summary_artists = []
    interval_artists = []
    for row_i, (task, bottom, hy, letter) in enumerate(
        zip(TASKS, detail_bottoms, detail_header_y, detail_letters)
    ):
        ax = fig.add_axes([left, bottom, width, detail_height], facecolor=COL_BG)
        detail_axes.append(ax)
        ax.set_axisbelow(True)
        ax.set_xlim(-0.55, 10.55)
        ax.set_ylim(*PANEL_Y_CONFIG[task]["ylim"])
        ax.set_yticks(PANEL_Y_CONFIG[task]["ticks"])
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
        ax.grid(axis="y", which="major", color=COL_GRID, linewidth=0.42, alpha=1.0, zorder=0)
        ax.axhline(0.0, color=COL_ZERO, linewidth=0.62, linestyle=(0, (2.0, 2.0)), zorder=1)

        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(COL_SPINE)
            ax.spines[side].set_linewidth(0.60)

        ax.set_xticks(x)
        if row_i == 0:
            ax.set_xticklabels(
                METHODS,
                rotation=48,
                ha="left",
                va="bottom",
                rotation_mode="anchor",
                fontsize=7.0,
                color=COL_TEXT,
            )
            ax.xaxis.set_ticks_position("top")
            ax.tick_params(
                axis="x",
                which="both",
                top=True,
                bottom=False,
                labeltop=True,
                labelbottom=False,
                length=0,
                pad=2.0,
                colors=COL_TEXT,
            )
        else:
            ax.set_xticklabels([])
            ax.tick_params(axis="x", which="both", top=False, bottom=False, length=0)
        ax.tick_params(axis="y", which="both", length=0, pad=3.2, labelsize=7.0, colors=COL_TEXT)

        # Uniform left alignment keeps GeneCodeR and iStar in separate text boxes.
        xlabels = ax.get_xticklabels()
        for tick_text in xlabels + ax.get_yticklabels():
            tick_text.set_zorder(20)

        # Each patient remains an explicit, fixed marker series in every band.
        # v69 lowers raw observations one visual level through modest size and
        # edge restraint; shape, fill state, hue and position stay unchanged.
        for patient_i, patient in enumerate(PATIENTS):
            style = PATIENT_STYLE[patient]
            ys = np.asarray([PATIENT_VALUES[task][method][patient_i] for method in METHODS], dtype=float)
            if style["filled"]:
                ax.scatter(
                    x + style["offset"],
                    ys,
                    s=PATIENT_FILLED_SIZE,
                    marker=style["marker"],
                    facecolor=style["color"],
                    edgecolor=style["color"],
                    linewidth=PATIENT_FILLED_EDGE,
                    alpha=0.55,
                    zorder=PATIENT_ZORDER,
                    clip_on=True,
                    gid=f"patient_{row_i}_{patient}",
                )
            else:
                ax.scatter(
                    x + style["offset"],
                    ys,
                    s=PATIENT_OPEN_SIZE,
                    marker=style["marker"],
                    facecolor=COL_BG,
                    edgecolor=COL_TEXT,
                    linewidth=PATIENT_OPEN_EDGE,
                    alpha=0.55,
                    zorder=PATIENT_ZORDER,
                    clip_on=True,
                    gid=f"patient_{row_i}_{patient}",
                )

        # Patient-equal means gain explicit visual priority; no data position changes.
        # New supplied patient-block mean intervals do not move any frozen mean.
        for method_i, method in enumerate(METHODS):
            mean = SUMMARY[task][method][0]
            mean_color = ACCENT.get(method, COL_SPINE)
            ci = METHOD_INTERVALS[(task, method)]
            low, high = float(ci["ci_low"]), float(ci["ci_high"])
            # Direct endpoint coordinates avoid multiplying the percent-formatted axis twice.
            ci_stem = ax.vlines(
                method_i, low, high, colors=mean_color, linewidth=MEAN_CI_LINEWIDTH,
                zorder=MEAN_CI_ZORDER, clip_on=True, gid=f"method_ci_{row_i}_{method}",
            )
            ci_caps, = ax.plot(
                [method_i, method_i], [low, high], linestyle="None", marker="_",
                color=mean_color, markersize=MEAN_CI_CAP_SIZE,
                markeredgewidth=MEAN_CI_LINEWIDTH, zorder=MEAN_CI_ZORDER, clip_on=True,
                gid=f"method_ci_caps_{row_i}_{method}",
            )
            interval_artists.append((task, method, low, high, ci_stem, ci_caps))
            summary_line, = ax.plot(
                [method_i], [mean], linestyle="None", marker="_",
                color=mean_color, markersize=SUMMARY_MARKER_SIZE,
                markeredgewidth=SUMMARY_LINEWIDTH, alpha=1.0, zorder=SUMMARY_ZORDER, clip_on=True,
            )
            summary_artists.append((row_i, method, mean, summary_line))

        panel_letter = fig.text(
            0.047,
            hy,
            letter,
            ha="left",
            va="center",
            fontsize=9.0,
            fontweight="bold",
            color=COL_TEXT,
            zorder=30,
        )
        panel_letter.set_gid(f"panel_letter_{letter}")
        row_header = fig.text(
            left,
            hy,
            TASK_DISPLAY[task],
            ha="left",
            va="center",
            fontsize=7.5,
            fontweight="normal",
            color=COL_TEXT,
            zorder=30,
        )
        row_header.set_gid(f"row_header_{row_i}")
        median_header = fig.text(
            right,
            hy,
            median_interval_label(task),
            ha=MEDIAN_HA,
            va=MEDIAN_VA,
            fontsize=MEDIAN_FONT_SIZE,
            fontweight=MEDIAN_FONT_WEIGHT,
            color=COL_TEXT,
            zorder=MEDIAN_ZORDER,
        )
        median_header.set_gid(f"median_header_{row_i}")

    # One compact patient key for the complete three-row evidence block.
    for patient in PATIENTS:
        style = PATIENT_STYLE[patient]
        patient_handles.append(
            Line2D(
                [0], [0], linestyle="None", marker=style["marker"],
                markersize=3.0 if not style["filled"] else 2.65,
                markerfacecolor=style["color"] if style["filled"] else COL_BG,
                markeredgecolor=style["color"] if style["filled"] else COL_TEXT,
                markeredgewidth=PATIENT_OPEN_EDGE if not style["filled"] else PATIENT_FILLED_EDGE,
            )
        )
    # The original patient key stays fixed; the caption defines the new mean CIs.
    legend = detail_axes[-1].legend(
        patient_handles,
        PATIENTS,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.070),
        ncol=8,
        frameon=False,
        fontsize=7.0,
        handlelength=0.72,
        handletextpad=0.22,
        columnspacing=0.76,
        borderpad=0.0,
        labelspacing=0.0,
    )
    legend.set_zorder(10)
    for text in legend.get_texts():
        text.set_color(COL_TEXT)
        text.set_zorder(11)

    shared_detail_y = fig.text(
        0.019,
        0.626,
        METRIC_LABEL,
        rotation=90,
        ha="center",
        va="center",
        fontsize=7.0,
        color=COL_TEXT,
        zorder=30,
    )
    shared_detail_y.set_gid("shared_detail_ylabel")

    # Three compact Tian-style rank facets, identical scales and mark grammar.
    # Their outer edges exactly reuse the detailed-row anchors; equal tight
    # gutters leave a rendered data-area W/H of 1.45 (user target 1.35--1.55).
    rank_lefts = [
        left,
        left + rank_width + rank_gutter,
        left + 2.0 * (rank_width + rank_gutter),
    ]
    rank_bottom = 0.103
    rank_height = rank_height_in / fig_height_in
    rank_axes = []
    rank_scatters = []
    rank_letters = ["d", "e", "f"]
    for facet_i, (task, facet_left, letter) in enumerate(zip(TASKS, rank_lefts, rank_letters)):
        ax = fig.add_axes([facet_left, rank_bottom, rank_width, rank_height], facecolor=COL_BG)
        rank_axes.append(ax)
        ax.set_axisbelow(True)
        ax.set_xlim(0.5, 11.5)
        ax.set_ylim(0.5, 11.5)
        ax.set_xticks([1, 3, 5, 7, 9, 11])
        ax.set_yticks([1, 3, 5, 7, 9, 11])
        ax.grid(axis="both", which="major", color=COL_GRID, linewidth=0.42, alpha=1.0, zorder=0)
        ax.plot([1, 11], [1, 11], color=COL_ZERO, linewidth=0.55,
                linestyle=(0, (2.4, 2.4)), zorder=1, clip_on=True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(COL_SPINE)
            ax.spines[side].set_linewidth(0.60)
        ax.tick_params(axis="both", which="both", length=0, pad=2.6,
                       labelsize=RANK_TICK_FONT_SIZE, colors=COL_TEXT)
        for tick_text in ax.get_xticklabels() + ax.get_yticklabels():
            tick_text.set_zorder(20)

        xs = np.asarray([OVERALL_RANK[m] for m in METHODS], dtype=float)
        ys = np.asarray([TASK_RANK[task][m] for m in METHODS], dtype=float)
        is_control = task == "Similar cell mix"
        # The intermediate control stays deliberately quiet through its marks:
        # one warm-grey field, slightly smaller points and no method label.
        # Heading typography remains uniform across all three facets.
        colors = (
            [COL_SUMMARY for _ in METHODS]
            if is_control
            else [ACCENT.get(m, COL_NEUTRAL) for m in METHODS]
        )
        sizes = (
            [CONTROL_RANK_SIZE for _ in METHODS]
            if is_control
            else [PRIMARY_RANK_ACCENT_SIZE if m in ACCENT else PRIMARY_RANK_NEUTRAL_SIZE for m in METHODS]
        )
        sc = ax.scatter(xs, ys, s=sizes, c=colors, marker="o",
                        edgecolors=COL_BG, linewidths=0.45 if is_control else 0.55, zorder=4,
                        gid=f"rank_points_{facet_i}")
        rank_scatters.append(sc)

        # Each facet owns three fixed header bands: letter, short title and tau.
        # Keeping every band inside the facet edges makes the visible inter-facet
        # gutter authoritative instead of letting headings borrow adjacent space.
        title_y = rank_bottom + rank_height + RANK_HEADER_OFFSET_IN / fig_height_in
        letter_artist = fig.text(
            facet_left, title_y, letter, ha="left", va="center",
            fontsize=9.0, fontweight="bold", color=COL_TEXT, zorder=30,
        )
        letter_artist.set_gid(f"panel_letter_{letter}")
        title_artist = fig.text(
            facet_left + RANK_LETTER_BAND_IN / fig_width_in,
            title_y, TASK_DISPLAY[task].replace(" + depth", " +\ndepth"), ha="left", va="center",
            fontsize=RANK_TITLE_FONT_SIZE,
            fontweight="normal",
            color=COL_TEXT,
            zorder=30,
        )
        title_artist.set_gid(f"rank_title_{facet_i}")
        tau_value = EXPECTED_TAU[task]  # Frozen tau; no statistics recomputed.
        tau_artist = fig.text(
            facet_left + rank_width - RANK_HEADER_EDGE_INSET_IN / fig_width_in,
            title_y,
            f"τ = {tau_value:.3f}",
            ha="right", va="center", fontsize=RANK_TAU_FONT_SIZE,
            fontweight="normal",
            color=COL_TEXT,
            zorder=30,
        )
        tau_artist.set_gid(f"rank_tau_{facet_i}")

        # Existing empty strip between rank ticks and the common x-axis title.
        ci_artist = fig.text(
            facet_left + rank_width / 2.0, 0.054,
            tau_interval_label(task), ha="center", va="center",
            fontsize=7.0, color=COL_TEXT, zorder=30,
        )
        ci_artist.set_gid(f"rank_tau_ci_{facet_i}")

        if task == "Within regions":
            label_egn = ax.annotate(
                "EGNv2", xy=(1, 7), xytext=(6, 8), textcoords="offset points",
                ha="left", va="bottom", fontsize=RANK_ANNOT_FONT_SIZE, fontweight="normal",
                color=COL_EGN, zorder=25,
                bbox=dict(facecolor=COL_BG, edgecolor="none", alpha=1.0, pad=0.45),
            )
            label_egn.set_gid("rank_label_egnv2")
            label_deep = ax.annotate(
                "DeepSpaCE", xy=(7, 1), xytext=(5, 7), textcoords="offset points",
                ha="left", va="bottom", fontsize=RANK_ANNOT_FONT_SIZE, fontweight="normal",
                color=COL_DEEPSPACE, zorder=25,
                bbox=dict(facecolor=COL_BG, edgecolor="none", alpha=1.0, pad=0.45),
            )
            label_deep.set_gid("rank_label_deepspace")

    shared_rank_x = fig.text(
        (left + right) / 2.0,
        0.020,
        "Overall Pearson rank",
        ha="center", va="bottom", fontsize=RANK_AXIS_FONT_SIZE,
        color=COL_TEXT, zorder=30,
    )
    shared_rank_x.set_gid("shared_rank_xlabel")
    shared_rank_y = fig.text(
        0.060, rank_bottom + rank_height / 2.0, "Ordering-gain rank",
        rotation=90, ha="center", va="center", fontsize=RANK_AXIS_FONT_SIZE,
        color=COL_TEXT, zorder=30,
    )
    shared_rank_y.set_gid("shared_rank_ylabel")
    return fig, detail_axes, rank_axes, legend, summary_artists, rank_scatters, interval_artists


def run_floor_checks(fig, detail_axes, rank_axes, legend, summary_artists, rank_scatters, interval_artists,
                     expected_size_inches=(7.0866, 151.0 / 25.4)):
    """Bounded science/layout gate for the supplied-interval revision."""
    from matplotlib.text import Text
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    checks = []
    def check(name, ok, detail):
        checks.append((name, bool(ok), detail))

    check("ordering_gain_percent_display",
          all(isinstance(ax.yaxis.get_major_formatter(), PercentFormatter) for ax in detail_axes)
          and not any(isinstance(ax.yaxis.get_major_formatter(), PercentFormatter) for ax in rank_axes),
          "Only a-c formatting: 0/20/40/60%; source units and rank/tau unchanged")
    check("median_percent_labels",
          [t.get_text() for t in fig.texts if str(t.get_gid()).startswith("median_header_")]
          == ["median = 28.9% [15.7, 40.8]", "median = 7.3% [3.5, 12.2]", "median = 6.6% [2.8, 10.8]"],
          "Three separate supplied task-median CIs; unchanged displayed point estimates")
    check("common_real_y_limits", all(np.allclose(ax.get_ylim(), COMMON_YLIM)
          for ax in detail_axes), f"a-c={COMMON_YLIM}")
    patient_count = 0
    marks_equal = True
    for row, (task, ax) in enumerate(zip(TASKS, detail_axes)):
        for pi, patient in enumerate(PATIENTS):
            col = next(c for c in ax.collections if c.get_gid() == f"patient_{row}_{patient}")
            actual = np.asarray(col.get_offsets())
            expected = np.column_stack([
                np.arange(11) + PATIENT_STYLE[patient]["offset"],
                [PATIENT_VALUES[task][m][pi] for m in METHODS],
            ])
            patient_count += len(actual)
            marks_equal &= np.array_equal(actual, expected)
        values = [v for m in METHODS for v in PATIENT_VALUES[task][m]]
        values += [v for m in METHODS for v in SUMMARY[task][m]]
        check(f"{task}_values_within_limits",
              min(values) > COMMON_YLIM[0] and max(values) < COMMON_YLIM[1],
              f"range={min(values):.6f}..{max(values):.6f}")
    check("patient_values_unchanged", marks_equal and patient_count == 264,
          f"{patient_count} patient marks; 11 methods x 8 patients x 3 tasks")
    summary_ok = len(summary_artists) == 33
    for row, method, mean, line in summary_artists:
        summary_ok &= np.array_equal(line.get_ydata(), [mean])
    check("supplied_means_unchanged", summary_ok, "33 frozen mean tick coordinates; supplied estimates agree at frozen 6-decimal precision")
    intervals_ok = len(interval_artists) == 33
    for task, method, low, high, stem, caps in interval_artists:
        row = METHOD_INTERVALS[(task, method)]
        x_position = METHODS.index(method)
        expected = np.asarray([[x_position, float(row["ci_low"])],
                               [x_position, float(row["ci_high"])]])
        intervals_ok &= np.array_equal(stem.get_segments()[0], expected)
        intervals_ok &= np.array_equal(caps.get_ydata(), [float(row["ci_low"]), float(row["ci_high"])])
        intervals_ok &= np.array_equal(caps.get_xdata(), [x_position, x_position])
        intervals_ok &= COMMON_YLIM[0] <= low <= high <= COMMON_YLIM[1]
    check("supplied_mean_intervals_exact_and_visible", intervals_ok,
          "33 direct raw ci_low/ci_high pairs; no rescaling, clipping or resampling")
    check("interval_layering",
          all(stem.get_zorder() < PATIENT_ZORDER and caps.get_zorder() < PATIENT_ZORDER
              for _, _, _, _, stem, caps in interval_artists)
          and MEAN_CI_LINEWIDTH < SUMMARY_LINEWIDTH,
          "0.60pt CIs below the retained patient symbols and 1.65pt mean ticks")
    check("rank_intervals_absent", all(len(ax.lines) == 1 and not ax.containers for ax in rank_axes),
          "d-f retain only their original dashed identity lines and fixed rank points")

    rank_count = 0
    ranks_equal = True
    for task, col in zip(TASKS, rank_scatters):
        expected = np.asarray([[OVERALL_RANK[m], TASK_RANK[task][m]] for m in METHODS])
        ranks_equal &= np.array_equal(col.get_offsets(), expected)
        rank_count += len(col.get_offsets())
    check("rank_evidence_unchanged", ranks_equal and rank_count == 33,
          f"33 method ranks; tau={list(EXPECTED_TAU.values())}")
    ci_texts = [t.get_text() for t in fig.texts if str(t.get_gid()).startswith("rank_tau_ci_")]
    check("three_supplied_tau_interval_labels", ci_texts == [tau_interval_label(task) for task in TASKS],
          "Three actual patient-block tau intervals; original point estimates unchanged")
    check("patient_legend", [t.get_text() for t in legend.get_texts()] == PATIENTS,
          "A-H shapes/fill preserved")
    check("summary_visual_priority",
          all(line.get_markeredgewidth() == SUMMARY_LINEWIDTH and line.get_markersize() == SUMMARY_MARKER_SIZE
              for _, _, _, line in summary_artists) and SUMMARY_LINEWIDTH > PATIENT_OPEN_EDGE and SUMMARY_ZORDER > PATIENT_ZORDER,
          "9pt x 1.65pt mean ticks; unchanged patient area 7/9pt2 and alpha 0.55")

    texts = [t for t in fig.findobj(match=Text) if t.get_visible() and t.get_text().strip()]
    boxes = [(t, t.get_window_extent(renderer)) for t in texts]
    boxes = [(t, b) for t, b in boxes if b.width > 0 and b.height > 0]
    clipped = [t.get_text() for t, b in boxes
               if b.x0 < -0.75 or b.y0 < -0.75
               or b.x1 > fig.bbox.width + 0.75 or b.y1 > fig.bbox.height + 0.75]
    overlap = [f"{ta.get_text()} / {tb.get_text()}"
               for i, (ta, ba) in enumerate(boxes) for tb, bb in boxes[i+1:]
               if ba.overlaps(bb)]
    check("minimum_font_7pt", min(t.get_fontsize() for t in texts) >= 7, "All visible source text >=7pt")
    check("text_canvas_bounds", not clipped, repr(clipped))
    check("visible_text_disjoint", not overlap, repr(overlap))
    check("text_obscured_by_marks",
          all(t.get_zorder() > SUMMARY_ZORDER for ax in detail_axes
              for t in ax.get_xticklabels() + ax.get_yticklabels() if t.get_visible()),
          "tick text above mean and patient layers; annotations have white backing")
    check("pdf_fonttype_42", plt.rcParams["pdf.fonttype"] == 42, "TrueType")
    check("page_size_matches_design",
          np.array_equal(fig.get_size_inches(), expected_size_inches),
          f"{expected_size_inches[0] * 25.4:.3f} x {expected_size_inches[1] * 25.4:.3f} mm; no tight crop or page rescaling")
    check("font_family_preserved",
          plt.rcParams["font.sans-serif"] == ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
          "Existing Arial-first sans-serif family")
    write_interval_manifest()
    fig.savefig(PNG_PATH, dpi=300, facecolor=COL_BG, bbox_inches=None, pad_inches=0)
    fig.savefig(PDF_PATH, facecolor=COL_BG, bbox_inches=None, pad_inches=0,
                metadata=PDF_METADATA)
    overall = all(ok for _, ok, _ in checks)
    CHECK_PATH.write_text("\n".join(
        f"{name}: {'PASS' if ok else 'FAIL'} | {detail}" for name, ok, detail in checks
    ) + f"\nOVERALL: {'PASS' if overall else 'FAIL'}\n", encoding="utf-8")
    if not overall:
        raise AssertionError("; ".join(name for name, ok, _ in checks if not ok))


def main():
    validate_data()
    fig, detail_axes, rank_axes, legend, summary_artists, rank_scatters, interval_artists = make_figure()
    run_floor_checks(fig, detail_axes, rank_axes, legend, summary_artists, rank_scatters, interval_artists)
    plt.close(fig)


if __name__ == "__main__":
    main()
