# ruff: noqa: E402
import os
import textwrap
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib-cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import nbformat as nbf
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_DIR = ROOT / "data"
ARTIFACT_DIR = ROOT / "artifacts"
ZIP_NAME = "G2505266E_ZHANG_JUNYI_MH6822_Assignment1.zip"
EXECUTED_NOTEBOOK_NAME = "Task3_AI_Credit_Governance_Prototype_executed.ipynb"

APPROVAL_THRESHOLD = 0.20
MANUAL_REVIEW_LOWER = 0.18
MANUAL_REVIEW_UPPER = 0.23

STUDENT = {
    "name": "ZHANG JUNYI",
    "matriculation_id": "G2505266E",
    "email": "JUNYI006@e.ntu.edu.sg",
}

SOURCES = [
    {
        "label": "Citi 2025 Annual Report",
        "url": "https://www.citigroup.com/global/investors/annual-reports-and-proxy-statements/2026/annual-report",
        "note": "Citi describes its mission as responsibly providing financial services that enable growth and economic progress. The URL path includes 2026 because it is the web version for Citi's 2025 Annual Report published in the 2026 reporting cycle.",
    },
    {
        "label": "Citi EMEA locations",
        "url": "https://careers.citigroup.com/locations/emea.html",
        "note": "Citi lists European locations including Warsaw, Budapest, Dublin, London and additional European countries, supporting the choice of Citi as a cross-border US/EU case.",
    },
    {
        "label": "CFPB 2026 Regulation B final rule",
        "url": "https://www.consumerfinance.gov/rules-policy/final-rules/equal-credit-opportunity-act-regulation-b/",
        "note": "The CFPB issued a 2026 final rule amending Regulation B provisions related to disparate impact, discouragement and special purpose credit programs.",
    },
    {
        "label": "Federal Register 2026 Regulation B final rule",
        "url": "https://www.federalregister.gov/documents/2026/04/22/2026-07804/equal-credit-opportunity-act-regulation-b",
        "note": "The Federal Register notice states that the final rule provides that ECOA does not authorize disparate-impact liability, further defines discouragement and adds SPCP conditions.",
    },
    {
        "label": "CFPB guidance on AI credit denials",
        "url": "https://www.consumerfinance.gov/about-us/newsroom/cfpb-issues-guidance-on-credit-denials-by-lenders-using-artificial-intelligence/",
        "note": "The CFPB states that creditors using AI or complex models must give consumers specific and accurate adverse-action reasons.",
    },
    {
        "label": "CFPB circular on black-box credit models",
        "url": "https://www.consumerfinance.gov/about-us/newsroom/cfpb-acts-to-protect-the-public-from-black-box-credit-models-using-complex-algorithms/",
        "note": "The CFPB says complex algorithms do not remove ECOA adverse-action notice duties.",
    },
    {
        "label": "EU AI Act, Regulation (EU) 2024/1689",
        "url": "https://eur-lex.europa.eu/eli/reg/2024/1689/oj",
        "note": "Annex III classifies AI systems used to evaluate creditworthiness or establish credit scores as high-risk, with exceptions such as fraud detection.",
    },
    {
        "label": "EU AI Act Service Desk, Article 10",
        "url": "https://ai-act-service-desk.ec.europa.eu/en/ai-act/article-10",
        "note": "Article 10 maps high-risk AI to training, validation and testing data governance, including data quality, representativeness, bias examination and mitigation.",
    },
    {
        "label": "EU AI Act Service Desk, Article 14",
        "url": "https://ai-act-service-desk.ec.europa.eu/en/ai-act/article-14",
        "note": "Article 14 explains human oversight expectations for high-risk AI systems.",
    },
    {
        "label": "EU AI Act Service Desk, Article 17",
        "url": "https://ai-act-service-desk.ec.europa.eu/en/ai-act/article-17",
        "note": "Article 17 requires a documented quality management system for high-risk AI, including regulatory compliance strategy, validation, data management, risk management and post-market monitoring.",
    },
    {
        "label": "Grand View Research RegTech market summary",
        "url": "https://www.grandviewresearch.com/industry-analysis/regulatory-technology-market",
        "note": "Grand View Research estimated the global RegTech market at USD 24.34 billion in 2025 and projected USD 112.10 billion by 2033, with large enterprises and risk/compliance management as major segments.",
    },
]

MODEL_FEATURES = [
    "age",
    "income",
    "credit_score",
    "debt_to_income",
    "loan_amount",
    "loan_term_months",
    "prior_default",
    "region_risk_score",
    "employment_status",
    "loan_purpose",
]

NUMERIC_FEATURES = [
    "age",
    "income",
    "credit_score",
    "debt_to_income",
    "loan_amount",
    "loan_term_months",
    "prior_default",
    "region_risk_score",
]

CATEGORICAL_FEATURES = ["employment_status", "loan_purpose"]


def ensure_dirs():
    DATA_DIR.mkdir(exist_ok=True)
    ARTIFACT_DIR.mkdir(exist_ok=True)
    Path(os.environ["MPLCONFIGDIR"]).mkdir(exist_ok=True)
    Path(os.environ["XDG_CACHE_HOME"]).mkdir(exist_ok=True)


def generate_data(seed=6822, n=5000):
    rng = np.random.default_rng(seed)
    jurisdictions = rng.choice(["US", "EU", "Singapore", "UK"], n, p=[0.43, 0.34, 0.13, 0.10])
    age = rng.integers(18, 76, n)
    gender = rng.choice(["Female", "Male"], n, p=[0.50, 0.50])
    protected_group = rng.binomial(1, 0.38, n)
    income = np.clip(rng.lognormal(mean=10.85, sigma=0.55, size=n), 18000, 260000)
    employment_status = rng.choice(
        ["Full-time", "Part-time", "Self-employed", "Unemployed"],
        n,
        p=[0.64, 0.16, 0.14, 0.06],
    )
    credit_score = np.clip(
        rng.normal(680, 85, n)
        - protected_group * 18
        - (employment_status == "Unemployed") * 35
        + (income > 120000) * 18,
        300,
        850,
    )
    debt_to_income = np.clip(
        rng.beta(2.2, 5.8, n) + protected_group * 0.035 + (employment_status == "Unemployed") * 0.08,
        0.02,
        0.88,
    )
    loan_amount = np.clip(rng.lognormal(mean=10.45, sigma=0.72, size=n), 5000, 800000)
    loan_term_months = rng.choice([12, 24, 36, 48, 60, 84], n, p=[0.08, 0.14, 0.30, 0.20, 0.22, 0.06])
    loan_purpose = rng.choice(
        ["Credit card consolidation", "Auto", "Education", "Home improvement", "Small business", "Personal"],
        n,
        p=[0.27, 0.18, 0.10, 0.17, 0.12, 0.16],
    )
    prior_default = rng.binomial(1, np.clip(0.08 + (credit_score < 600) * 0.18 + (debt_to_income > 0.55) * 0.10, 0, 0.65))
    region_risk_score = np.clip(rng.normal(0.50, 0.17, n) + protected_group * 0.035, 0.05, 0.95)

    raw_risk = (
        -4.15
        + 0.0095 * (700 - credit_score)
        + 3.8 * debt_to_income
        - 0.000007 * income
        + 1.15 * prior_default
        + 1.10 * region_risk_score
        + 0.27 * protected_group
        + 0.22 * (employment_status == "Unemployed")
    )
    default_prob = 1 / (1 + np.exp(-raw_risk))
    default_next_12m = rng.binomial(1, np.clip(default_prob, 0.01, 0.95))

    def age_bucket(x):
        if x < 25:
            return "18-24"
        if x < 35:
            return "25-34"
        if x < 50:
            return "35-49"
        if x < 65:
            return "50-64"
        return "65+"

    df = pd.DataFrame(
        {
            "applicant_id": [f"APP-{i:05d}" for i in range(1, n + 1)],
            "jurisdiction": jurisdictions,
            "age": age,
            "age_group": [age_bucket(x) for x in age],
            "gender": gender,
            "protected_group": protected_group,
            "income": income.round(0).astype(int),
            "employment_status": employment_status,
            "credit_score": credit_score.round(0).astype(int),
            "debt_to_income": debt_to_income.round(3),
            "loan_amount": loan_amount.round(0).astype(int),
            "loan_term_months": loan_term_months,
            "loan_purpose": loan_purpose,
            "prior_default": prior_default,
            "region_risk_score": region_risk_score.round(3),
            "default_next_12m": default_next_12m,
        }
    )
    return df


def reason_1(row):
    candidates = [
        ("High debt-to-income ratio", row.debt_to_income, 0.42),
        ("Low credit score", 700 - row.credit_score, 80),
        ("Prior default history", row.prior_default, 0.5),
        ("Income insufficient for requested loan", row.loan_amount / max(row.income, 1), 2.8),
        ("Elevated regional portfolio risk", row.region_risk_score, 0.62),
    ]
    scored = [(name, value / scale) for name, value, scale in candidates]
    return max(scored, key=lambda x: x[1])[0]


def reason_2(row):
    first = reason_1(row)
    candidates = [
        ("High debt-to-income ratio", row.debt_to_income, 0.42),
        ("Low credit score", 700 - row.credit_score, 80),
        ("Prior default history", row.prior_default, 0.5),
        ("Income insufficient for requested loan", row.loan_amount / max(row.income, 1), 2.8),
        ("Elevated regional portfolio risk", row.region_risk_score, 0.62),
    ]
    scored = [(name, value / scale) for name, value, scale in candidates if name != first]
    return max(scored, key=lambda x: x[1])[0]


def jurisdiction_rules():
    return pd.DataFrame(
        [
            {
                "jurisdiction": "US",
                "regulatory_focus": "Fair lending and specific adverse-action explanations",
                "fairness_metric": "disparate_impact_ratio",
                "fairness_threshold": 0.80,
                "explainability_required": "Yes",
                "human_review_required": "Medium",
                "documentation_level": "Medium",
                "effective_date": "2026-04-27",
                "rule_version": "US-RegB-AI-2026-demo",
                "threshold_basis": "Internal disparate-impact style warning level; not a legal conclusion or statutory safe harbour.",
            },
            {
                "jurisdiction": "EU",
                "regulatory_focus": "High-risk AI bias governance, documentation and oversight",
                "fairness_metric": "approval_parity_ratio",
                "fairness_threshold": 0.90,
                "explainability_required": "Yes",
                "human_review_required": "High",
                "documentation_level": "High",
                "effective_date": "2026-04-27",
                "rule_version": "EU-AIAct-2026-demo",
                "threshold_basis": "The EU 0.90 threshold is not a statutory threshold. It is an internal governance escalation threshold chosen to reflect lower tolerance for unexplained bias in high-risk AI systems.",
            },
            {
                "jurisdiction": "Singapore",
                "regulatory_focus": "AI accountability, explainability and internal governance",
                "fairness_metric": "internal_fairness_review",
                "fairness_threshold": 0.85,
                "explainability_required": "Yes",
                "human_review_required": "High",
                "documentation_level": "Medium",
                "effective_date": "2026-04-27",
                "rule_version": "SG-AI-Gov-2026-demo",
                "threshold_basis": "Extensible internal governance example only; not analyzed as a legal threshold in this assignment.",
            },
            {
                "jurisdiction": "UK",
                "regulatory_focus": "Consumer Duty outcome monitoring and fair outcomes",
                "fairness_metric": "outcome_gap",
                "fairness_threshold": 0.85,
                "explainability_required": "Yes",
                "human_review_required": "Medium",
                "documentation_level": "High",
                "effective_date": "2026-04-27",
                "rule_version": "UK-ConsumerDuty-2026-demo",
                "threshold_basis": "Extensible internal governance example only; not analyzed as a legal threshold in this assignment.",
            },
        ]
    )


def eu_ai_act_obligations():
    return pd.DataFrame(
        [
            {
                "obligation_id": "EU-AIA-10-01",
                "jurisdiction": "EU",
                "source": "AI Act",
                "article": "Art 10",
                "control": "data suitability and bias checks",
                "evidence_output": "data quality report + bias metrics",
                "owner": "AI Governance",
            },
            {
                "obligation_id": "EU-AIA-14-01",
                "jurisdiction": "EU",
                "source": "AI Act",
                "article": "Art 14",
                "control": "human oversight",
                "evidence_output": "reviewer sign-off log",
                "owner": "Model Risk",
            },
            {
                "obligation_id": "EU-AIA-17-01",
                "jurisdiction": "EU",
                "source": "AI Act",
                "article": "Art 17",
                "control": "QMS and version control",
                "evidence_output": "rule version + validation record",
                "owner": "Compliance",
            },
        ]
    )


def data_dictionary():
    rows = [
        ("applicant_id", "Synthetic application identifier.", False, "No", "None.", "Operational ID only; not a credit-risk predictor."),
        ("jurisdiction", "Active reporting jurisdiction for the simulated application.", False, "Potential proxy", "Assigned to demonstrate cross-border rule switching.", "Use for governance routing, not credit approval."),
        ("age", "Applicant age in years.", True, "Protected/proxy", "Age distribution is synthetic.", "Age may be legally restricted or require counsel review before production use."),
        ("age_group", "Binned age label for portfolio review.", False, "Protected/proxy", "Derived from synthetic age.", "Governance reporting only; avoid in production scoring unless explicitly permitted."),
        ("gender", "Synthetic gender attribute for documentation realism.", False, "Protected", "Randomly assigned; not used in the model.", "Do not use in production credit scoring."),
        ("protected_group", "Synthetic protected-group proxy used for fairness monitoring.", False, "Protected/proxy", "Used to introduce a mild controlled disparity for monitoring demonstration.", "Fairness analysis only; do not use as a production credit model input."),
        ("income", "Annual income in synthetic currency units.", True, "Potential proxy", "Generated from a lognormal distribution.", "May be valid with affordability policy controls and data-quality validation."),
        ("employment_status", "Synthetic employment category.", True, "Potential proxy", "Unemployment affects synthetic credit score, debt burden and default outcome.", "Review for fair-lending proxy risk and reason-code suitability."),
        ("credit_score", "Synthetic bureau-style credit score.", True, "Potential proxy", "Protected-group proxy slightly lowers the generated score to create measurable disparity.", "Potentially valid only with bureau lineage, permissible-purpose checks and adverse-action mapping."),
        ("debt_to_income", "Synthetic debt-to-income ratio.", True, "No", "Higher values increase synthetic default risk.", "Potentially valid affordability input with source validation."),
        ("loan_amount", "Requested synthetic loan amount.", True, "Potential proxy", "Generated independently from income and purpose.", "Use with affordability controls; avoid encoding geography or protected status."),
        ("loan_term_months", "Requested loan term.", True, "No", "Sampled from common term buckets.", "Potentially valid input if product policy permits."),
        ("loan_purpose", "Synthetic purpose for the requested loan.", True, "Potential proxy", "Categorical feature for reason-code demonstration.", "Review for disparate-impact proxy risk before production use."),
        ("prior_default", "Synthetic prior-default indicator.", True, "Potential proxy", "Probability rises when synthetic credit score and debt burden worsen.", "Use only with accurate source data, recency policy and legal review."),
        ("region_risk_score", "Synthetic portfolio/geographic risk proxy.", True, "Potential proxy", "Protected group has a mild upward shift to demonstrate proxy-bias monitoring.", "High-risk proxy; avoid or strictly validate before production scoring."),
        ("default_next_12m", "Synthetic target label for model training.", False, "No", "Generated from the synthetic risk equation.", "Training target only; production needs validated repayment outcome lineage."),
        ("model_score", "Trained logistic-regression predicted probability of default.", False, "Derived", "Generated after fitting the model with predict_proba.", "Model output; audit and monitor, not an independent input."),
        ("approved", "Approval flag derived from model_score and the internal threshold.", False, "Derived", "Score below approval threshold is approved in the prototype.", "Decision-support output only; production requires policy and human-control design."),
        ("manual_review", "Borderline score-band flag for human review.", False, "Derived", "Derived from the internal manual-review score band.", "Operational routing output; production needs reviewer workflow and SLA controls."),
        ("decision_reason_1", "Top mapped adverse-action reason for rejected applications.", False, "Derived", "Generated from model-aligned risk factors.", "Consumer-facing reason candidate; legal review required."),
        ("decision_reason_2", "Second mapped adverse-action reason for rejected applications.", False, "Derived", "Generated from model-aligned risk factors.", "Consumer-facing reason candidate; legal review required."),
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "field_name",
            "description",
            "used_in_model",
            "protected_or_proxy",
            "synthetic_bias_role",
            "production_credit_model_note",
        ],
    )


def train_model(df):
    X = df[MODEL_FEATURES]
    y = df["default_next_12m"]
    pre = ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    model = Pipeline(
        [
            ("preprocess", pre),
            ("model", LogisticRegression(max_iter=1000, random_state=6822)),
        ]
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=6822, stratify=y)
    model.fit(X_train, y_train)
    pred_prob = model.predict_proba(X_test)[:, 1]
    pred = (pred_prob >= 0.5).astype(int)
    return model, {
        "auc": float(roc_auc_score(y_test, pred_prob)),
        "accuracy": float(accuracy_score(y_test, pred)),
        "test_size": int(len(y_test)),
    }


def score_applications(model, df):
    scored = df.copy()
    model_score = model.predict_proba(scored[MODEL_FEATURES])[:, 1]
    scored["model_score"] = np.round(model_score, 4)
    scored["approved"] = (scored["model_score"] < APPROVAL_THRESHOLD).astype(int)
    scored["manual_review"] = (
        (scored["model_score"] >= MANUAL_REVIEW_LOWER)
        & (scored["model_score"] < MANUAL_REVIEW_UPPER)
    ).astype(int)
    scored["decision_reason_1"] = [
        reason_1(row) if row.approved == 0 else "Approved"
        for row in scored.itertuples()
    ]
    scored["decision_reason_2"] = [
        reason_2(row) if row.approved == 0 else "Approved"
        for row in scored.itertuples()
    ]
    return scored


def train_random_forest_benchmark(df):
    X = df[MODEL_FEATURES]
    y = df["default_next_12m"]
    pre = ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    model = Pipeline(
        [
            ("preprocess", pre),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=250,
                    max_depth=8,
                    min_samples_leaf=20,
                    class_weight="balanced_subsample",
                    random_state=6822,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=6822, stratify=y)
    model.fit(X_train, y_train)
    pred_prob = model.predict_proba(X_test)[:, 1]
    pred = (pred_prob >= 0.5).astype(int)
    feature_names = [clean_feature_name(name) for name in model.named_steps["preprocess"].get_feature_names_out()]
    importances = pd.DataFrame(
        {
            "technical_feature": feature_names,
            "random_forest_importance": model.named_steps["model"].feature_importances_,
            "mapped_consumer_reason": [map_feature_to_consumer_reason(name) for name in feature_names],
        }
    ).sort_values("random_forest_importance", ascending=False)
    metrics = {
        "model_name": "Random Forest challenger",
        "auc": float(roc_auc_score(y_test, pred_prob)),
        "accuracy": float(accuracy_score(y_test, pred)),
        "test_size": int(len(y_test)),
        "purpose": "Non-linear benchmark for technical comparison; not selected as primary governance model.",
    }
    return model, metrics, importances


def apply_jurisdiction_logic(df, rules):
    rows = []
    protected = df[df["protected_group"] == 1]
    reference = df[df["protected_group"] == 0]
    protected_approval = protected["approved"].mean()
    reference_approval = reference["approved"].mean()
    parity = protected_approval / reference_approval
    protected_default = protected["default_next_12m"].mean()
    reference_default = reference["default_next_12m"].mean()
    protected_fpr = ((protected["approved"] == 0) & (protected["default_next_12m"] == 0)).sum() / max((protected["default_next_12m"] == 0).sum(), 1)
    reference_fpr = ((reference["approved"] == 0) & (reference["default_next_12m"] == 0)).sum() / max((reference["default_next_12m"] == 0).sum(), 1)
    protected_fnr = ((protected["approved"] == 1) & (protected["default_next_12m"] == 1)).sum() / max((protected["default_next_12m"] == 1).sum(), 1)
    reference_fnr = ((reference["approved"] == 1) & (reference["default_next_12m"] == 1)).sum() / max((reference["default_next_12m"] == 1).sum(), 1)
    for rule in rules.itertuples():
        threshold = float(rule.fairness_threshold)
        flag = parity < threshold
        if rule.jurisdiction == "US":
            action = (
                "Trigger fair lending review; verify specific adverse-action reasons and evaluate fairness risk without presenting the metric as a legal conclusion."
                if flag
                else "Continue monitoring adverse-action reason quality and approval parity as internal governance evidence."
            )
            conflict = "US legal treatment of disparate impact is changing after the 2026 Regulation B final rule; the tool treats disparity as risk evidence, not legal advice."
        elif rule.jurisdiction == "EU":
            action = (
                "Trigger high-risk AI bias governance review, document testing, assign human oversight and remediation owner."
                if flag
                else "Maintain high-risk AI documentation, human oversight evidence and periodic bias testing."
            )
            conflict = "EU mode treats creditworthiness AI as a high-risk system and requires stronger documentation and oversight even when US mode does not escalate."
        elif rule.jurisdiction == "Singapore":
            action = "Escalate to AI governance owner for accountable review." if flag else "Continue accountable AI monitoring."
            conflict = "Included as extensible design only; the assignment analysis focuses on US and EU."
        else:
            action = "Review customer outcome gap under Consumer Duty governance." if flag else "Continue outcome monitoring."
            conflict = "Included as extensible design only; the assignment analysis focuses on US and EU."
        rows.append(
            {
                "active_jurisdiction": rule.jurisdiction,
                "metric_name": rule.fairness_metric,
                "protected_group_approval_rate": round(protected_approval, 4),
                "reference_group_approval_rate": round(reference_approval, 4),
                "approval_parity_ratio": round(parity, 4),
                "protected_group_default_rate": round(protected_default, 4),
                "reference_group_default_rate": round(reference_default, 4),
                "protected_group_false_positive_rate": round(protected_fpr, 4),
                "reference_group_false_positive_rate": round(reference_fpr, 4),
                "protected_group_false_negative_rate": round(protected_fnr, 4),
                "reference_group_false_negative_rate": round(reference_fnr, 4),
                "threshold": threshold,
                "threshold_basis": rule.threshold_basis,
                "fairness_flag": bool(flag),
                "required_governance_action": action,
                "jurisdiction_conflict_note": conflict,
            }
        )
    return pd.DataFrame(rows)


def build_sensitivity_analysis(df, rules):
    rows = []
    shifts = [-0.04, -0.02, 0.00, 0.02, 0.04, 0.06]
    for shift in shifts:
        temp = df.copy()
        shifted_score = temp["model_score"].copy()
        shifted_score.loc[temp["protected_group"] == 1] = np.clip(
            shifted_score.loc[temp["protected_group"] == 1] + shift,
            0.001,
            0.999,
        )
        temp["approved"] = (shifted_score < APPROVAL_THRESHOLD).astype(int)
        protected = temp[temp["protected_group"] == 1]
        reference = temp[temp["protected_group"] == 0]
        parity = protected["approved"].mean() / reference["approved"].mean()
        for rule in rules[rules["jurisdiction"].isin(["US", "EU"])].itertuples():
            rows.append(
                {
                    "protected_group_score_shift": shift,
                    "active_jurisdiction": rule.jurisdiction,
                    "approval_parity_ratio": round(parity, 4),
                    "threshold": float(rule.fairness_threshold),
                    "fairness_flag": bool(parity < float(rule.fairness_threshold)),
                    "interpretation": (
                        "Escalate under this jurisdiction"
                        if parity < float(rule.fairness_threshold)
                        else "Monitor without escalation"
                    ),
                }
            )
    return pd.DataFrame(rows)


def build_shap_explanations(model, df):
    features = [
        "age",
        "income",
        "credit_score",
        "debt_to_income",
        "loan_amount",
        "loan_term_months",
        "prior_default",
        "region_risk_score",
        "employment_status",
        "loan_purpose",
    ]
    sample = df[df["approved"].eq(0)].head(3).copy()
    preprocessor = model.named_steps["preprocess"]
    classifier = model.named_steps["model"]
    transformed_all = preprocessor.transform(df[features])
    transformed_sample = preprocessor.transform(sample[features])
    try:
        transformed_all = transformed_all.toarray()
        transformed_sample = transformed_sample.toarray()
    except AttributeError:
        pass
    background_mean = np.asarray(transformed_all).mean(axis=0)
    transformed_sample = np.asarray(transformed_sample)
    coefs = classifier.coef_[0]
    feature_names = preprocessor.get_feature_names_out()
    rows = []
    for sample_pos, (_, row) in enumerate(sample.iterrows()):
        contributions = coefs * (transformed_sample[sample_pos] - background_mean)
        positive_idx = [idx for idx in np.argsort(contributions)[::-1] if contributions[idx] > 0]
        selected_idx = []
        seen_reasons = set()
        for idx in positive_idx:
            feature = clean_feature_name(feature_names[idx])
            reason = map_feature_to_consumer_reason(feature)
            if reason not in seen_reasons:
                selected_idx.append(idx)
                seen_reasons.add(reason)
            if len(selected_idx) >= 3:
                break
        for rank, idx in enumerate(selected_idx, start=1):
            feature = clean_feature_name(feature_names[idx])
            rows.append(
                {
                    "applicant_id": row["applicant_id"],
                    "model_score": row["model_score"],
                    "rank": rank,
                    "technical_feature": feature,
                    "linear_shap_log_odds": round(float(contributions[idx]), 4),
                    "direction": "increases predicted default risk",
                    "mapped_consumer_reason": map_feature_to_consumer_reason(feature),
                }
            )
    return pd.DataFrame(rows)


def clean_feature_name(name):
    name = name.replace("num__", "").replace("cat__", "")
    name = name.replace("employment_status_", "employment_status=")
    name = name.replace("loan_purpose_", "loan_purpose=")
    return name


def map_feature_to_consumer_reason(feature):
    if feature == "debt_to_income":
        return "High debt-to-income ratio"
    if feature == "credit_score":
        return "Low credit score"
    if feature == "income":
        return "Income insufficient for requested loan"
    if feature == "prior_default":
        return "Prior default history"
    if feature == "region_risk_score":
        return "Elevated regional portfolio risk"
    if feature.startswith("employment_status="):
        return "Employment status increases repayment risk"
    if feature == "loan_amount":
        return "Loan amount high relative to profile"
    if feature == "loan_term_months":
        return "Loan term increases repayment risk"
    return "Other model risk factor"


def build_model_outputs(df, metrics):
    us_action = metrics.loc[metrics["active_jurisdiction"] == "US", "required_governance_action"].iloc[0]
    eu_action = metrics.loc[metrics["active_jurisdiction"] == "EU", "required_governance_action"].iloc[0]
    out = df[
        [
            "applicant_id",
            "jurisdiction",
            "protected_group",
            "model_score",
            "approved",
            "manual_review",
            "decision_reason_1",
            "decision_reason_2",
        ]
    ].copy()
    out["us_mode_output"] = np.where(
        out["approved"] == 0,
        "Adverse-action notice required: " + out["decision_reason_1"] + "; " + out["decision_reason_2"],
        "Approved; monitor portfolio-level fair lending metrics",
    )
    out["eu_mode_output"] = np.where(
        out["manual_review"] == 1,
        "Human oversight required for borderline high-risk AI credit decision",
        "Bias testing and high-risk AI documentation evidence required",
    )
    out["portfolio_us_governance_action"] = us_action
    out["portfolio_eu_governance_action"] = eu_action
    return out


def add_wrapped(ax, text, x, y, width=92, size=10.5, weight="normal", color="#1f2933", line_gap=0.045):
    lines = []
    for para in str(text).split("\n"):
        if not para.strip():
            lines.append("")
        else:
            lines.extend(textwrap.wrap(para, width=width))
    for line in lines:
        ax.text(x, y, line, transform=ax.transAxes, fontsize=size, fontweight=weight, color=color, va="top")
        y -= line_gap
    return y


def pdf_page(pdf, title, sections, subtitle=None, footer=None, landscape=False):
    figsize = (11, 8.5) if landscape else (8.5, 11)
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")
    ax.add_patch(plt.Rectangle((0, 0.955), 1, 0.045, transform=ax.transAxes, color="#17324d", zorder=-1))
    ax.text(0.055, 0.925, title, transform=ax.transAxes, fontsize=20 if not landscape else 22, fontweight="bold", color="#17324d", va="top")
    y = 0.88
    if subtitle:
        y = add_wrapped(ax, subtitle, 0.055, y, width=105 if landscape else 85, size=10.5, color="#52616b", line_gap=0.035)
        y -= 0.02
    for heading, body in sections:
        ax.text(0.055, y, heading, transform=ax.transAxes, fontsize=12.5, fontweight="bold", color="#0b7285", va="top")
        y -= 0.038
        y = add_wrapped(ax, body, 0.075, y, width=120 if landscape else 88, size=9.8 if landscape else 10.2, color="#1f2933", line_gap=0.036 if landscape else 0.038)
        y -= 0.025
        if y < 0.11:
            break
    if footer:
        ax.text(0.055, 0.045, footer, transform=ax.transAxes, fontsize=8.5, color="#6b7280", va="bottom")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def task1_jurisdiction_comparison_page(pdf):
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis("off")
    ax.add_patch(plt.Rectangle((0, 0.94), 1, 0.06, transform=ax.transAxes, color="#17324d", zorder=-1))
    ax.text(0.04, 0.91, "Task 1 - Jurisdiction Comparison Matrix", transform=ax.transAxes, fontsize=21, fontweight="bold", color="#17324d", va="top")
    ax.text(0.04, 0.86, "The same credit model output creates different governance duties depending on the active jurisdiction.", transform=ax.transAxes, fontsize=10.5, color="#52616b", va="top")
    rows = [
        ["Question", "United States", "European Union", "Tool design consequence"],
        [
            "Primary concern",
            "ECOA / Regulation B;\nadverse-action reasons",
            "AI Act high-risk AI;\ndata, oversight, bias controls",
            "Separate consumer reasons\nfrom governance evidence",
        ],
        [
            "Disparity signal",
            "Internal risk evidence after 2026\nRegulation B policy shift",
            "Bias governance trigger for\nhigh-risk AI documentation",
            "Store statistical disparity, but avoid\ncalling it a legal violation",
        ],
        [
            "Threshold stance",
            "0.80 internal warning level",
            "0.90 conservative\ngovernance level",
            "Expose thresholds and rule versions\nrather than hiding them in code",
        ],
        [
            "Human role",
            "Review reason quality\nand fair-lending risk",
            "Document oversight,\ntesting and remediation owner",
            "Escalate to accountable owners\ninstead of auto-remediation",
        ],
    ]
    table = ax.table(
        cellText=rows[1:],
        colLabels=rows[0],
        cellLoc="left",
        colLoc="left",
        bbox=[0.035, 0.19, 0.93, 0.58],
        colWidths=[0.16, 0.26, 0.27, 0.31],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.2)
    table.scale(1, 1.6)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#adb5bd")
        cell.set_linewidth(0.7)
        if row == 0:
            cell.set_facecolor("#d0ebff")
            cell.set_text_props(weight="bold", color="#17324d")
        elif col == 0:
            cell.set_facecolor("#f8f9fa")
            cell.set_text_props(weight="bold")
        elif col == 3:
            cell.set_facecolor("#fff9db")
    ax.text(0.04, 0.10, "Visual takeaway: the tool is jurisdiction-aware because it changes the governance response, not because it merely labels a row as US or EU.", transform=ax.transAxes, fontsize=9.5, color="#1f2933")
    ax.text(0.04, 0.055, f"{STUDENT['name']} | {STUDENT['matriculation_id']}", transform=ax.transAxes, fontsize=8.5, color="#6b7280")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def task2_stakeholder_matrix_page(pdf):
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis("off")
    ax.add_patch(plt.Rectangle((0, 0.94), 1, 0.06, transform=ax.transAxes, color="#17324d", zorder=-1))
    ax.text(0.04, 0.91, "Task 2 - Stakeholder Trade-Off Matrix", transform=ax.transAxes, fontsize=21, fontweight="bold", color="#17324d", va="top")
    ax.text(0.04, 0.86, "Values audit: who benefits, who bears risk, and what the product does in response.", transform=ax.transAxes, fontsize=10.5, color="#52616b", va="top")
    rows = [
        ["Stakeholder", "What they want", "If the tool is wrong", "Design response"],
        [
            "Citi Compliance /\nModel Risk",
            "Audit evidence, explainable outputs,\nrule-version control",
            "False comfort; wrong jurisdiction;\nmissed regulatory issue",
            "Versioned rules, conflict notes,\nmonthly fairness review",
        ],
        [
            "Consumers /\napplicants",
            "Fair access, understandable reasons,\nability to challenge",
            "Wrong denial, opaque explanation,\nunequal review burden",
            "Specific adverse-action mapping;\npositive-risk SHAP features only",
        ],
        [
            "Regulators",
            "Repeatable governance evidence,\nhuman accountability",
            "Dashboard theatre without real\nrisk detection",
            "FPR/FNR/parity metrics plus\nlimitations and escalation owner",
        ],
        [
            "Meridian",
            "Defensible product niche and\nrepeatable subscription model",
            "Overclaiming legal compliance;\nloss of trust",
            "Narrow scope: governance tool,\nnot legal advice or auto-approval",
        ],
    ]
    table = ax.table(
        cellText=rows[1:],
        colLabels=rows[0],
        cellLoc="left",
        colLoc="left",
        bbox=[0.035, 0.18, 0.93, 0.59],
        colWidths=[0.17, 0.28, 0.27, 0.28],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.8)
    table.scale(1, 1.65)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#adb5bd")
        cell.set_linewidth(0.7)
        if row == 0:
            cell.set_facecolor("#d0ebff")
            cell.set_text_props(weight="bold", color="#17324d")
        elif col == 0:
            cell.set_facecolor("#f8f9fa")
            cell.set_text_props(weight="bold")
        elif col == 3:
            cell.set_facecolor("#e6fcf5")
    ax.text(0.04, 0.095, "Visual takeaway: the product is not neutral. It chooses to make trade-offs visible, documented and reviewable.", transform=ax.transAxes, fontsize=9.5, color="#1f2933")
    ax.text(0.04, 0.055, f"{STUDENT['name']} | {STUDENT['matriculation_id']}", transform=ax.transAxes, fontsize=8.5, color="#6b7280")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def write_task1_pdf(path):
    with PdfPages(path) as pdf:
        pdf_page(
            pdf,
            "Task 1 - Selection and Research",
            [
                (
                    "Regulated entity",
                    "The selected entity is Citi / Citigroup Inc., a named global bank headquartered in the United States. Citi is a US-headquartered global bank with operations across North America, Europe, the Middle East, Africa and Asia. Citi's public materials list European locations such as Warsaw, Budapest, Dublin and London, and Citi's 2025 Annual Report states its mission as responsibly providing financial services that enable growth and economic progress. That makes AI credit governance a credible strategic topic rather than a purely academic example.",
                ),
                (
                    "Regulatory domain",
                    "The domain is AI credit scoring, fair lending, algorithmic fairness and AI governance. The tool is designed for a cross-border bank using machine-learning credit models to support consumer or small-business credit decisions. The compliance question is not whether the model is accurate in a generic sense; it is whether the model can be governed differently when US fair-lending explainability expectations and EU high-risk AI governance expectations apply.",
                ),
                (
                    "Evidence boundary",
                    "The public sources support Citi as a cross-border bank with US headquarters and European operations. They do not prove that Citi currently uses the exact AI credit model simulated here in the EU. This is an honest design boundary: the assignment uses Citi as a realistic regulated case and builds a plausible RegTech prototype, not a factual allegation about Citi's live systems.",
                ),
            ],
            subtitle=f"{STUDENT['name']} | {STUDENT['matriculation_id']} | {STUDENT['email']}",
        )
        pdf_page(
            pdf,
            "Task 1 - Jurisdictional Rationale",
            [
                (
                    "Jurisdictional divergence",
                    "In the US, the practical focus is on fair lending governance, specific adverse-action reasons and the changing legal treatment of disparate impact under ECOA / Regulation B. The CFPB has warned that use of AI or complex models does not remove the creditor's obligation to explain adverse action specifically and accurately. The 2026 Regulation B final rule shows a policy turn: disparate-impact treatment is politically contested, so the tool separates statistical disparity evidence from legal conclusions. In the EU, the AI Act classifies creditworthiness and credit-score AI systems as high-risk, subject to stronger expectations around risk management, data governance, documentation, transparency, bias testing and human oversight.",
                ),
                (
                    "Why this is a RegTech problem",
                    "A memo can describe the two regimes, but it cannot continuously apply different thresholds, explanations and governance actions to live model outputs. A jurisdiction-aware tool can store regulatory parameters, apply them to the same model output, and show why the US and EU modes produce different compliance consequences.",
                ),
            ],
            footer="References are listed on the final page.",
        )
        pdf_page(
            pdf,
            "Task 1 - CFPB 2026 Policy Shift",
            [
                (
                    "What changed from the Biden-era posture",
                    "Under the earlier CFPB posture, the practical RegTech emphasis was that AI or complex credit models still had to produce specific, accurate adverse-action explanations, and fair-lending monitoring often treated disparate-impact evidence as legally salient. The 2026 Regulation B final rule changes the federal ECOA posture by removing effects-test language and stating that ECOA does not authorize disparate-impact liability under Regulation B. It also narrows discouragement toward intentional oral or written statements and adds tighter conditions for for-profit special purpose credit programs.",
                ),
                (
                    "Why this matters for the tool",
                    "A weak tool would simply delete disparate-impact monitoring after the rule change. A stronger RegTech design treats the change as a governance-versioning problem. The tool should keep approval parity, FPR and FNR as internal risk evidence, but label them differently: not as automatic legal violations under US federal ECOA, but as model-risk, consumer-harm, reputational, state-law, EU and litigation-risk signals.",
                ),
                (
                    "Concrete design response",
                    "US mode therefore separates three layers: first, individual adverse-action reasons remain necessary for rejected applicants; second, group disparity metrics remain useful for internal governance and management review; third, the legal conclusion is deliberately withheld and routed to Legal / Compliance. The rule file stores effective dates and version notes so outputs generated before and after the 2026 change are not collapsed into one interpretation.",
                ),
            ],
        )
        task1_jurisdiction_comparison_page(pdf)
        pdf_page(
            pdf,
            "Task 1 - Political Choice",
            [
                (
                    "RegTech tools as political artefacts",
                    "The US/EU threshold difference is not only a technical parameter. It reflects a political choice about where to place the burden of uncertainty. US mode uses a lower internal review threshold and emphasizes adverse-action explanations because the US framework is more focused on individual credit decisions, lender process and the changing legal boundaries of disparate impact. EU mode uses a stricter internal 0.90 escalation threshold because Meridian chooses a lower tolerance for unexplained bias in high-risk AI governance; this is not presented as an AI Act statutory number.",
                ),
                (
                    "Design consequence of the US policy shift",
                    "The 2026 CFPB Regulation B final rule changes the operating posture of a RegTech tool. A high-quality product should not simply delete disparate-impact monitoring when enforcement philosophy shifts. It should retain the metric as internal governance evidence, version the legal interpretation, avoid saying legal violation, and route borderline cases to adverse-action reason review and management sign-off.",
                ),
                (
                    "Why 0.80 vs 0.90",
                    "The 0.80 US threshold is used as an internal disparate-impact style warning level, not as a legal conclusion. The EU 0.90 threshold is not a statutory threshold. It is Meridian's internal conservative escalation threshold for high-risk AI governance, chosen to reflect lower tolerance for unexplained bias. Legal obligations are mapped separately through Article 10 data governance, Article 14 human oversight and Article 17 quality-management controls.",
                ),
                (
                    "Business implication",
                    "A global RegTech company cannot pretend that one universal fairness standard is neutral. Choosing a threshold is also choosing which errors, costs and stakeholders receive priority. This is why the tool exposes thresholds, rule versions and conflict notes instead of hiding political choices inside model code.",
                ),
            ],
        )
        refs_a = "\n".join([f"- {s['label']}: {s['url']}. {s['note']}" for s in SOURCES[:4]])
        refs_b = "\n".join([f"- {s['label']}: {s['url']}. {s['note']}" for s in SOURCES[4:7]])
        refs_c = "\n".join([f"- {s['label']}: {s['url']}. {s['note']}" for s in SOURCES[7:]])
        pdf_page(
            pdf,
            "Task 1 - References",
            [("Public sources used", refs_a)],
        )
        pdf_page(
            pdf,
            "Task 1 - References Continued",
            [("Public sources used", refs_b)],
        )
        pdf_page(
            pdf,
            "Task 1 - References Continued",
            [("Public sources used", refs_c)],
            footer="URLs were checked during preparation on 2026-04-27.",
        )


def write_task2_pdf(path, metrics):
    us = metrics[metrics["active_jurisdiction"].eq("US")].iloc[0]
    with PdfPages(path) as pdf:
        pdf_page(
            pdf,
            "Task 2 - Values Audit",
            [
                (
                    "Company mission and identity",
                    "The hypothetical company is Meridian FairAI Governance, an early-stage RegTech firm that helps cross-border financial institutions govern AI credit models through explainable, auditable and jurisdiction-aware compliance tools. Its core values are practical fairness, transparent accountability, regulatory humility and human judgement. The company is deliberately small: about 25-40 specialists across model risk, AI explainability, compliance operations, regulatory engineering and customer implementation. Its near-term aspiration is a focused enterprise SaaS model for US/EU banks: low hundreds of thousands of dollars per annual bank deployment, implementation support sold separately, and expansion only where Meridian can keep legal humility and model-governance quality intact.",
                ),
                (
                    "Whose perspective the tool serves",
                    "The paying users are the bank's CCO, CRO, Fair Lending Team, AI Governance Team and Model Risk Management function. That choice is partly commercial: these teams have budget, urgency and accountability. But the tool also embeds consumer and public-interest concerns by measuring approval gaps, false-positive / false-negative differences and adverse-action reason quality. There is a real tension: the bank may want efficient approvals and lower regulatory exposure, while consumers want fair access and understandable reasons, and regulators want market-wide accountability.",
                ),
            ],
            subtitle="There are no correct answers, but shallow answers would hide the trade-offs. This audit states them directly.",
        )
        task2_stakeholder_matrix_page(pdf)
        pdf_page(
            pdf,
            "Task 2 - Values Audit",
            [
                (
                    "Genuine risk or documentation?",
                    "The tool is designed to measure genuine model and consumer harm risk, not merely to produce a checklist. One design choice that reflects this is group-level monitoring of approval parity plus false positive and false negative rates. These metrics are not required simply to produce a document; they help identify whether a model is systematically denying lower-risk applicants from a protected group or approving higher-risk applicants in a way that later harms the bank and consumers. Documentation remains necessary, but the dashboard starts from risk detection.",
                ),
                (
                    "Who bears the cost when the tool is wrong?",
                    "A false positive can wrongly classify a low-risk applicant as high risk, causing a consumer to be denied credit or pushed into worse terms. A false negative can approve genuinely risky credit, increasing losses for Citi and possibly encouraging unsustainable borrowing. A fairness false negative is more serious from a public-interest perspective because it can miss systematic discrimination. A fairness false positive raises compliance cost and may slow lending without improving fairness. A jurisdiction misconfiguration can make the bank apply the wrong rule set, creating regulatory exposure and consumer harm. Model drift can silently degrade both credit performance and fairness over time.",
                ),
                (
                    "Boundary of responsibility",
                    "The tool serves management judgement; it does not replace it. Its ethical stance is that automation should surface risks, evidence and unresolved tensions, while final credit policy, legal interpretation and customer remediation remain accountable human decisions.",
                ),
            ],
        )
        pdf_page(
            pdf,
            "Task 2 - Error-Rate Tension",
            [
                (
                    "The FNR result is not a simple fairness victory",
                    f"The generated portfolio has a protected-group false negative rate of {us['protected_group_false_negative_rate']:.4f} and a reference-group false negative rate of {us['reference_group_false_negative_rate']:.4f}. A lower FNR for the protected group means fewer actual defaulters in that group are incorrectly approved. On its own, that sounds safer. But the protected group also has a lower approval rate, so the model may be achieving lower default leakage partly by being more conservative for that group.",
                ),
                (
                    "Who is helped and who may be harmed",
                    "From the bank's risk perspective, lower protected-group FNR reduces credit losses. From the consumer perspective, the same pattern can mean more borderline protected-group applicants are denied or pushed into manual review. The ethical tension is therefore between safety-and-soundness risk and access-to-credit harm.",
                ),
                (
                    "How the tool should present it",
                    "The tool should not label the lower FNR as proof of fairness. It should show FNR together with approval parity, false positive rate and adverse-action reasons. If approval parity is weak while FNR is low, management should ask whether the model is too conservative for one group rather than simply celebrating lower default risk.",
                ),
            ],
        )
        pdf_page(
            pdf,
            "Task 2 - Model Choice as Value Choice",
            [
                (
                    "Random Forest creates a new ethical tension",
                    "The prototype includes a Random Forest challenger to test whether a more complex non-linear model materially improves performance. In this run, the challenger does not provide a significant advantage over logistic regression, while it is harder to translate into stable consumer-facing adverse-action reasons without additional SHAP validation.",
                ),
                (
                    "Why logistic regression is not only a technical choice",
                    "Choosing logistic regression as the primary governance model is therefore a values decision as well as a modelling decision. Meridian is prioritizing explainability, auditability and the consumer's ability to understand or challenge a denial over a small or uncertain performance gain from a more opaque model.",
                ),
                (
                    "Genuine tension",
                    "This choice is not cost-free. A more complex model could eventually capture legitimate non-linear risk patterns. But if that extra complexity does not clearly improve outcomes, using it in a credit context may shift too much burden onto applicants who receive less understandable reasons. The product's default position is that model performance must earn its opacity.",
                ),
            ],
        )
        pdf_page(
            pdf,
            "Task 2 - Commercial and Ethical Choice",
            [
                (
                    "Why this customer segment can pay",
                    "This product choice is primarily based on Meridian's competencies, not market size. The team's advantage is explainable AI, model-risk governance and jurisdiction rule configuration, so AI credit governance is where it can build a defensible product. The market evidence supports feasibility rather than driving the decision: Grand View Research estimated the global RegTech market at USD 24.34 billion in 2025 and projected USD 112.10 billion by 2033, with large enterprises and risk/compliance management as important segments. A global bank therefore has both the compliance budget and the operational need; Meridian's constraint is not demand, but staying narrow enough that its controls remain auditable.",
                ),
                (
                    "Tension I am not hiding",
                    "A bank may buy this tool to reduce regulatory exposure, while a consumer may want a simpler answer: was I treated fairly and can I challenge the decision? The tool tries to reduce that gap by forcing specific adverse-action reasons and group-level harm metrics into the management view. It still cannot guarantee that the consumer understands the explanation or has equal bargaining power.",
                ),
                (
                    "A feature intentionally not included",
                    "The prototype does not auto-retrain the model when a fairness flag appears. Automatic remediation would look sophisticated, but it could hide policy choices inside code. The tool instead escalates to human review because deciding whether to change a credit model, adjust a threshold or remediate customers is a governance judgement, not a purely technical optimization.",
                ),
            ],
        )
        pdf_page(
            pdf,
            "Task 2 - Evidence Boundary",
            [
                (
                    "Case-study honesty",
                    "The Citi evidence supports a realistic cross-border banking case, but it does not prove Citi's actual EU AI-credit exposure. Meridian would treat a real sales cycle as discovery-led: first verify model inventory, jurisdictions, products, customer segments and decision workflows, then configure the governance controls. This avoids turning public research into unsupported claims about a named bank.",
                ),
                (
                    "Why this still fits the assignment",
                    "The assignment asks for a named regulated entity and a RegTech product opportunity. The submission therefore uses public evidence to justify Citi as a plausible cross-border case and uses synthetic data to demonstrate the product mechanics. It does not claim access to proprietary Citi model documentation.",
                ),
            ],
        )
        pdf_page(
            pdf,
            "Task 2 - Competitive Position",
            [
                (
                    "What Meridian is good at",
                    "Meridian's competence is not generic consulting. Its niche is the translation layer between model outputs, explainability evidence, fairness metrics and jurisdiction-specific governance actions. The product is designed for model-risk and compliance teams that need repeatable evidence rather than one-off memos.",
                ),
                (
                    "Differentiation from larger incumbents",
                    "Large consultancies such as Accenture or KPMG can advise on regulatory transformation, and established RegTech vendors can support KYC, AML or entity resolution. Meridian's differentiator is narrower: AI credit governance that connects SHAP-style explanations, adverse-action reason evidence, fairness thresholds and rule versioning in one workflow. The company wins by being more specialized and auditable, not by offering every compliance service.",
                ),
                (
                    "Revenue logic",
                    "The likely commercial model is annual subscription plus implementation support for large banks, with pricing tied to number of models, jurisdictions and governance users. This aligns with the buyer's ability to pay because the tool supports recurring model monitoring, rule updates and audit evidence rather than a one-time report.",
                ),
            ],
        )


def write_summary_pdf(path, metrics, model_metrics):
    us = metrics[metrics["active_jurisdiction"] == "US"].iloc[0]
    eu = metrics[metrics["active_jurisdiction"] == "EU"].iloc[0]
    body = (
        "This project designs and partially implements a jurisdiction-aware AI credit governance tool for Citi, comparing US fair-lending explainability expectations with EU AI Act high-risk AI governance expectations. "
        "The prototype uses 5,000 rows of synthetic credit application data, a logistic-regression credit risk model, configurable jurisdiction rules and portfolio-level fairness monitoring. "
        f"The same portfolio produces an approval parity ratio of {us['approval_parity_ratio']:.3f}. In US mode, the tool compares this result with a {us['threshold']:.2f} threshold and emphasizes adverse-action reasons and fair-lending review. "
        f"In EU mode, the same result is compared with Meridian's internal {eu['threshold']:.2f} governance threshold and triggers high-risk AI governance actions, including bias testing, documentation and human oversight. The threshold is not an AI Act statutory number; Article 10, 14 and 17 obligations are mapped separately. "
        f"The demonstration model achieved AUC {model_metrics['auc']:.3f} and accuracy {model_metrics['accuracy']:.3f} on a held-out synthetic test set. "
        "The tool also reports false positive and false negative rates by group and includes a sensitivity analysis showing how small score-distribution shifts can change escalation conclusions. "
        "It is intentionally limited: it is not legal advice, not a production credit approval engine, not trained on Citi customer data and not a replacement for human judgement. Its value is that it turns regulatory divergence into executable governance logic rather than a static memo."
    )
    with PdfPages(path) as pdf:
        pdf_page(
            pdf,
            "Task 3 - One-Page Summary",
            [
                ("Design choice in plain language", body),
                (
                    "What senior management should take away",
                    "The product is right for a cross-border bank because it helps compliance, risk and AI governance teams see the same model through different regulatory lenses. It solves a problem a spreadsheet cannot solve well: rule versioning, repeated metric calculation, jurisdiction-specific reporting and explicit escalation actions. Its most important limitation is that it demonstrates governance logic with synthetic data; production use would require validated customer data, legal review, model validation and operational controls.",
                ),
            ],
            subtitle=f"{STUDENT['name']} | {STUDENT['matriculation_id']} | {STUDENT['email']}",
        )


def write_model_card_pdf(path, model_metrics, rf_metrics):
    with PdfPages(path) as pdf:
        pdf_page(
            pdf,
            "Task 3 - Model Card and Governance Stub",
            [
                ("Model purpose", "Support demonstration of credit default risk scoring, approval decisioning, adverse-action reason generation and fairness monitoring for a jurisdiction-aware RegTech tool."),
                ("Intended users", "Chief Compliance Officer, Chief Risk Officer, Fair Lending Team, AI Governance Team and Model Risk Management Team. The model is a decision-support artifact, not an autonomous approval authority."),
                ("Data", "Synthetic credit application data with 5,000 rows. Variables include income, credit score, debt-to-income ratio, loan amount, loan term, employment status, prior default, protected-group proxy, jurisdiction and default outcome. The trained logistic model's predict_proba output is then written as model_score. The data is designed with reference to public lending and credit-risk datasets, but it is not real Citi customer data."),
                ("Model and metrics", f"Primary governance model: logistic regression with standard scaling. Held-out synthetic test performance: AUC {model_metrics['auc']:.3f}; accuracy {model_metrics['accuracy']:.3f}; test observations {model_metrics['test_size']}. Random Forest challenger: AUC {rf_metrics['auc']:.3f}; accuracy {rf_metrics['accuracy']:.3f}. The challenger tests non-linear signal capture, while the logistic model remains primary because its feature contributions map more directly to adverse-action reasons."),
            ],
        )
        pdf_page(
            pdf,
            "Task 3 - Governance Stub",
            [
                ("Jurisdiction logic", "Rules are stored in Task3_jurisdiction_rules.csv with jurisdiction, regulatory focus, fairness metric, threshold, threshold basis, explainability requirement, human-review level, documentation level, effective date and rule version. US mode uses a 0.80 internal warning level. EU mode uses Meridian's internal 0.90 governance escalation level, not an AI Act statutory threshold."),
                ("Human oversight", "The tool flags borderline cases for manual review and escalates portfolio-level fairness issues. Human reviewers must understand the system's limitations, override outputs where appropriate and document remediation decisions."),
                ("Error-rate interpretation", "The lower protected-group FNR should not be read as simple evidence that the model is fairer to that group. It may indicate that the model is more conservative for protected-group applicants: fewer risky applicants are approved, but more credit access may be denied. The model card therefore reports FNR together with approval parity, FPR and reason distributions."),
                ("Failure modes", "Rule changes mid-period can make stored parameters stale; mitigation is versioned rules and effective dates. Model drift can reduce accuracy or fairness; mitigation is periodic performance and data drift monitoring. Missing protected attributes can make fairness analysis incomplete; mitigation is explicit limitation flags. Explainability over-reliance can make technical explanations look like legal reasons; mitigation is a clear distinction between model explanation and legal advice."),
                ("Limitations", "Not legal advice, not a production underwriting engine, not a substitute for final credit approval, not trained on real customer data and not proof of compliance with every jurisdiction."),
            ],
        )
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis("off")
        ax.add_patch(plt.Rectangle((0, 0.94), 1, 0.06, transform=ax.transAxes, color="#17324d", zorder=-1))
        ax.text(0.04, 0.91, "Task 3 - EU AI Act Obligation Matrix", transform=ax.transAxes, fontsize=21, fontweight="bold", color="#17324d", va="top")
        ax.text(0.04, 0.85, "EU mode is not only a 0.90 threshold. It maps high-risk AI obligations to evidence outputs and owners.", transform=ax.transAxes, fontsize=10.5, color="#52616b", va="top")
        obligations = eu_ai_act_obligations()
        table = ax.table(
            cellText=obligations.values,
            colLabels=obligations.columns,
            cellLoc="left",
            colLoc="left",
            bbox=[0.035, 0.24, 0.93, 0.48],
            colWidths=[0.14, 0.10, 0.10, 0.10, 0.23, 0.23, 0.10],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(7.2)
        table.scale(1, 1.55)
        for (row, col), cell in table.get_celld().items():
            cell.set_edgecolor("#adb5bd")
            if row == 0:
                cell.set_facecolor("#d0ebff")
                cell.set_text_props(weight="bold", color="#17324d")
        ax.text(0.04, 0.13, "Threshold note: the 0.90 threshold is an internal conservative escalation level. Legal duties are represented through the controls above.", transform=ax.transAxes, fontsize=9.5, color="#1f2933")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        pdf_page(
            pdf,
            "Task 3 - EU AI Act Control Mapping",
            [
                (
                    "Article 10: data governance",
                    "The prototype maps Article 10 to explicit data lineage and quality controls: synthetic-data disclosure, protected-group proxy field, geographic and jurisdiction fields, feature definitions, bias metrics, data-suitability limitations and repeatable sensitivity analysis. In production, this would become source-system lineage, representativeness testing, missing-data controls and bias mitigation records for training, validation and test data.",
                ),
                (
                    "Article 14: human oversight",
                    "The prototype maps Article 14 to reviewer sign-off for borderline applications and fairness escalations. A production workflow would define reviewer competence, override authority, escalation SLAs and records of human decisions.",
                ),
                (
                    "Article 17: quality management system",
                    "The prototype maps Article 17 to rule versioning, effective dates, documented model metrics, validation cadence, human oversight, limitation statements, conflict notes and required governance actions. In production, these would sit inside a quality management system with written policies, change management, post-market monitoring, incident reporting and accountability ownership.",
                ),
                (
                    "Why the mapping matters",
                    "EU mode is not merely a stricter threshold. The 0.90 level is Meridian's internal governance trigger. The AI Act obligations are mapped separately through Article 10 data governance, Article 14 human oversight and Article 17 quality-management evidence.",
                ),
            ],
        )
        pdf_page(
            pdf,
            "Task 3 - Operating Controls",
            [
                ("Rule update control", "Each rule has a version and effective date. If a rule changes mid-period, the tool should preserve prior outputs under the old version, apply the new rule prospectively and require sign-off from Legal, Compliance and Model Risk before changing production thresholds."),
                ("Contradictory requirements", "If one jurisdiction permits a lower governance threshold while another requires stricter bias testing, the tool does not average the rules. It selects the active jurisdiction for reporting and flags the conflict to a human governance owner. The default product stance is to preserve the stricter evidence trail where cross-border re-use is plausible."),
                ("Monitoring cadence", "Portfolio fairness metrics should be reviewed monthly, model performance quarterly and rule configuration whenever a material regulatory update occurs. High-risk EU outputs should retain evidence of bias testing, human oversight and documentation review."),
                ("Defensible uncertainty", "The tool is designed to expose uncertainty. It separates statistical disparity, legal compliance, technical explanation and customer-facing adverse-action reason so management cannot pretend that one metric answers every question."),
            ],
        )


def write_model_comparison_pdf(path, model_comparison, rf_importance):
    with PdfPages(path) as pdf:
        fig = plt.figure(figsize=(11, 8.5))
        fig.patch.set_facecolor("#f7f9fb")
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis("off")
        ax.add_patch(plt.Rectangle((0, 0.92), 1, 0.08, transform=ax.transAxes, color="#17324d"))
        ax.text(0.055, 0.955, "Task 3 - Model Benchmark Comparison", transform=ax.transAxes, fontsize=18, fontweight="bold", color="white", va="center")
        ax.text(0.055, 0.885, "Primary model choice is a governance decision, not only an accuracy contest.", transform=ax.transAxes, fontsize=10.5, color="#52616b")

        table_ax = fig.add_axes([0.055, 0.63, 0.89, 0.18])
        table_ax.axis("off")
        display = model_comparison[["model_name", "auc", "accuracy", "test_size", "governance_role"]].copy()
        display["auc"] = display["auc"].map(lambda v: f"{v:.3f}")
        display["accuracy"] = display["accuracy"].map(lambda v: f"{v:.3f}")
        table = table_ax.table(
            cellText=display.values,
            colLabels=["Model", "AUC", "Accuracy", "Test N", "Governance role"],
            loc="center",
            cellLoc="left",
            colColours=["#e8f3f5"] * 5,
            colWidths=[0.20, 0.10, 0.10, 0.10, 0.50],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8.5)
        table.scale(1, 1.5)
        for (row, col), cell in table.get_celld().items():
            cell.set_edgecolor("#adb5bd")
            if row == 0:
                cell.set_text_props(weight="bold", color="#17324d")

        bars_ax = fig.add_axes([0.09, 0.20, 0.42, 0.33])
        top = rf_importance.head(8).sort_values("random_forest_importance")
        bars_ax.barh(top["technical_feature"], top["random_forest_importance"], color="#0b7285")
        bars_ax.set_title("Random Forest challenger: top feature importances", fontsize=11, fontweight="bold", color="#17324d")
        bars_ax.set_xlabel("Mean impurity importance")
        bars_ax.tick_params(axis="y", labelsize=8)
        bars_ax.grid(axis="x", alpha=0.25)

        text_ax = fig.add_axes([0.56, 0.20, 0.37, 0.36])
        text_ax.axis("off")
        comparison_text = (
            "Interpretation:\n"
            "- Logistic regression is kept as the primary model because it gives stable, signed feature contributions that can be mapped to consumer-facing adverse-action reasons.\n"
            "- Random Forest is included as a challenger to test non-linear predictive structure. It can improve modelling flexibility, but its native feature importances are less suitable for individual consumer notices.\n"
            "- A production version could add TreeSHAP for the challenger model, but the submitted governance workflow deliberately prioritizes explainability, auditability and reason quality."
        )
        add_wrapped(text_ax, comparison_text, 0.02, 0.95, width=58, size=9.5, color="#1f2933", line_gap=0.065)

        ax.text(0.055, 0.09, "Conclusion: the non-linear benchmark strengthens technical diligence, while the chosen primary model remains aligned with fair-lending explanation duties.", transform=ax.transAxes, fontsize=9.5, color="#1f2933")
        ax.text(0.055, 0.045, f"{STUDENT['name']} | {STUDENT['matriculation_id']}", transform=ax.transAxes, fontsize=8.5, color="#6b7280")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def write_technical_appendix_pdf(path, metrics, sensitivity, model_metrics, rf_metrics):
    us = metrics[metrics["active_jurisdiction"].eq("US")].iloc[0]
    eu = metrics[metrics["active_jurisdiction"].eq("EU")].iloc[0]
    with PdfPages(path) as pdf:
        pdf_page(
            pdf,
            "Task 3 - Technical Appendix",
            [
                (
                    "Data generation",
                    "The prototype generates 5,000 synthetic credit applications with applicant features and a synthetic default outcome. It then trains the primary logistic-regression model and uses the trained model's predict_proba output as model_score. Approval decisions, manual-review flags and ranked decision reasons are derived after scoring. The data is intentionally synthetic because real cross-border credit data with protected attributes is commercially sensitive and usually unavailable for coursework.",
                ),
                (
                    "Risk logic",
                    "Default risk increases when credit score is lower, debt-to-income is higher, income is lower, prior default is present, regional portfolio risk is higher and employment is weaker. A mild controlled protected-group score shift is included so the tool can demonstrate fairness monitoring without inventing an extreme or unrealistic disparity.",
                ),
                (
                    "Model implementation",
                    f"The generation script trains a logistic-regression primary model and a Random Forest challenger. The application-level model_score column is the trained logistic model's predicted default probability, not a synthetic pre-model score. Logistic performance: AUC {model_metrics['auc']:.3f}, accuracy {model_metrics['accuracy']:.3f}. Random Forest performance: AUC {rf_metrics['auc']:.3f}, accuracy {rf_metrics['accuracy']:.3f}. Logistic regression remains the primary governance model because its feature contributions can be mapped cleanly to adverse-action reasons.",
                ),
            ],
            subtitle=f"{STUDENT['name']} | {STUDENT['matriculation_id']} | {STUDENT['email']}",
        )
        pdf_page(
            pdf,
            "Task 3 - Metric Formulas",
            [
                (
                    "Approval parity ratio",
                    f"approval_parity_ratio = approval_rate(protected group) / approval_rate(reference group). In the generated portfolio, protected-group approval is {us['protected_group_approval_rate']:.4f}, reference-group approval is {us['reference_group_approval_rate']:.4f} and approval parity is {us['approval_parity_ratio']:.4f}.",
                ),
                (
                    "False positive rate by group",
                    "In this credit context, a false positive means the tool rejects or flags a low-risk applicant who did not default in the synthetic outcome. The group-level false positive rate checks whether one group bears more erroneous denials or review burdens.",
                ),
                (
                    "False negative rate by group",
                    "A false negative means the model approves an applicant who later defaults in the synthetic outcome. This metric matters because a tool that only optimizes consumer approvals may ignore bank safety and soundness risk.",
                ),
                (
                    "FNR tension in the generated data",
                    f"The protected-group FNR is {us['protected_group_false_negative_rate']:.4f}, while the reference-group FNR is {us['reference_group_false_negative_rate']:.4f}. This means the protected group has fewer actual defaulters incorrectly approved, but also faces a lower approval rate overall. The governance question is therefore not simply whether one group is more risky; it is whether the model has become more conservative for that group in a way that shifts consumer burden.",
                ),
                (
                    "Threshold method",
                    f"US mode compares the observed parity {us['approval_parity_ratio']:.3f} with threshold {us['threshold']:.2f}. EU mode compares the same observed parity with threshold {eu['threshold']:.2f}. The EU 0.90 threshold is not a statutory threshold; it is Meridian's internal conservative escalation level. Legal obligations are mapped separately through Article 10, Article 14 and Article 17 controls.",
                ),
            ],
        )
        pdf_page(
            pdf,
            "Task 3 - Jurisdiction Logic",
            [
                (
                    "US output",
                    "US mode focuses on adverse-action reason quality and internal fair-lending risk evidence. The tool is careful not to say that a statistical disparity is automatically a legal violation, especially because the 2026 CFPB Regulation B final rule changes the treatment of disparate impact. This is why the US output says monitoring or review rather than legal conclusion.",
                ),
                (
                    "EU output",
                    "EU mode treats AI creditworthiness evaluation as high-risk AI governance. The tool therefore emphasizes bias testing, documentation, human oversight and remediation ownership. This output is not just the US report translated into EU language; it is a different governance package.",
                ),
                (
                    "EU AI Act Article 10, 14 and 17 mapping",
                    "Article 10 is implemented as data-governance evidence: data suitability, feature definitions, jurisdiction fields, protected-group monitoring, bias metrics and sensitivity checks. Article 14 is implemented as reviewer sign-off and human-oversight evidence. Article 17 is implemented as quality-management evidence: rule versions, effective dates, validation results, conflict notes, limitation records and monitoring cadence.",
                ),
                (
                    "Contradictory rules",
                    "If two jurisdictions impose different expectations, the prototype does not average them. It uses the active jurisdiction for the formal output and records a conflict note so a human governance owner can decide whether the stricter evidence trail should be retained for cross-border re-use.",
                ),
                (
                    "Why Singapore and UK appear in the rule file",
                    "The submitted jurisdiction_rules.csv includes Singapore and UK rows only to demonstrate that the rule engine is extensible. The assignment's substantive comparison is US versus EU. Singapore and UK are therefore not analyzed in depth, and their rows should be treated as future-work configuration examples rather than additional research claims.",
                ),
            ],
        )
        selected_lines = []
        for shift in [0.00, 0.02, 0.04]:
            sub = sensitivity[np.isclose(sensitivity["protected_group_score_shift"], shift)]
            us_row = sub[sub["active_jurisdiction"].eq("US")].iloc[0]
            eu_row = sub[sub["active_jurisdiction"].eq("EU")].iloc[0]
            selected_lines.append(
                f"Shift {shift:.2f}: parity {us_row['approval_parity_ratio']:.4f}; "
                f"US flag {bool(us_row['fairness_flag'])}; EU flag {bool(eu_row['fairness_flag'])}."
            )
        selected = "\n".join(selected_lines)
        pdf_page(
            pdf,
            "Task 3 - Sensitivity Result",
            [
                (
                    "Sensitivity result",
                    "The sensitivity test shifts protected-group model scores and recalculates parity under US and EU thresholds. Selected scenarios:\n" + selected,
                ),
            ],
        )
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis("off")
        ax.add_patch(plt.Rectangle((0, 0.94), 1, 0.06, transform=ax.transAxes, color="#17324d", zorder=-1))
        ax.text(0.04, 0.91, "Task 3 - Data Dictionary Extract", transform=ax.transAxes, fontsize=21, fontweight="bold", color="#17324d", va="top")
        ax.text(0.04, 0.85, "The full field-level dictionary is exported as data/Task3_data_dictionary.csv. This page highlights the production-risk fields a reviewer should challenge first.", transform=ax.transAxes, fontsize=10.5, color="#52616b", va="top")
        dictionary = data_dictionary()
        display = dictionary.loc[
            dictionary["field_name"].isin(["protected_group", "credit_score", "region_risk_score", "model_score"]),
            ["field_name", "used_in_model", "protected_or_proxy", "production_credit_model_note"],
        ].copy()
        display["production_credit_model_note"] = display["production_credit_model_note"].map(lambda text: wrap_cell(text, 38))
        table = ax.table(
            cellText=display.values,
            colLabels=["Field", "Model input?", "Protected/proxy?", "Production stance"],
            cellLoc="left",
            colLoc="left",
            bbox=[0.055, 0.31, 0.89, 0.39],
            colWidths=[0.17, 0.12, 0.18, 0.53],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(7.8)
        table.scale(1, 1.9)
        for (row, col), cell in table.get_celld().items():
            cell.set_edgecolor("#adb5bd")
            if row == 0:
                cell.set_facecolor("#d0ebff")
                cell.set_text_props(weight="bold", color="#17324d")
            elif col == 0:
                cell.set_facecolor("#f8f9fa")
                cell.set_text_props(weight="bold")
        ax.add_patch(plt.Rectangle((0.055, 0.16), 0.89, 0.08, transform=ax.transAxes, facecolor="#fff9db", edgecolor="#ffe066", linewidth=1))
        ax.text(0.075, 0.205, "Audit note", transform=ax.transAxes, fontsize=10, fontweight="bold", color="#17324d", va="center")
        ax.text(0.18, 0.205, wrap_cell("Protected attributes and proxies are used for monitoring and explanation governance, not as production credit-model inputs unless separately approved.", 86), transform=ax.transAxes, fontsize=8.5, color="#1f2933", va="center", linespacing=1.2)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        pdf_page(
            pdf,
            "Task 3 - Limits and Production Hardening",
            [
                (
                    "What the prototype does not prove",
                    "It does not prove Citi's actual compliance posture, does not use real customer data, does not provide legal advice, does not decide credit applications autonomously and does not automatically repair model bias. These limits are not excuses; they are part of the governance design.",
                ),
                (
                    "Production hardening needed",
                    "A real deployment would need validated source systems, model inventory integration, access control, audit logs, legal-change workflow, adverse-action notice review, drift monitoring, appeal workflow and documented accountability for overrides.",
                ),
                (
                    "Why the limitation is acceptable for this assignment",
                    "The assignment asks for a working prototype or mock-up that demonstrates jurisdiction-aware logic, not a production bank platform. The submitted tool therefore prioritizes transparent regulatory mechanics, reproducible synthetic data and honest boundaries over unsupported claims of deployment readiness.",
                ),
            ],
        )


def write_sensitivity_pdf(path, sensitivity):
    with PdfPages(path) as pdf:
        us = sensitivity[sensitivity["active_jurisdiction"] == "US"]
        eu = sensitivity[sensitivity["active_jurisdiction"] == "EU"]
        us_threshold = us["threshold"].iloc[0]
        eu_threshold = eu["threshold"].iloc[0]
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis("off")
        ax.add_patch(plt.Rectangle((0, 0.955), 1, 0.045, transform=ax.transAxes, color="#17324d", zorder=-1))
        ax.text(0.055, 0.925, "Task 3 - Sensitivity Analysis", transform=ax.transAxes, fontsize=20, fontweight="bold", color="#17324d", va="top")
        y = 0.86
        y = add_wrapped(
            ax,
            "This page tests how much the conclusion changes if the protected-group score distribution shifts. A positive shift means the model assigns slightly higher risk scores to protected-group applicants, lowering approval rates. The key result is that EU mode escalates earlier because Meridian uses a stricter internal high-risk AI governance threshold.",
            0.055,
            y,
            width=88,
            size=10.2,
            line_gap=0.038,
        )
        plot_ax = fig.add_axes([0.12, 0.43, 0.76, 0.28])
        plot_ax.plot(us["protected_group_score_shift"], us["approval_parity_ratio"], marker="o", label="US mode")
        plot_ax.plot(eu["protected_group_score_shift"], eu["approval_parity_ratio"], marker="o", label="EU mode")
        plot_ax.axhline(us_threshold, color="#1f77b4", linestyle="--", linewidth=1, label=f"US threshold {us_threshold:.2f}")
        plot_ax.axhline(eu_threshold, color="#d62728", linestyle="--", linewidth=1, label=f"EU threshold {eu_threshold:.2f}")
        plot_ax.set_xlabel("Protected-group model-score shift")
        plot_ax.set_ylabel("Approval parity ratio")
        plot_ax.set_title("Sensitivity of Governance Conclusion")
        plot_ax.grid(True, alpha=0.25)
        plot_ax.legend(fontsize=8)
        y = 0.35
        table_text = sensitivity[sensitivity["protected_group_score_shift"].isin([0.0, 0.02, 0.04])][
            ["protected_group_score_shift", "active_jurisdiction", "approval_parity_ratio", "threshold", "fairness_flag"]
        ].to_string(index=False)
        ax.text(0.055, y, "Selected scenarios", transform=ax.transAxes, fontsize=12, fontweight="bold", color="#0b7285", va="top")
        ax.text(0.055, y - 0.04, table_text, transform=ax.transAxes, fontsize=8.5, family="monospace", color="#1f2933", va="top")
        ax.text(
            0.055,
            0.055,
            "Interpretation: the conclusion is not a fragile one-line claim. The tool makes the threshold choice visible, so management can see when a small distribution shift changes the required governance action.",
            transform=ax.transAxes,
            fontsize=9,
            color="#52616b",
            va="bottom",
        )
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        pdf_page(
            pdf,
            "Task 3 - Sensitivity Interpretation",
            [
                (
                    "Why this matters",
                    "A fairness conclusion should not depend on a hidden modeling assumption. The sensitivity test makes the threshold boundary visible by changing the protected-group score distribution in small increments and recalculating the approval parity ratio. This is not meant to predict a real macroeconomic shock; it is a governance stress test that asks whether the conclusion is robust near the rule threshold.",
                ),
                (
                    "What changes across jurisdictions",
                    f"At the observed parity ratio of {us[us['protected_group_score_shift'].eq(0.0)]['approval_parity_ratio'].iloc[0]:.3f}, US mode is compared with the internal {us_threshold:.2f} review threshold, while EU mode is compared with Meridian's internal {eu_threshold:.2f} high-risk AI governance threshold. The EU threshold is not statutory; it is a conservative escalation setting attached to separate Article 10, 14 and 17 controls.",
                ),
                (
                    "Management implication",
                    "The result tells senior management that the product should not present a single global fairness conclusion. The same model output requires different governance evidence, escalation language and human oversight depending on the active jurisdiction. A future production version should automate this sensitivity test monthly and attach the result to model-risk committee materials.",
                ),
                (
                    "Limitation of the sensitivity analysis",
                    "The analysis changes model scores synthetically rather than retraining the model on new real-world data. It is useful for governance reasoning, but not a substitute for production drift monitoring, challenger models, legal review or customer remediation analysis.",
                ),
            ],
        )


def write_dashboard_pdf(path, metrics, df, outputs, sensitivity):
    with PdfPages(path) as pdf:
        fig = plt.figure(figsize=(11, 8.5))
        fig.patch.set_facecolor("white")
        title = fig.add_axes([0, 0, 1, 1])
        title.axis("off")
        title.add_patch(plt.Rectangle((0, 0.94), 1, 0.06, transform=title.transAxes, color="#17324d", zorder=-1))
        title.text(0.04, 0.915, "Task 3 - Dashboard Mockup", fontsize=22, fontweight="bold", color="#17324d", va="top")
        title.text(0.04, 0.875, "Cross-Border AI Credit Risk & Fairness Governance Tool", fontsize=11, color="#52616b", va="top")

        ax1 = fig.add_axes([0.07, 0.55, 0.38, 0.27])
        approval = metrics[metrics["active_jurisdiction"].eq("US")].iloc[0]
        ax1.bar(["Protected", "Reference"], [approval["protected_group_approval_rate"], approval["reference_group_approval_rate"]], color=["#0b7285", "#74b816"])
        ax1.set_ylim(0, 1)
        ax1.set_title("Approval Rates by Group")
        ax1.set_ylabel("Approval rate")
        ax1.grid(axis="y", alpha=0.25)

        ax2 = fig.add_axes([0.56, 0.55, 0.36, 0.27])
        m = metrics[metrics["active_jurisdiction"].isin(["US", "EU"])]
        x = np.arange(len(m))
        ax2.bar(x - 0.15, m["approval_parity_ratio"], width=0.3, label="Observed parity", color="#0b7285")
        ax2.bar(x + 0.15, m["threshold"], width=0.3, label="Threshold", color="#f08c00")
        ax2.set_xticks(x, m["active_jurisdiction"])
        ax2.set_ylim(0.65, 1.0)
        ax2.set_title("Jurisdiction-Specific Thresholds")
        ax2.legend(fontsize=8)
        ax2.grid(axis="y", alpha=0.25)

        ax3 = fig.add_axes([0.07, 0.17, 0.38, 0.27])
        rejected = outputs[outputs["approved"].eq(0)]["decision_reason_1"].value_counts().head(5)
        ax3.barh(rejected.index[::-1], rejected.values[::-1], color="#5c7cfa")
        ax3.set_title("Top Adverse-Action Reasons")
        ax3.tick_params(axis="y", labelsize=8)

        ax4 = fig.add_axes([0.56, 0.17, 0.36, 0.27])
        eu = sensitivity[sensitivity["active_jurisdiction"].eq("EU")]
        eu_threshold = eu["threshold"].iloc[0]
        ax4.plot(eu["protected_group_score_shift"], eu["approval_parity_ratio"], marker="o", color="#d6336c")
        ax4.axhline(eu_threshold, linestyle="--", color="#d6336c", linewidth=1)
        ax4.set_title("EU Sensitivity Test")
        ax4.set_xlabel("Protected-group score shift")
        ax4.set_ylabel("Approval parity")
        ax4.grid(True, alpha=0.25)

        title.text(
            0.04,
            0.08,
            "Management interpretation: US mode continues monitoring at the current parity level; EU mode escalates because the same parity falls below Meridian's stricter internal high-risk AI governance threshold.",
            fontsize=10,
            color="#1f2933",
            va="bottom",
        )
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def write_shap_pdf(path, shap_df):
    with PdfPages(path) as pdf:
        for applicant_id in shap_df["applicant_id"].unique():
            fig = plt.figure(figsize=(8.5, 11))
            fig.patch.set_facecolor("white")
            fig.add_artist(plt.Rectangle((0, 0.94), 1, 0.06, transform=fig.transFigure, color="#17324d", zorder=-1))
            fig.text(0.06, 0.958, "Task 3 - SHAP Explainability", fontsize=18, fontweight="bold", color="white", va="center")
            sub = shap_df[shap_df["applicant_id"].eq(applicant_id)].sort_values("rank")
            first = sub.iloc[0]
            reasons = " / ".join(sub["mapped_consumer_reason"].tolist())
            threshold = APPROVAL_THRESHOLD
            summary_ax = fig.add_axes([0.06, 0.79, 0.88, 0.12])
            summary_ax.axis("off")
            summary_ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=summary_ax.transAxes, facecolor="#f8f9fa", edgecolor="#ced4da", linewidth=1))
            summary_ax.text(0.025, 0.78, f"{applicant_id}", transform=summary_ax.transAxes, fontsize=12, fontweight="bold", color="#17324d", va="center")
            summary_ax.text(0.025, 0.52, wrap_cell(reasons, 92), transform=summary_ax.transAxes, fontsize=8.8, color="#1f2933", va="center")
            summary_ax.text(0.025, 0.18, f"Model score: {first['model_score']:.4f}", transform=summary_ax.transAxes, fontsize=9.5, fontweight="bold", color="#d6336c", va="center")
            summary_ax.text(0.32, 0.18, f"Reject if score >= {threshold:.2f}", transform=summary_ax.transAxes, fontsize=9, color="#1f2933", va="center")
            summary_ax.text(0.62, 0.18, "Status: rejected", transform=summary_ax.transAxes, fontsize=9, color="#1f2933", va="center")

            ax = fig.add_axes([0.18, 0.52, 0.68, 0.22])
            plot_sub = sub.sort_values("linear_shap_log_odds")
            ax.barh(plot_sub["technical_feature"], plot_sub["linear_shap_log_odds"], height=0.34, color="#d6336c")
            ax.axvline(0, color="#495057", linewidth=0.8)
            ax.set_xlabel("Linear SHAP contribution to default log-odds")
            ax.tick_params(axis="y", labelsize=9)
            ax.grid(axis="x", alpha=0.25)
            ax.set_title("Technical feature contribution", fontsize=10, loc="left", pad=8)

            ax_tbl = fig.add_axes([0.08, 0.25, 0.84, 0.20])
            ax_tbl.axis("off")
            table_rows = [
                [
                    wrap_cell(item.mapped_consumer_reason, 30),
                    wrap_cell(item.technical_feature, 24),
                    f"+{item.linear_shap_log_odds:.4f}",
                ]
                for item in sub.sort_values("rank").itertuples()
            ]
            table = ax_tbl.table(
                cellText=table_rows,
                colLabels=["Consumer-Facing Reason", "Technical Feature", "SHAP Log-Odds"],
                cellLoc="left",
                colLoc="left",
                colWidths=[0.45, 0.33, 0.18],
                bbox=[0, 0, 1, 1],
            )
            table.auto_set_font_size(False)
            table.set_fontsize(8.0)
            table.scale(1, 1.25)
            for (row, col), cell in table.get_celld().items():
                cell.set_edgecolor("#adb5bd")
                cell.set_linewidth(0.7)
                if row == 0:
                    cell.set_facecolor("#d0ebff")
                    cell.set_text_props(weight="bold")

            note_ax = fig.add_axes([0.08, 0.08, 0.84, 0.12])
            note_ax.axis("off")
            note_ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=note_ax.transAxes, facecolor="#fff9db", edgecolor="#ffe066", linewidth=1))
            note_ax.text(0.025, 0.72, "How to read this page", transform=note_ax.transAxes, fontsize=9.5, fontweight="bold", color="#1f2933", va="center")
            note_ax.text(0.025, 0.47, "Consumer-facing reason <- technical feature <- positive SHAP-style contribution.", transform=note_ax.transAxes, fontsize=8.7, color="#1f2933", va="center")
            note_ax.text(0.025, 0.25, "Model link: baseline log-odds + feature contributions -> applicant log-odds -> default score.", transform=note_ax.transAxes, fontsize=8.4, color="#1f2933", va="center")
            note_ax.text(0.025, 0.06, "Lower-risk features are omitted from rejection explanations; these are explanation evidence, not legal advice.", transform=note_ax.transAxes, fontsize=8.2, color="#52616b", va="bottom")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
        pdf_page(
            pdf,
            "Task 3 - SHAP Methodology Note",
            [
                (
                    "What is being explained",
                    "The SHAP-style pages explain why selected rejected applicants received high predicted default-risk scores. They connect the model's technical feature contributions to consumer-facing adverse-action reason categories.",
                ),
                (
                    "Why only positive contributions are shown",
                    "For an adverse-action explanation, the relevant features are those that pushed the decision toward rejection. Features that lowered default risk are useful for model debugging, but including them in a rejection-reason table would confuse the explanation given to management or consumers.",
                ),
                (
                    "How technical explanation becomes governance evidence",
                    "The technical feature contribution is not copied directly into a consumer notice. It is mapped into a clearer reason category, such as high debt-to-income ratio or low credit score. Compliance and legal reviewers would still need to validate the final notice language.",
                ),
                (
                    "Boundary",
                    "SHAP-style explanations improve transparency, but they do not prove legal compliance and do not replace human review. The tool treats them as evidence for governance and adverse-action reason quality, not as automatic legal conclusions.",
                ),
            ],
        )
        fig, ax = plt.subplots(figsize=(11.8, 8.5))
        ax.axis("off")
        ax.add_patch(plt.Rectangle((0, 0.94), 1, 0.06, transform=ax.transAxes, color="#17324d", zorder=-1))
        ax.text(0.04, 0.91, "Selected Top Contributions and Adverse-Action Mapping", transform=ax.transAxes, fontsize=18, fontweight="bold", color="#17324d", va="top")
        ax.text(
            0.04,
            0.855,
            "Only positive SHAP-style contributions are shown, because the purpose is to explain factors that pushed the decision toward rejection.",
            transform=ax.transAxes,
            fontsize=9.5,
            color="#52616b",
            va="top",
        )
        table_rows = []
        for applicant_id, sub in shap_df.groupby("applicant_id", sort=False):
            first_row = True
            for item in sub.sort_values("rank").itertuples():
                table_rows.append(
                    [
                        applicant_id if first_row else "",
                        wrap_cell(item.mapped_consumer_reason, 26),
                        wrap_cell(item.technical_feature, 24),
                        f"+{item.linear_shap_log_odds:.4f}",
                        "Increases\nrisk",
                    ]
                )
                first_row = False
        col_labels = [
            "Applicant ID",
            "Consumer-Facing Reason",
            "Technical Feature",
            "SHAP Log-Odds",
            "Direction",
        ]
        table = ax.table(
            cellText=table_rows,
            colLabels=col_labels,
            cellLoc="left",
            colLoc="left",
            colWidths=[0.11, 0.33, 0.23, 0.14, 0.14],
            bbox=[0.035, 0.19, 0.93, 0.60],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(7.7)
        table.scale(1, 1.45)
        for (row, col), cell in table.get_celld().items():
            cell.set_edgecolor("#adb5bd")
            cell.set_linewidth(0.7)
            if row == 0:
                cell.set_facecolor("#d0ebff")
                cell.set_text_props(weight="bold")
                cell.set_linewidth(1.1)
            elif col == 0 and cell.get_text().get_text():
                cell.set_facecolor("#f8f9fa")
                cell.set_text_props(weight="bold")
        ax.text(
            0.04,
            0.095,
            "Mapping logic: technical feature contributions support explanation, while the consumer-facing reason is the clearer adverse-action category. Lower-risk features are intentionally excluded from this table to avoid confusing the rejection explanation.",
            transform=ax.transAxes,
            fontsize=9,
            color="#1f2933",
            va="top",
        )
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def wrap_box_text(text, width):
    lines = []
    for para in str(text).split("\n"):
        if para.strip():
            lines.extend(
                textwrap.wrap(
                    para,
                    width=width,
                    break_long_words=False,
                    break_on_hyphens=False,
                )
            )
        else:
            lines.append("")
    return "\n".join(lines)


def wrap_cell(text, width):
    return wrap_box_text(text, width)


def deck_problem_page(pdf, metrics):
    us = metrics[metrics["active_jurisdiction"].eq("US")].iloc[0]
    eu = metrics[metrics["active_jurisdiction"].eq("EU")].iloc[0]
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis("off")
    ax.add_patch(plt.Rectangle((0, 0.94), 1, 0.06, transform=ax.transAxes, color="#17324d", zorder=-1))
    ax.text(0.04, 0.91, "1. Problem", transform=ax.transAxes, fontsize=23, fontweight="bold", color="#17324d", va="top")
    ax.text(0.04, 0.855, "One trained credit model can create different governance duties in different jurisdictions.", transform=ax.transAxes, fontsize=12, color="#52616b", va="top")

    cards = [
        (0.050, "Input", "Trained logistic score", "predict_proba becomes model_score", "#e7f5ff"),
        (0.365, "US lens", f"{us['threshold']:.2f} warning level", "Adverse-action reasons + monitoring", "#e6fcf5"),
        (0.680, "EU lens", f"{eu['threshold']:.2f} internal escalation", "Article 10/14/17 evidence package", "#fff4e6"),
    ]
    for x, label, headline, body, color in cards:
        ax.add_patch(plt.Rectangle((x, 0.55), 0.27, 0.20, transform=ax.transAxes, facecolor=color, edgecolor="#495057", linewidth=1.1))
        ax.text(x + 0.02, 0.715, label.upper(), transform=ax.transAxes, fontsize=8.5, fontweight="bold", color="#52616b", va="center")
        ax.text(x + 0.02, 0.650, wrap_box_text(headline, 21), transform=ax.transAxes, fontsize=11.5, fontweight="bold", color="#17324d", va="center", linespacing=1.12)
        ax.text(x + 0.02, 0.590, wrap_box_text(body, 27), transform=ax.transAxes, fontsize=8.8, color="#1f2933", va="center", linespacing=1.25)
    for start in [0.320, 0.635]:
        ax.annotate("", xy=(start + 0.040, 0.65), xytext=(start, 0.65), xycoords=ax.transAxes, arrowprops=dict(arrowstyle="->", lw=1.6, color="#6b7280"))

    ax.add_patch(plt.Rectangle((0.055, 0.205), 0.89, 0.22, transform=ax.transAxes, facecolor="#f8f9fa", edgecolor="#ced4da", linewidth=1.0))
    ax.text(0.075, 0.375, "Why this cannot be a memo or static spreadsheet", transform=ax.transAxes, fontsize=12, fontweight="bold", color="#17324d", va="center")
    bullets = [
        "rules need effective dates and versions",
        "same metric needs jurisdiction-specific interpretation",
        "explanation, fairness and oversight evidence must stay linked to model outputs",
    ]
    bullet_y = 0.335
    for bullet in bullets:
        wrapped = wrap_box_text(f"- {bullet}", 86)
        ax.text(0.095, bullet_y, wrapped, transform=ax.transAxes, fontsize=9.2, color="#1f2933", va="top", linespacing=1.25)
        bullet_y -= 0.035 * len(wrapped.splitlines()) + 0.014
    ax.text(0.04, 0.06, f"{STUDENT['name']} | {STUDENT['matriculation_id']} | Senior management deck", transform=ax.transAxes, fontsize=8.5, color="#6b7280")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def deck_citi_meridian_page(pdf):
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis("off")
    ax.add_patch(plt.Rectangle((0, 0.94), 1, 0.06, transform=ax.transAxes, color="#17324d", zorder=-1))
    ax.text(0.04, 0.91, "2. Why Citi, Why Meridian", transform=ax.transAxes, fontsize=23, fontweight="bold", color="#17324d", va="top")
    ax.text(0.04, 0.855, "A realistic cross-border bank case matched to a deliberately narrow RegTech product.", transform=ax.transAxes, fontsize=12, color="#52616b", va="top")
    columns = [
        (0.06, "Why Citi", "#e7f5ff", [
            "US-headquartered global bank",
            "public European operating footprint",
            "credit governance is plausible but not alleged",
        ]),
        (0.53, "Why Meridian", "#e6fcf5", [
            "25-40 person specialist RegTech",
            "explainability + model-risk governance niche",
            "enterprise SaaS plus implementation support",
        ]),
    ]
    for x, title, color, bullets in columns:
        ax.add_patch(plt.Rectangle((x, 0.31), 0.40, 0.41, transform=ax.transAxes, facecolor=color, edgecolor="#495057", linewidth=1.1))
        ax.text(x + 0.03, 0.66, title, transform=ax.transAxes, fontsize=16, fontweight="bold", color="#17324d", va="center")
        bullet_y = 0.60
        for bullet in bullets:
            wrapped = wrap_box_text(f"- {bullet}", 36)
            ax.text(x + 0.05, bullet_y, wrapped, transform=ax.transAxes, fontsize=9.7, color="#1f2933", va="top", linespacing=1.25)
            bullet_y -= 0.074 + 0.026 * (len(wrapped.splitlines()) - 1)
    ax.add_patch(plt.Rectangle((0.18, 0.13), 0.64, 0.105, transform=ax.transAxes, facecolor="#fff9db", edgecolor="#ffe066", linewidth=1.0))
    ax.text(
        0.50,
        0.183,
        wrap_box_text("Boundary: this is a plausible design case, not proof of Citi's actual model inventory or compliance posture.", 74),
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=9.0,
        color="#1f2933",
        linespacing=1.25,
    )
    ax.text(0.04, 0.06, f"{STUDENT['name']} | {STUDENT['matriculation_id']} | Senior management deck", transform=ax.transAxes, fontsize=8.5, color="#6b7280")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def deck_meridian_value_page(pdf):
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis("off")
    ax.add_patch(plt.Rectangle((0, 0.94), 1, 0.06, transform=ax.transAxes, color="#17324d", zorder=-1))
    ax.text(0.04, 0.91, "3. Why The Product Fits Meridian", transform=ax.transAxes, fontsize=23, fontweight="bold", color="#17324d", va="top")
    ax.text(
        0.04,
        0.855,
        wrap_box_text("The product is valuable because it converts regulatory divergence into repeatable evidence, not because it promises legal certainty.", 112),
        transform=ax.transAxes,
        fontsize=11.2,
        color="#52616b",
        va="top",
        linespacing=1.2,
    )
    rows = [
        ("Explainability", "linear contributions -> ranked adverse-action reasons", "#e7f5ff"),
        ("Fairness monitoring", "approval parity + FPR/FNR + sensitivity tests", "#fff9db"),
        ("Rule engineering", "versioned thresholds + conflict notes + owners", "#f3f0ff"),
        ("Human judgement", "review bands + sign-off logs + no auto-remediation", "#ffe3e3"),
    ]
    for i, (title, body, color) in enumerate(rows):
        y = 0.68 - i * 0.13
        ax.add_patch(plt.Rectangle((0.08, y), 0.84, 0.095, transform=ax.transAxes, facecolor=color, edgecolor="#adb5bd", linewidth=1.0))
        ax.text(0.11, y + 0.048, title, transform=ax.transAxes, fontsize=12, fontweight="bold", color="#17324d", va="center")
        ax.text(0.39, y + 0.048, wrap_box_text(body, 62), transform=ax.transAxes, fontsize=9.7, color="#1f2933", va="center", linespacing=1.15)
    ax.text(
        0.50,
        0.12,
        wrap_box_text("Commercial logic: sell a narrow, auditable product to global banks that must monitor model and rule changes continuously.", 94),
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=9.4,
        fontweight="bold",
        color="#0b7285",
        linespacing=1.2,
    )
    ax.text(0.04, 0.06, f"{STUDENT['name']} | {STUDENT['matriculation_id']} | Senior management deck", transform=ax.transAxes, fontsize=8.5, color="#6b7280")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def deck_architecture_page(pdf):
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis("off")
    ax.add_patch(plt.Rectangle((0, 0.94), 1, 0.06, transform=ax.transAxes, color="#17324d", zorder=-1))
    ax.text(0.04, 0.91, "4. Architecture View", transform=ax.transAxes, fontsize=22, fontweight="bold", color="#17324d", va="top")
    ax.text(0.04, 0.865, "Where automation stops and accountable judgement begins", transform=ax.transAxes, fontsize=11, color="#52616b", va="top")
    boxes = [
        (0.05, 0.62, "Synthetic credit\napplications", "#e7f5ff"),
        (0.27, 0.62, "Credit risk model\n+ rejection reasons", "#e6fcf5"),
        (0.49, 0.62, "Fairness monitor\nparity / FPR / FNR", "#fff9db"),
        (0.71, 0.62, "Jurisdiction rule\nengine", "#f3f0ff"),
        (0.38, 0.28, "Governance report\nUS / EU outputs", "#fff4e6"),
        (0.66, 0.28, "Human review\nCCO / CRO / AI owner", "#ffe3e3"),
    ]
    for x, y, label, color in boxes:
        ax.add_patch(plt.Rectangle((x, y), 0.18, 0.14, transform=ax.transAxes, facecolor=color, edgecolor="#495057", linewidth=1.2))
        ax.text(
            x + 0.09,
            y + 0.07,
            wrap_box_text(label, 18),
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=9.2,
            fontweight="bold",
            color="#1f2933",
            linespacing=1.12,
        )
    arrows = [
        ((0.23, 0.69), (0.27, 0.69)),
        ((0.45, 0.69), (0.49, 0.69)),
        ((0.67, 0.69), (0.71, 0.69)),
        ((0.80, 0.62), (0.52, 0.42)),
        ((0.58, 0.35), (0.66, 0.35)),
    ]
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start, xycoords=ax.transAxes, arrowprops=dict(arrowstyle="->", lw=1.6, color="#495057"))
    ax.text(
        0.05,
        0.13,
        wrap_box_text("Design principle: the tool calculates, records and escalates; it does not make the final credit-policy or legal-compliance judgement.", 122),
        transform=ax.transAxes,
        fontsize=9.6,
        color="#1f2933",
        linespacing=1.2,
    )
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def deck_meridian_fit_page(pdf):
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis("off")
    ax.add_patch(plt.Rectangle((0, 0.94), 1, 0.06, transform=ax.transAxes, color="#17324d", zorder=-1))
    ax.text(0.04, 0.91, "Why This Product Fits Meridian", transform=ax.transAxes, fontsize=22, fontweight="bold", color="#17324d", va="top")
    ax.text(0.04, 0.86, "The product is attractive from the seller's perspective because it sits exactly where Meridian is strongest.", transform=ax.transAxes, fontsize=10.5, color="#52616b", va="top")
    cards = [
        (0.07, 0.56, "Explainability", "SHAP-style evidence\nAdverse-action mapping\nConsumer-facing reasons", "#e7f5ff"),
        (0.39, 0.56, "Fairness Monitoring", "Approval parity\nFPR / FNR tension\nSensitivity testing", "#e6fcf5"),
        (0.71, 0.56, "Rule Engineering", "Jurisdiction thresholds\nVersioned effective dates\nConflict notes", "#fff9db"),
    ]
    for x, y, title, body, color in cards:
        ax.add_patch(plt.Rectangle((x, y), 0.23, 0.20, transform=ax.transAxes, facecolor=color, edgecolor="#495057", linewidth=1.2))
        ax.text(x + 0.115, y + 0.155, title, transform=ax.transAxes, ha="center", va="center", fontsize=12, fontweight="bold", color="#17324d")
        ax.text(x + 0.115, y + 0.075, body, transform=ax.transAxes, ha="center", va="center", fontsize=9, color="#1f2933", linespacing=1.45)
    ax.annotate("", xy=(0.34, 0.66), xytext=(0.30, 0.66), xycoords=ax.transAxes, arrowprops=dict(arrowstyle="->", lw=1.5, color="#495057"))
    ax.annotate("", xy=(0.66, 0.66), xytext=(0.62, 0.66), xycoords=ax.transAxes, arrowprops=dict(arrowstyle="->", lw=1.5, color="#495057"))
    ax.add_patch(plt.Rectangle((0.16, 0.25), 0.68, 0.15, transform=ax.transAxes, facecolor="#f8f9fa", edgecolor="#adb5bd", linewidth=1.0))
    ax.text(0.50, 0.335, "Strategic fit", transform=ax.transAxes, ha="center", va="center", fontsize=13, fontweight="bold", color="#17324d")
    ax.text(0.50, 0.285, "A narrow, auditable AI credit governance product is more defensible for Meridian than a broad consulting platform.", transform=ax.transAxes, ha="center", va="center", fontsize=9.5, color="#1f2933")
    ax.text(0.04, 0.075, f"{STUDENT['name']} | {STUDENT['matriculation_id']} | Seller-side fit visual", transform=ax.transAxes, fontsize=8.5, color="#6b7280")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def deck_rule_table_page(pdf, metrics):
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis("off")
    ax.add_patch(plt.Rectangle((0, 0.94), 1, 0.06, transform=ax.transAxes, color="#17324d", zorder=-1))
    ax.text(0.04, 0.91, "5. Rule Engine", transform=ax.transAxes, fontsize=22, fontweight="bold", color="#17324d", va="top")
    rows = [
        ["Dimension", "US Mode", "EU Mode"],
        ["Regulatory focus", "Adverse-action reasons;\nfair-lending risk governance", "High-risk AI;\nbias testing and oversight"],
        ["Metric", "Disparate-impact style\napproval parity evidence", "Approval parity ratio\nas high-risk AI bias test"],
        ["Threshold", "0.80 internal warning", "0.90 internal escalation\nnot statutory"],
        ["Observed parity", f"{metrics.iloc[0]['approval_parity_ratio']:.3f}", f"{metrics.iloc[0]['approval_parity_ratio']:.3f}"],
        ["Tool conclusion", "Monitor; do not state\nlegal violation", "Escalate EU high-risk\nAI governance review"],
        ["Human role", "Review adverse-action\nreason quality", "Document bias testing,\nArticle 14 oversight sign-off"],
    ]
    table = ax.table(cellText=rows[1:], colLabels=rows[0], cellLoc="left", colLoc="left", bbox=[0.04, 0.16, 0.92, 0.68])
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1, 1.2)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#adb5bd")
        if row == 0:
            cell.set_facecolor("#d0ebff")
            cell.set_text_props(weight="bold")
        elif col == 0:
            cell.set_facecolor("#f8f9fa")
            cell.set_text_props(weight="bold")
    ax.text(0.05, 0.09, "Thresholds are governance settings. EU legal duties are mapped separately through Article 10, 14 and 17 controls.", transform=ax.transAxes, fontsize=10, color="#1f2933")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def deck_rule_change_flow_page(pdf):
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis("off")
    ax.add_patch(plt.Rectangle((0, 0.94), 1, 0.06, transform=ax.transAxes, color="#17324d", zorder=-1))
    ax.text(0.04, 0.91, "Rule Change Mid-Period", transform=ax.transAxes, fontsize=22, fontweight="bold", color="#17324d", va="top")
    ax.text(0.04, 0.86, "A spreadsheet stores values. A governance tool stores rule versions, effective dates and escalation language.", transform=ax.transAxes, fontsize=10.5, color="#52616b", va="top")
    steps = [
        (0.06, "Old rule\nThreshold 0.80\nVersioned output"),
        (0.30, "Regulatory update\nRegulation B\npolicy posture changes"),
        (0.54, "Rule file update\nEffective date\nLegal review sign-off"),
        (0.78, "Prospective output\nNew language\nPrior evidence preserved"),
    ]
    for x, label in steps:
        ax.add_patch(plt.Rectangle((x, 0.50), 0.17, 0.19, transform=ax.transAxes, facecolor="#e7f5ff", edgecolor="#495057", linewidth=1.2))
        ax.text(x + 0.085, 0.595, label, transform=ax.transAxes, ha="center", va="center", fontsize=8.2, fontweight="bold", color="#1f2933", linespacing=1.35)
    for start in [0.22, 0.46, 0.70]:
        ax.annotate("", xy=(start + 0.07, 0.595), xytext=(start, 0.595), xycoords=ax.transAxes, arrowprops=dict(arrowstyle="->", lw=1.8, color="#495057"))
    ax.add_patch(plt.Rectangle((0.15, 0.22), 0.70, 0.14, transform=ax.transAxes, facecolor="#fff9db", edgecolor="#f08c00", linewidth=1.2))
    ax.text(0.50, 0.305, "Design implication", transform=ax.transAxes, ha="center", va="center", fontsize=10.5, fontweight="bold", color="#17324d")
    ax.text(
        0.50,
        0.265,
        "Do not overwrite prior conclusions. Preserve rule history, apply new rules prospectively,\nand require Legal / Compliance / Model Risk sign-off.",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=8.7,
        color="#1f2933",
        linespacing=1.35,
    )
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def deck_regulatory_divergence_visual_page(pdf, metrics):
    us = metrics[metrics["active_jurisdiction"].eq("US")].iloc[0]
    eu = metrics[metrics["active_jurisdiction"].eq("EU")].iloc[0]
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis("off")
    ax.add_patch(plt.Rectangle((0, 0.94), 1, 0.06, transform=ax.transAxes, color="#17324d", zorder=-1))
    ax.text(0.04, 0.91, "6. US/EU Divergence", transform=ax.transAxes, fontsize=22, fontweight="bold", color="#17324d", va="top")
    ax.text(0.04, 0.86, "One model. One portfolio. Two governance conclusions.", transform=ax.transAxes, fontsize=10.5, color="#52616b", va="top")
    panels = [
        (0.07, "US mode", "#e7f5ff", f"Threshold {us['threshold']:.2f}", "Monitor;\ndo not state legal violation", "Specific adverse-action reasons\nFair-lending risk evidence\nLegal posture versioned"),
        (0.55, "EU mode", "#fff4e6", f"Internal threshold {eu['threshold']:.2f}", "Escalate high-risk\nAI governance review", "Article 10 data governance\nArticle 14 oversight\nArticle 17 quality system"),
    ]
    for x, title, color, threshold, conclusion, bullets in panels:
        ax.add_patch(plt.Rectangle((x, 0.28), 0.38, 0.46, transform=ax.transAxes, facecolor=color, edgecolor="#495057", linewidth=1.2))
        ax.text(x + 0.19, 0.68, title, transform=ax.transAxes, ha="center", va="center", fontsize=15, fontweight="bold", color="#17324d")
        ax.text(x + 0.19, 0.59, threshold, transform=ax.transAxes, ha="center", va="center", fontsize=12, fontweight="bold", color="#0b7285")
        ax.text(x + 0.19, 0.48, conclusion, transform=ax.transAxes, ha="center", va="center", fontsize=11, fontweight="bold", color="#1f2933", linespacing=1.35)
        ax.text(x + 0.19, 0.36, bullets, transform=ax.transAxes, ha="center", va="center", fontsize=9.5, color="#1f2933", linespacing=1.45)
    ax.text(0.50, 0.79, f"Observed approval parity = {us['approval_parity_ratio']:.3f}", transform=ax.transAxes, ha="center", va="center", fontsize=13, fontweight="bold", color="#d6336c")
    ax.annotate("", xy=(0.55, 0.515), xytext=(0.45, 0.515), xycoords=ax.transAxes, arrowprops=dict(arrowstyle="<->", lw=1.5, color="#6b7280"))
    ax.text(0.50, 0.18, "The EU threshold is not statutory; it triggers Meridian's evidence workflow for separately mapped AI Act controls.", transform=ax.transAxes, ha="center", fontsize=10, color="#1f2933")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def deck_dashboard_page(pdf, metrics, outputs, sensitivity):
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor("white")
    title = fig.add_axes([0, 0, 1, 1])
    title.axis("off")
    title.add_patch(plt.Rectangle((0, 0.94), 1, 0.06, transform=title.transAxes, color="#17324d", zorder=-1))
    title.text(0.04, 0.91, "7. Dashboard Snapshot", fontsize=22, fontweight="bold", color="#17324d", va="top")
    title.text(0.04, 0.86, "A management view should make the governance consequence legible in seconds.", fontsize=10.8, color="#52616b", va="top")
    base = metrics[metrics["active_jurisdiction"].eq("US")].iloc[0]

    kpis = [
        ("Approval parity", f"{base['approval_parity_ratio']:.3f}", "Below EU internal threshold", "#fff4e6"),
        ("US mode", "Monitor", "No legal conclusion", "#e7f5ff"),
        ("EU mode", "Escalate", "Bias + oversight evidence", "#ffe3e3"),
    ]
    for i, (label, value, note, color) in enumerate(kpis):
        x0 = 0.06 + i * 0.30
        title.add_patch(plt.Rectangle((x0, 0.72), 0.25, 0.10, transform=title.transAxes, facecolor=color, edgecolor="#ced4da", linewidth=1))
        title.text(x0 + 0.018, 0.795, label.upper(), transform=title.transAxes, fontsize=7.8, fontweight="bold", color="#52616b", va="center")
        title.text(x0 + 0.018, 0.765, value, transform=title.transAxes, fontsize=15, fontweight="bold", color="#17324d", va="center")
        title.text(x0 + 0.018, 0.735, note, transform=title.transAxes, fontsize=8.5, color="#1f2933", va="center")

    ax1 = fig.add_axes([0.07, 0.43, 0.27, 0.22])
    bars = ax1.bar(["Protected", "Reference"], [base["protected_group_approval_rate"], base["reference_group_approval_rate"]], color=["#0b7285", "#74b816"])
    ax1.set_ylim(0, 1)
    ax1.set_title("Approval rates", fontsize=10, fontweight="bold", color="#17324d")
    ax1.tick_params(axis="x", labelsize=8)
    ax1.tick_params(axis="y", labelsize=8)
    ax1.grid(axis="y", alpha=0.25)
    for bar in bars:
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.025, f"{bar.get_height():.1%}", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#1f2933")

    ax2 = fig.add_axes([0.40, 0.43, 0.25, 0.22])
    m = metrics[metrics["active_jurisdiction"].isin(["US", "EU"])]
    x = np.arange(len(m))
    ax2.bar(x - 0.16, m["approval_parity_ratio"], width=0.32, label="Observed", color="#0b7285")
    ax2.bar(x + 0.16, m["threshold"], width=0.32, label="Threshold", color="#f08c00")
    ax2.set_xticks(x, m["active_jurisdiction"])
    ax2.set_ylim(0.65, 1.0)
    ax2.set_title("Observed vs threshold", fontsize=10, fontweight="bold", color="#17324d")
    ax2.legend(fontsize=8)
    ax2.tick_params(axis="x", labelsize=8)
    ax2.tick_params(axis="y", labelsize=8)
    ax2.grid(axis="y", alpha=0.25)

    ax4 = fig.add_axes([0.72, 0.43, 0.22, 0.22])
    eu = sensitivity[sensitivity["active_jurisdiction"].eq("EU")].sort_values("protected_group_score_shift")
    eu_threshold = eu["threshold"].iloc[0]
    ax4.plot(eu["protected_group_score_shift"], eu["approval_parity_ratio"], marker="o", color="#d6336c", linewidth=1.8)
    ax4.axhline(eu_threshold, linestyle="--", color="#d6336c", linewidth=1)
    ax4.set_title("EU sensitivity", fontsize=10, fontweight="bold", color="#17324d")
    ax4.set_xlabel("Score shift", fontsize=8)
    ax4.tick_params(axis="both", labelsize=7.5)
    ax4.grid(True, alpha=0.25)

    ax3 = fig.add_axes([0.08, 0.16, 0.50, 0.20])
    reasons = outputs[outputs["approved"].eq(0)]["decision_reason_1"].value_counts().head(5)
    ax3.barh([wrap_cell(idx, 28) for idx in reasons.index[::-1]], reasons.values[::-1], color="#5c7cfa")
    for y, value in enumerate(reasons.values[::-1]):
        ax3.text(value + max(reasons.values) * 0.02, y, f"{value}", va="center", fontsize=8, color="#1f2933")
    ax3.tick_params(axis="y", labelsize=8)
    ax3.tick_params(axis="x", labelsize=8)
    ax3.set_title("Top rejection reasons", fontsize=10, fontweight="bold", color="#17324d")
    ax3.spines[["top", "right"]].set_visible(False)

    title.add_patch(plt.Rectangle((0.64, 0.17), 0.30, 0.16, transform=title.transAxes, facecolor="#f8f9fa", edgecolor="#ced4da", linewidth=1))
    title.text(0.66, 0.285, "Management readout", transform=title.transAxes, fontsize=10.5, fontweight="bold", color="#17324d")
    title.text(0.66, 0.245, wrap_cell("US mode keeps monitoring; EU mode escalates because parity falls below Meridian's internal high-risk AI governance threshold.", 44), transform=title.transAxes, fontsize=8.6, color="#1f2933", va="top", linespacing=1.25)
    title.text(0.04, 0.055, f"{STUDENT['name']} | {STUDENT['matriculation_id']} | Senior management deck", transform=title.transAxes, fontsize=8.5, color="#6b7280")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def deck_sample_output_page(pdf, outputs):
    rejected = outputs[outputs["approved"].eq(0)].head(5)[
        ["applicant_id", "model_score", "decision_reason_1", "decision_reason_2", "us_mode_output", "eu_mode_output"]
    ].copy()
    rejected["model_score"] = rejected["model_score"].map(lambda x: f"{x:.3f}")
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis("off")
    ax.add_patch(plt.Rectangle((0, 0.94), 1, 0.06, transform=ax.transAxes, color="#17324d", zorder=-1))
    ax.text(0.04, 0.91, "8. Explainability Output", transform=ax.transAxes, fontsize=22, fontweight="bold", color="#17324d", va="top")
    ax.text(0.04, 0.85, "The tool creates consumer-facing reason evidence and separate jurisdiction-specific governance text.", transform=ax.transAxes, fontsize=10.5, color="#52616b", va="top")
    display = rejected[["applicant_id", "model_score", "decision_reason_1", "decision_reason_2"]].copy()
    for col in ["decision_reason_1", "decision_reason_2"]:
        display[col] = display[col].map(lambda text: wrap_box_text(text, 23))
    table = ax.table(
        cellText=display.values,
        colLabels=display.columns,
        cellLoc="left",
        colLoc="left",
        colWidths=[0.22, 0.15, 0.315, 0.315],
        bbox=[0.04, 0.38, 0.92, 0.36],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.1)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#adb5bd")
        if row == 0:
            cell.set_facecolor("#d0ebff")
            cell.set_text_props(weight="bold")
    ax.text(0.04, 0.26, "US output example: adverse-action notice required with ranked principal reasons.", transform=ax.transAxes, fontsize=10, fontweight="bold", color="#0b7285")
    ax.text(0.04, 0.21, "EU output example: high-risk AI documentation / human oversight evidence is required separately from customer-facing reasons.", transform=ax.transAxes, fontsize=10, fontweight="bold", color="#d6336c")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def deck_sensitivity_page(pdf, sensitivity):
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis("off")
    ax.add_patch(plt.Rectangle((0, 0.94), 1, 0.06, transform=ax.transAxes, color="#17324d", zorder=-1))
    ax.text(0.04, 0.91, "Sensitivity Test", transform=ax.transAxes, fontsize=22, fontweight="bold", color="#17324d", va="top")
    plot_ax = fig.add_axes([0.10, 0.30, 0.78, 0.45])
    for jurisdiction, color in [("US", "#1f77b4"), ("EU", "#d62728")]:
        sub = sensitivity[sensitivity["active_jurisdiction"].eq(jurisdiction)]
        plot_ax.plot(sub["protected_group_score_shift"], sub["approval_parity_ratio"], marker="o", label=f"{jurisdiction} observed parity", color=color)
    us_threshold = sensitivity[sensitivity["active_jurisdiction"].eq("US")]["threshold"].iloc[0]
    eu_threshold = sensitivity[sensitivity["active_jurisdiction"].eq("EU")]["threshold"].iloc[0]
    plot_ax.axhline(us_threshold, color="#1f77b4", linestyle="--", linewidth=1, label="US threshold")
    plot_ax.axhline(eu_threshold, color="#d62728", linestyle="--", linewidth=1, label="EU threshold")
    plot_ax.set_xlabel("Protected-group model-score shift")
    plot_ax.set_ylabel("Approval parity ratio")
    plot_ax.grid(True, alpha=0.25)
    plot_ax.legend(fontsize=8)
    ax.text(
        0.10,
        0.09,
        "Management meaning: EU escalation is more sensitive because Meridian chooses a stricter internal high-risk AI governance threshold.",
        transform=ax.transAxes,
        fontsize=10,
        color="#1f2933",
        bbox=dict(facecolor="white", edgecolor="#dee2e6", boxstyle="round,pad=0.35"),
    )
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def deck_model_comparison_page(pdf, model_comparison):
    fig = plt.figure(figsize=(11, 6.2))
    fig.patch.set_facecolor("#ffffff")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.text(0.055, 0.91, "Model Choice: Accuracy vs Explainability", transform=ax.transAxes, fontsize=20, fontweight="bold", color="#17324d")
    ax.text(0.055, 0.84, "The challenger benchmark adds technical depth; the selected primary model preserves auditable consumer explanations.", transform=ax.transAxes, fontsize=10.5, color="#52616b")
    plot_ax = fig.add_axes([0.09, 0.22, 0.38, 0.46])
    x = np.arange(len(model_comparison))
    width = 0.32
    plot_ax.bar(x - width / 2, model_comparison["auc"], width, label="AUC", color="#0b7285")
    plot_ax.bar(x + width / 2, model_comparison["accuracy"], width, label="Accuracy", color="#f08c00")
    plot_ax.set_xticks(x)
    plot_ax.set_xticklabels(model_comparison["model_name"], rotation=10, ha="right", fontsize=8)
    plot_ax.set_ylim(0, 1.0)
    plot_ax.set_ylabel("Held-out score")
    plot_ax.legend(frameon=False)
    plot_ax.grid(axis="y", alpha=0.25)
    text_ax = fig.add_axes([0.54, 0.20, 0.38, 0.50])
    text_ax.axis("off")
    body = (
        "Governance interpretation:\n"
        "- Logistic regression remains the official model because signed contributions connect technical features to adverse-action reasons.\n"
        "- Random Forest is a challenger that tests whether non-linear structure changes performance or feature priorities.\n"
        "- If a bank adopted the challenger, TreeSHAP and additional validation would be required before consumer-facing use."
    )
    add_wrapped(text_ax, body, 0.02, 0.96, width=62, size=10, color="#1f2933", line_gap=0.07)
    ax.text(0.055, 0.06, f"{STUDENT['name']} | {STUDENT['matriculation_id']} | Model benchmark visual", transform=ax.transAxes, fontsize=8.5, color="#6b7280")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def deck_boundary_choice_page(pdf):
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis("off")
    ax.add_patch(plt.Rectangle((0, 0.94), 1, 0.06, transform=ax.transAxes, color="#17324d", zorder=-1))
    ax.text(0.04, 0.91, "10. Boundaries", transform=ax.transAxes, fontsize=22, fontweight="bold", color="#17324d", va="top")
    ax.text(0.04, 0.86, "What the tool does not do, and why those limits are part of good governance.", transform=ax.transAxes, fontsize=10.5, color="#52616b", va="top")
    boundaries = [
        ("Not legal advice", "Reason: statistical disparity is evidence for review,\nnot a legal conclusion."),
        ("Not autonomous approval", "Reason: credit policy and overrides require\naccountable human judgement."),
        ("Not auto-remediation", "Reason: changing thresholds or retraining models\ncan create new consumer harms."),
        ("Not proof of Citi compliance", "Reason: the prototype uses synthetic data and\npublic case-study evidence only."),
    ]
    positions = [(0.07, 0.55), (0.55, 0.55), (0.07, 0.27), (0.55, 0.27)]
    colors = ["#e7f5ff", "#e6fcf5", "#fff9db", "#ffe3e3"]
    for (title, body), (x, y), color in zip(boundaries, positions, colors):
        ax.add_patch(plt.Rectangle((x, y), 0.38, 0.18, transform=ax.transAxes, facecolor=color, edgecolor="#495057", linewidth=1.1))
        ax.text(x + 0.03, y + 0.135, title, transform=ax.transAxes, fontsize=11.2, fontweight="bold", color="#17324d", va="center")
        ax.text(
            x + 0.03,
            y + 0.090,
            wrap_box_text(body, 39),
            transform=ax.transAxes,
            fontsize=8.5,
            color="#1f2933",
            va="top",
            linespacing=1.25,
        )
    ax.text(
        0.50,
        0.13,
        wrap_box_text("Management message: the tool escalates evidence to humans; it does not hide policy choices inside automation.", 104),
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=9.7,
        color="#1f2933",
        linespacing=1.2,
    )
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def write_deck_pdf(path, metrics, df, outputs, sensitivity, model_comparison):
    us = metrics[metrics["active_jurisdiction"] == "US"].iloc[0]
    eu = metrics[metrics["active_jurisdiction"] == "EU"].iloc[0]
    with PdfPages(path) as pdf:
        footer = f"{STUDENT['name']} | {STUDENT['matriculation_id']} | Senior management deck"
        deck_problem_page(pdf, metrics)
        deck_citi_meridian_page(pdf)
        deck_meridian_value_page(pdf)
        deck_architecture_page(pdf)
        deck_rule_table_page(pdf, metrics)
        deck_regulatory_divergence_visual_page(pdf, metrics)
        deck_dashboard_page(pdf, metrics, outputs, sensitivity)
        deck_sample_output_page(pdf, outputs)
        pdf_page(
            pdf,
            "9. Failure Modes",
            [
                (
                    "",
                    "Key risks are rule changes mid-period, model drift, data drift, jurisdiction misconfiguration, missing protected attributes, contradictory rules and over-reliance on technical explanations. The tool responds by versioning rules, preserving effective dates, separating legal conclusions from statistical evidence and escalating unresolved choices to human owners.",
                )
            ],
            subtitle="The dashboard must surface risk, not hide it",
            footer=footer,
            landscape=True,
        )
        deck_boundary_choice_page(pdf)
        pdf_page(
            pdf,
            "11. Ask / Next Steps",
            [
                (
                    "",
                    "The ask is approval to treat Meridian FairAI Governance as a narrow RegTech prototype for Citi-style cross-border AI credit governance. The next build would validate with real customer data, legal sign-off, model validation, audit logging, access control, production monitoring and integration with model inventory and issue-management workflows.",
                )
            ],
            subtitle=f"Current demo parity {us['approval_parity_ratio']:.3f}; US threshold {us['threshold']:.2f}; EU internal threshold {eu['threshold']:.2f}",
            footer=footer,
            landscape=True,
        )


def write_readme(path):
    text = f"""# MH6822 Assignment 1

{STUDENT['name']}
{STUDENT['matriculation_id']}
{STUDENT['email']}

## Topic

A Jurisdiction-Aware AI Credit Governance Tool for Cross-Border Banks: Comparing US Fair Lending Explainability and EU AI Act Bias Governance.

## Regulated entity and domain

The selected regulated entity is Citi / Citigroup. The selected domain is AI credit scoring, fair lending, algorithmic fairness and AI governance. The primary comparison is between the United States and the European Union.

Singapore and UK appear in the jurisdiction rule file only as extensible design examples. The substantive assignment analysis focuses on US and EU.

## Contents

- `Task1_Selection_and_Research.pdf`: entity selection, domain rationale, jurisdictional comparison and references.
- `Task2_Values_Audit.pdf`: mission statement, stakeholder perspective, risk-vs-documentation discussion and error-cost analysis.
- `Task3_Tool_Design_Deck.pdf`: senior management deck.
- `Task3_Dashboard_Mockup.pdf`: visual dashboard mockup of approval rates, thresholds, adverse-action reasons and sensitivity.
- `Task3_SHAP_Explainability.pdf`: applicant-level SHAP-style explanation plots and mapping to adverse-action reason categories.
- `Task3_notebook_shap_bar.png`: plot generated by the notebook's SHAP/fallback visualization cell.
- `Task3_notebook_treeshap_bar.png`: TreeSHAP plot for the Random Forest challenger generated by the notebook workflow.
- `Task3_AI_Credit_Governance_Prototype.ipynb`: executable notebook demonstrating the data, model, optional SHAP explainability, fairness metrics and jurisdiction-aware rule layer.
- `Task3_AI_Credit_Governance_Prototype_executed.ipynb`: pre-executed notebook with saved cell outputs.
- `Task3_Model_Card.pdf`: governance documentation stub and model card.
- `Task3_Model_Comparison.pdf`: logistic-regression primary model versus Random Forest challenger benchmark.
- `Task3_One_Page_Summary.pdf`: plain-language one-page summary.
- `Task3_Sensitivity_Analysis.pdf`: quantitative sensitivity analysis of threshold conclusions under protected-group score shifts.
- `Task3_Technical_Appendix.pdf`: technical appendix explaining data generation, model logic, formulas, jurisdiction logic and limitations.
- `Task3_build_assignment.py`: generation script for data, PDFs, notebook and archive.
- `data/Task3_synthetic_credit_applications.csv`: simulated application-level data.
- `data/Task3_jurisdiction_rules.csv`: configurable jurisdiction rule layer.
- `data/Task3_model_outputs.csv`: model outputs and jurisdiction-specific reporting text.
- `data/Task3_fairness_metrics_by_jurisdiction.csv`: fairness metrics and governance actions by active jurisdiction.
- `data/Task3_sensitivity_analysis.csv`: sensitivity test results for US and EU thresholds.
- `data/Task3_shap_explanations.csv`: top applicant-level SHAP-style feature contributions for rejected applicants.
- `data/Task3_model_comparison.csv`: primary versus challenger model metrics.
- `data/Task3_random_forest_feature_importance.csv`: Random Forest challenger feature importance table.
- `data/Task3_eu_ai_act_obligations.csv`: EU AI Act obligation-to-control matrix for Article 10, Article 14 and Article 17.
- `data/Task3_data_dictionary.csv`: field-level data dictionary, including model-use status, protected/proxy status, synthetic-bias role and production-use cautions.

## Data note

This assignment uses synthetic credit application data. The fields are designed with reference to public lending and credit-risk datasets and regulatory concepts, but the data is not real Citi customer data and should not be interpreted as evidence about Citi's actual models, customers or compliance posture.

The `model_score` column is generated after training the primary logistic-regression model. It is the trained model's predicted probability of default, not a separate pre-model synthetic score.

## Data collaboration

The synthetic data for this prepared submission was generated independently. No other students were involved in the data generation portion.

## How to reproduce the generated artifacts

Run:

```bash
python3 Task3_build_assignment.py
```

In the working folder used to prepare the submission, the same script is named `build_assignment.py`.
"""
    path.write_text(text, encoding="utf-8")


def write_notebook(path):
    nb = nbf.v4.new_notebook()
    nb["cells"] = [
            nbf.v4.new_markdown_cell(
                f"""# Task 3 Prototype: Jurisdiction-Aware AI Credit Governance Tool

Student: {STUDENT['name']}  
Matriculation ID: {STUDENT['matriculation_id']}  
Email: {STUDENT['email']}

This notebook demonstrates the core logic of a jurisdiction-aware RegTech prototype for AI credit scoring governance. It uses the generated CSV files in the `data/` folder.
""",
            ),
            nbf.v4.new_code_cell(
                """import pandas as pd
import numpy as np
from pathlib import Path

candidate_roots = [Path.cwd(), Path.cwd().parent, Path.cwd().parent.parent]
PROJECT_ROOT = next(
    root for root in candidate_roots
    if (root / "data" / "Task3_synthetic_credit_applications.csv").exists()
)
DATA_DIR = PROJECT_ROOT / "data"

applications = pd.read_csv(DATA_DIR / "Task3_synthetic_credit_applications.csv")
rules = pd.read_csv(DATA_DIR / "Task3_jurisdiction_rules.csv")
metrics = pd.read_csv(DATA_DIR / "Task3_fairness_metrics_by_jurisdiction.csv")
outputs = pd.read_csv(DATA_DIR / "Task3_model_outputs.csv")
sensitivity = pd.read_csv(DATA_DIR / "Task3_sensitivity_analysis.csv")
shap_explanations = pd.read_csv(DATA_DIR / "Task3_shap_explanations.csv")
model_comparison = pd.read_csv(DATA_DIR / "Task3_model_comparison.csv")
rf_importance = pd.read_csv(DATA_DIR / "Task3_random_forest_feature_importance.csv")

print(f"Loaded data from: {DATA_DIR}")

applications.head()
""",
            ),
            nbf.v4.new_markdown_cell(
                """## Train a demonstration model

The submitted `model_score` column is produced by the scikit-learn logistic-regression pipeline in `Task3_build_assignment.py` using `predict_proba`. The compact model below is included only so the notebook can show the mechanics of a logistic default-risk model. Production deployment would require model validation, legal review, privacy review and real operational controls.
""",
            ),
            nbf.v4.new_code_cell(
                """features = [
    "age", "income", "credit_score", "debt_to_income", "loan_amount",
    "loan_term_months", "prior_default", "region_risk_score"
]
X = applications[features].astype(float).to_numpy()
y = applications["default_next_12m"].astype(float).to_numpy()

rng = np.random.default_rng(6822)
idx = rng.permutation(len(y))
test_size = int(0.30 * len(y))
test_idx, train_idx = idx[:test_size], idx[test_size:]

mean = X[train_idx].mean(axis=0)
std = X[train_idx].std(axis=0) + 1e-9
X_scaled = (X - mean) / std
X_design = np.column_stack([np.ones(len(X_scaled)), X_scaled])

X_train, y_train = X_design[train_idx], y[train_idx]
X_test, y_test = X_design[test_idx], y[test_idx]

weights = np.zeros(X_train.shape[1])
learning_rate = 0.08
for _ in range(1200):
    prob_train = 1 / (1 + np.exp(-(X_train @ weights)))
    gradient = X_train.T @ (prob_train - y_train) / len(y_train)
    weights -= learning_rate * gradient

prob = 1 / (1 + np.exp(-(X_test @ weights)))
pred = (prob >= 0.5).astype(int)

def auc_rank(y_true, score):
    order = np.argsort(score)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(score) + 1)
    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos
    return (ranks[y_true == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)

{"AUC": round(float(auc_rank(y_test, prob)), 3), "Accuracy": round(float((pred == y_test).mean()), 3)}
""",
            ),
            nbf.v4.new_markdown_cell(
                """## Model benchmark comparison

The submitted package includes a scikit-learn logistic-regression primary model and a Random Forest challenger. The challenger demonstrates non-linear benchmarking, while the primary model is retained for governance because it produces signed contributions that map more cleanly to consumer-facing adverse-action reasons.
""",
            ),
            nbf.v4.new_code_cell(
                """display(model_comparison)
display(rf_importance.head(8))
""",
            ),
            nbf.v4.new_markdown_cell(
                """## TreeSHAP for the Random Forest challenger

This cell trains a small Random Forest challenger inside the notebook and, when the `shap` package is available, computes true TreeSHAP values. This is separate from the primary logistic-regression governance model: it demonstrates how a non-linear challenger would need its own explanation evidence before being used in a consumer-facing credit workflow.
""",
            ),
            nbf.v4.new_code_cell(
                """from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import matplotlib.pyplot as plt

rf_X_train = applications.iloc[train_idx][features].astype(float)
rf_X_test = applications.iloc[test_idx][features].astype(float)
rf_y_train = applications.iloc[train_idx]["default_next_12m"].astype(int)
rf_y_test = applications.iloc[test_idx]["default_next_12m"].astype(int)

rf_challenger = RandomForestClassifier(
    n_estimators=160,
    max_depth=8,
    min_samples_leaf=20,
    class_weight="balanced_subsample",
    random_state=6822,
    n_jobs=-1,
)
rf_challenger.fit(rf_X_train, rf_y_train)
rf_prob = rf_challenger.predict_proba(rf_X_test)[:, 1]
rf_metrics_notebook = {
    "AUC": round(float(roc_auc_score(rf_y_test, rf_prob)), 3),
    "Accuracy": round(float(accuracy_score(rf_y_test, (rf_prob >= 0.5).astype(int))), 3),
}

try:
    try:
        import shap
    except ImportError:
        import sys
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "shap", "-q"])
        import shap

    tree_sample = rf_X_test.head(200)
    tree_explainer = shap.TreeExplainer(rf_challenger)
    tree_values = tree_explainer.shap_values(tree_sample)
    if isinstance(tree_values, list):
        class_one_values = tree_values[1]
    elif getattr(tree_values, "ndim", 0) == 3:
        class_one_values = tree_values[:, :, 1]
    else:
        class_one_values = tree_values

    treeshap_importance = (
        pd.Series(np.abs(class_one_values).mean(axis=0), index=features)
        .sort_values(ascending=True)
    )
    ax = treeshap_importance.plot(kind="barh", figsize=(8, 5), color="#d6336c")
    ax.set_title("TreeSHAP: Random Forest challenger feature contribution magnitude")
    ax.set_xlabel("Mean absolute TreeSHAP value")
    plt.tight_layout()
    plt.savefig("Task3_notebook_treeshap_bar.png", dpi=100)
    plt.show()
    treeshap_status = "TreeSHAP executed"
except Exception as exc:
    fallback_importance = pd.Series(rf_challenger.feature_importances_, index=features).sort_values(ascending=True)
    ax = fallback_importance.plot(kind="barh", figsize=(8, 5), color="#868e96")
    ax.set_title("Random Forest feature importance fallback")
    ax.set_xlabel("Impurity-based importance")
    plt.tight_layout()
    plt.savefig("Task3_notebook_treeshap_bar.png", dpi=100)
    plt.show()
    treeshap_status = f"TreeSHAP unavailable; fallback RF importance used ({type(exc).__name__})"

print(treeshap_status)
rf_metrics_notebook
""",
            ),
            nbf.v4.new_markdown_cell(
                """## Optional SHAP explainability

The assignment brief mentions SHAP/LIME as a reasonable way to demonstrate explainable AI. This cell first tries to import `shap`; if missing, it attempts a quiet local pip install. If installation is blocked, the notebook falls back to the linear model's transparent SHAP-style feature contributions so the prototype still runs end-to-end.
""",
            ),
            nbf.v4.new_code_cell(
                """feature_names = ["intercept"] + features
explanation_sample = X_test[:5]

def predict_from_design_matrix(z):
    return 1 / (1 + np.exp(-(z @ weights)))

try:
    try:
        import shap
        import matplotlib.pyplot as plt
    except ImportError:
        import sys
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "shap", "-q"])
        import shap
        import matplotlib.pyplot as plt
    background = X_train[:100]
    explainer = shap.Explainer(predict_from_design_matrix, background)
    shap_values = explainer(explanation_sample)
    explainability_output = pd.DataFrame(
        shap_values.values,
        columns=feature_names,
        index=[f"test_applicant_{i+1}" for i in range(len(explanation_sample))]
    ).round(4)
    explanation_method = "Package SHAP values"
except Exception as exc:
    contributions = explanation_sample * weights
    explainability_output = pd.DataFrame(
        contributions,
        columns=feature_names,
        index=[f"test_applicant_{i+1}" for i in range(len(explanation_sample))]
    ).round(4)
    explanation_method = f"SHAP unavailable; fallback linear contributions used ({type(exc).__name__})"

print(explanation_method)
explainability_output
""",
            ),
            nbf.v4.new_markdown_cell(
                """## Notebook SHAP visualization

This cell produces a notebook-native SHAP bar plot when package SHAP is available. If package SHAP could not run, it creates an equivalent bar chart from the fallback linear contributions so that the notebook still contains a visual explanation.
""",
            ),
            nbf.v4.new_code_cell(
                """import matplotlib.pyplot as plt

if explanation_method == "Package SHAP values":
    raw_names = feature_names
    clean_names = [
        name.replace("num__", "")
            .replace("cat__", "")
            .replace("employment_status_", "employment_status=")
            .replace("loan_purpose_", "loan_purpose=")
        for name in raw_names
    ]
    try:
        shap_values.feature_names = clean_names
    except Exception:
        pass
    shap.plots.bar(shap_values, show=False)
    plt.tight_layout()
    plt.savefig("Task3_notebook_shap_bar.png", dpi=100)
    plt.show()
else:
    mean_abs = explainability_output.abs().mean().sort_values(ascending=True)
    ax = mean_abs.plot(kind="barh", figsize=(8, 5), color="#0b7285")
    ax.set_title("Fallback SHAP-style feature contribution magnitude")
    ax.set_xlabel("Mean absolute contribution")
    plt.tight_layout()
    plt.savefig("Task3_notebook_shap_bar.png", dpi=100)
    plt.show()
""",
            ),
            nbf.v4.new_markdown_cell(
                """## SHAP-style explanations for rejected applicants

The generated package includes applicant-level linear SHAP-style contributions from the benchmark logistic-regression model. Positive values increase predicted default risk and are mapped to adverse-action reason categories.
""",
            ),
            nbf.v4.new_code_cell(
                """shap_explanations.head(12)
""",
            ),
            nbf.v4.new_markdown_cell(
                """## Jurisdiction-aware output

The same approval data is evaluated under different regulatory thresholds and governance expectations. The EU 0.90 threshold is not a statutory AI Act threshold; it is Meridian's internal conservative governance escalation threshold, while legal obligations are mapped separately through Article 10 data governance, Article 14 human oversight and Article 17 quality-management controls.
""",
            ),
            nbf.v4.new_code_cell(
                """metrics[[
    "active_jurisdiction", "metric_name", "protected_group_approval_rate",
    "reference_group_approval_rate", "approval_parity_ratio",
    "protected_group_false_positive_rate", "reference_group_false_positive_rate",
    "protected_group_false_negative_rate", "reference_group_false_negative_rate",
    "threshold",
    "fairness_flag", "required_governance_action"
]]
""",
            ),
            nbf.v4.new_code_cell(
                """sample = outputs.loc[outputs["approved"].eq(0), [
    "applicant_id", "model_score", "decision_reason_1", "decision_reason_2",
    "us_mode_output", "eu_mode_output"
]].head(10)
sample
""",
            ),
            nbf.v4.new_markdown_cell(
                """## Rule change mid-period simulation

This cell demonstrates how an internal jurisdiction rule change changes governance output. It simulates Meridian's EU governance threshold moving from 0.90 to 0.85 and compares the flag before and after the change.
""",
            ),
            nbf.v4.new_code_cell(
                """approval_parity_ratio = metrics.loc[
    metrics["active_jurisdiction"].eq("EU"),
    "approval_parity_ratio"
].iloc[0]

old_threshold = metrics.loc[
    metrics["active_jurisdiction"].eq("EU"),
    "threshold"
].iloc[0]
new_threshold = 0.85

old_flag = approval_parity_ratio < old_threshold
new_flag = approval_parity_ratio < new_threshold

governance_effect = (
    "Escalation removed under updated threshold"
    if old_flag and not new_flag
    else "Escalation unchanged"
)

print("Rule change simulation")
print(f"Approval parity ratio: {approval_parity_ratio:.4f}")
print(f"Old EU threshold: {old_threshold:.2f}")
print(f"Old governance flag: {bool(old_flag)}")
print(f"Updated EU threshold: {new_threshold:.2f}")
print(f"Updated governance flag: {bool(new_flag)}")
print(f"Governance effect: {governance_effect}")
""",
            ),
            nbf.v4.new_markdown_cell(
                """## Sensitivity analysis

The same tool can test how conclusions change if the protected-group score distribution shifts. This helps management see whether a governance conclusion is robust or sitting near a threshold.
""",
            ),
            nbf.v4.new_code_cell(
                """sensitivity.pivot(
    index="protected_group_score_shift",
    columns="active_jurisdiction",
    values="fairness_flag"
)
""",
            ),
            nbf.v4.new_markdown_cell(
                """## Interpretation

US mode prioritizes specific adverse-action explanation and fair-lending review. EU mode treats creditworthiness AI as a high-risk AI governance problem, so the same portfolio-level metric can trigger stricter bias-testing, documentation and human-oversight action.
""",
            ),
        ]
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
    }
    nbf.write(nb, path)


def write_notebook_treeshap_png(df, path):
    features = [
        "age",
        "income",
        "credit_score",
        "debt_to_income",
        "loan_amount",
        "loan_term_months",
        "prior_default",
        "region_risk_score",
    ]
    X = df[features].astype(float)
    y = df["default_next_12m"].astype(int)
    X_train, X_test, y_train, _y_test = train_test_split(X, y, test_size=0.30, random_state=6822, stratify=y)
    rf = RandomForestClassifier(
        n_estimators=160,
        max_depth=8,
        min_samples_leaf=20,
        class_weight="balanced_subsample",
        random_state=6822,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    try:
        import shap

        tree_sample = X_test.head(200)
        tree_values = shap.TreeExplainer(rf).shap_values(tree_sample)
        if isinstance(tree_values, list):
            class_one_values = tree_values[1]
        elif getattr(tree_values, "ndim", 0) == 3:
            class_one_values = tree_values[:, :, 1]
        else:
            class_one_values = tree_values
        values = pd.Series(np.abs(class_one_values).mean(axis=0), index=features).sort_values(ascending=True)
        title = "TreeSHAP: Random Forest challenger feature contribution magnitude"
        color = "#d6336c"
        xlabel = "Mean absolute TreeSHAP value"
    except Exception:
        values = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=True)
        title = "Random Forest feature importance fallback"
        color = "#868e96"
        xlabel = "Impurity-based importance"

    fig, ax = plt.subplots(figsize=(8, 5))
    values.plot(kind="barh", ax=ax, color=color)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def execute_notebook(notebook_path, executed_path):
    from nbconvert.preprocessors import ExecutePreprocessor

    notebook = nbf.read(notebook_path, as_version=4)
    executor = ExecutePreprocessor(timeout=900, kernel_name="python3")
    executor.preprocess(notebook, {"metadata": {"path": str(ROOT)}})
    nbf.write(notebook, executed_path)


def build_zip(files, aliases=None):
    aliases = aliases or {}
    zip_path = ROOT / ZIP_NAME
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file in files:
            arcname = aliases.get(file, file.relative_to(ROOT))
            zf.write(file, arcname)
    return zip_path


def main():
    ensure_dirs()
    df = generate_data()
    rules = jurisdiction_rules()
    model, model_metrics = train_model(df)
    df = score_applications(model, df)
    _rf_model, rf_metrics, rf_importance = train_random_forest_benchmark(df)
    model_comparison = pd.DataFrame(
        [
            {
                "model_name": "Logistic regression",
                "auc": model_metrics["auc"],
                "accuracy": model_metrics["accuracy"],
                "test_size": model_metrics["test_size"],
                "governance_role": "Primary model for adverse-action mapping and auditability.",
            },
            {
                "model_name": "Random Forest",
                "auc": rf_metrics["auc"],
                "accuracy": rf_metrics["accuracy"],
                "test_size": rf_metrics["test_size"],
                "governance_role": "Challenger benchmark for non-linear signal and explanation contrast.",
            },
        ]
    )
    metrics = apply_jurisdiction_logic(df, rules)
    sensitivity = build_sensitivity_analysis(df, rules)
    outputs = build_model_outputs(df, metrics)
    shap_explanations = build_shap_explanations(model, df)
    obligations = eu_ai_act_obligations()
    dictionary = data_dictionary()

    data_files = [
        DATA_DIR / "Task3_synthetic_credit_applications.csv",
        DATA_DIR / "Task3_jurisdiction_rules.csv",
        DATA_DIR / "Task3_model_outputs.csv",
        DATA_DIR / "Task3_fairness_metrics_by_jurisdiction.csv",
        DATA_DIR / "Task3_sensitivity_analysis.csv",
        DATA_DIR / "Task3_shap_explanations.csv",
        DATA_DIR / "Task3_model_comparison.csv",
        DATA_DIR / "Task3_random_forest_feature_importance.csv",
        DATA_DIR / "Task3_eu_ai_act_obligations.csv",
        DATA_DIR / "Task3_data_dictionary.csv",
    ]
    df.to_csv(data_files[0], index=False)
    rules.to_csv(data_files[1], index=False)
    outputs.to_csv(data_files[2], index=False)
    metrics.to_csv(data_files[3], index=False)
    sensitivity.to_csv(data_files[4], index=False)
    shap_explanations.to_csv(data_files[5], index=False)
    model_comparison.to_csv(data_files[6], index=False)
    rf_importance.to_csv(data_files[7], index=False)
    obligations.to_csv(data_files[8], index=False)
    dictionary.to_csv(data_files[9], index=False)

    readme = ROOT / "README.md"
    task1 = ROOT / "Task1_Selection_and_Research.pdf"
    task2 = ROOT / "Task2_Values_Audit.pdf"
    deck = ROOT / "Task3_Tool_Design_Deck.pdf"
    dashboard = ROOT / "Task3_Dashboard_Mockup.pdf"
    shap_pdf = ROOT / "Task3_SHAP_Explainability.pdf"
    model_card = ROOT / "Task3_Model_Card.pdf"
    summary = ROOT / "Task3_One_Page_Summary.pdf"
    sensitivity_pdf = ROOT / "Task3_Sensitivity_Analysis.pdf"
    technical_appendix = ROOT / "Task3_Technical_Appendix.pdf"
    model_comparison_pdf = ROOT / "Task3_Model_Comparison.pdf"
    notebook = ROOT / "Task3_AI_Credit_Governance_Prototype.ipynb"
    executed_notebook = ROOT / EXECUTED_NOTEBOOK_NAME
    notebook_shap_png = ROOT / "Task3_notebook_shap_bar.png"
    notebook_treeshap_png = ROOT / "Task3_notebook_treeshap_bar.png"

    write_readme(readme)
    write_task1_pdf(task1)
    write_task2_pdf(task2, metrics)
    write_deck_pdf(deck, metrics, df, outputs, sensitivity, model_comparison)
    write_dashboard_pdf(dashboard, metrics, df, outputs, sensitivity)
    write_shap_pdf(shap_pdf, shap_explanations)
    write_model_card_pdf(model_card, model_metrics, rf_metrics)
    write_summary_pdf(summary, metrics, model_metrics)
    write_sensitivity_pdf(sensitivity_pdf, sensitivity)
    write_technical_appendix_pdf(technical_appendix, metrics, sensitivity, model_metrics, rf_metrics)
    write_model_comparison_pdf(model_comparison_pdf, model_comparison, rf_importance)
    write_notebook(notebook)
    execute_notebook(notebook, executed_notebook)
    write_notebook_treeshap_png(df, notebook_treeshap_png)
    fig, ax = plt.subplots(figsize=(8, 5))
    shap_explanations.groupby("technical_feature")["linear_shap_log_odds"].apply(lambda s: s.abs().mean()).sort_values().plot(
        kind="barh", ax=ax, color="#0b7285"
    )
    ax.set_title("Notebook SHAP-style feature contribution magnitude")
    ax.set_xlabel("Mean absolute contribution")
    fig.tight_layout()
    fig.savefig(notebook_shap_png, dpi=100)
    plt.close(fig)

    self_script = Path(__file__).resolve()
    zip_path = build_zip(
        [readme, task1, task2, deck, dashboard, shap_pdf, notebook_shap_png, notebook_treeshap_png, model_card, summary, sensitivity_pdf, technical_appendix, model_comparison_pdf, notebook, executed_notebook, self_script, *data_files],
        aliases={self_script: "Task3_build_assignment.py"},
    )
    print(f"Generated {zip_path.name}")
    print(metrics.to_string(index=False))
    print(model_metrics)
    print(rf_metrics)


if __name__ == "__main__":
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib-cache"))
    main()
