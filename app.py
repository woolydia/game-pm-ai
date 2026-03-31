import os
import re

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
from openai import OpenAI
from dotenv import load_dotenv

# =========================
# 환경변수 로드
# =========================
load_dotenv(".env.local")

# =========================
# 한글 폰트 자동 설정
# =========================
from pathlib import Path
from matplotlib import font_manager as fm

def set_korean_font():
    font_file_candidates = [
        Path("fonts/NotoSansKR-Regular.ttf"),
        Path("fonts/NotoSansKR-Regular.otf"),
        Path("fonts/NotoSansCJKkr-Regular.otf"),
        Path("fonts/NanumGothic.ttf"),
        Path("fonts/NanumBarunGothic.ttf"),
    ]

    selected_font = None

    for font_path in font_file_candidates:
        if font_path.exists():
            fm.fontManager.addfont(str(font_path))
            selected_font = fm.FontProperties(fname=str(font_path)).get_name()
            plt.rcParams["font.family"] = selected_font
            break

    if not selected_font:
        selected_font = "DejaVu Sans"
        plt.rcParams["font.family"] = selected_font

    plt.rcParams["axes.unicode_minus"] = False
    return selected_font


selected_font = set_korean_font()

# =========================
# 설정
# =========================
api_key = st.secrets.get(
    "OPENAI_API_KEY",
    os.getenv("OPENAI_API_KEY")
)

if not api_key:
    st.error(
        "OPENAI_API_KEY를 불러오지 못했습니다. "
        "로컬(.env.local) 또는 배포 secrets 설정을 확인해주세요."
    )
    st.stop()

client = OpenAI(api_key=api_key)

st.set_page_config(page_title="게임 PM AI 분석기", layout="wide")
st.title("게임 PM AI 분석기 v2")
st.markdown("CSV를 업로드하면 KPI 분석, 위험 신호 탐지, 개선안, 종합 인사이트를 제공합니다.")

if selected_font:
    st.caption(f"그래프 폰트: {selected_font}")
else:
    st.caption("그래프 폰트: 기본 폰트 사용 중")

EXPECTED_COLUMNS = [
    "일자",
    "접속유저", "신규유저", "복귀유저", "기존유저", "누적유저수",
    "최고동접", "평균동접", "플레이시간", "총매출", "개인매출",
    "PU", "PUR", "ARPPU"
]

NUMERIC_COLUMNS = [
    "접속유저", "신규유저", "복귀유저", "기존유저", "누적유저수",
    "최고동접", "평균동접", "플레이시간", "총매출", "개인매출",
    "PU", "PUR", "ARPPU"
]

THRESHOLDS = {
    "접속유저": {"warning": -5, "critical": -10},
    "신규유저": {"warning": -5, "critical": -10},
    "복귀유저": {"warning": -5, "critical": -10},
    "기존유저": {"warning": -3, "critical": -7},
    "평균동접": {"warning": -5, "critical": -10},
    "최고동접": {"warning": -5, "critical": -10},
    "플레이시간": {"warning": -5, "critical": -10},
    "총매출": {"warning": -5, "critical": -10},
    "개인매출": {"warning": -3, "critical": -7},
    "PU": {"warning": -5, "critical": -10},
    "PUR": {"warning": -3, "critical": -7},
    "ARPPU": {"warning": -3, "critical": -7},
}

# =========================
# AI / 텍스트 포맷 유틸
# =========================
def ask_ai(system_role: str, user_prompt: str) -> str:
    response = client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {
                "role": "system",
                "content": (
                    f"{system_role}\n"
                    "당신은 형식 준수에 매우 엄격해야 합니다. "
                    "사용자가 지정한 섹션 제목과 출력 순서를 정확히 지키고, "
                    "불필요한 장식, 가로줄, 코드블록, 표를 생성하지 마세요."
                )
            },
            {"role": "user", "content": user_prompt}
        ]
    )
    return response.output_text.strip()


def ai_format_rules(section_titles: list[str]) -> str:
    titles_text = "\n".join([f"{i+1}. {title}" for i, title in enumerate(section_titles)])

    return f"""
반드시 아래 형식을 정확히 지켜 작성하세요.

[출력 형식 규칙]
- 아래에 제시한 섹션 제목을 정확히 그대로 사용하세요.
- 각 최상위 섹션 제목은 반드시 '숫자. 제목' 형식으로 한 줄 단독 작성하세요.
- 최상위 섹션 제목 앞뒤에 다른 장식 문자(###, **, -, :, 괄호 등)를 붙이지 마세요.
- 가로줄(---, ***, ___)은 절대 출력하지 마세요.
- 코드블록(```)은 절대 사용하지 마세요.
- 표는 사용하지 마세요.
- 각 섹션 아래 내용은 불릿포인트(-) 또는 하위 불릿(  - ) 형태로만 작성하세요.
- 최상위 번호는 한 번만 사용하고, 같은 제목을 반복하지 마세요.
- 섹션 순서는 반드시 아래 순서를 따르세요.
- 불필요한 서론/결론 문장은 쓰지 마세요.
- 숫자 해석은 간결하게 쓰고, PM이 바로 이해할 수 있는 실행 관점으로 작성하세요.

[반드시 사용할 섹션 제목]
{titles_text}
"""


def format_ai_section_text(text: str, section_type: str = "default") -> str:
    """
    AI 응답 텍스트를 Streamlit markdown용으로 후처리한다.

    기능
    - 최상위 번호 제목(예: 1. 지표별 분석, 2. 핵심 변화 요약)을 굵게 처리
    - 개선 방안(section_type='improvement')에서는
      기존 구분선(---)을 제거하고
      '2. 중기 개선안', '3. 바로 실행할 과제' 직전에만 구분선을 삽입
    """
    if not text:
        return ""

    lines = text.splitlines()
    formatted_lines = []

    for raw_line in lines:
        line = raw_line.rstrip()
        stripped = line.strip()

        if not stripped:
            formatted_lines.append("")
            continue

        if section_type == "improvement" and stripped in ("---", "***", "___"):
            continue

        if re.match(r"^\d+\.\s+.+", stripped) and raw_line == raw_line.lstrip():
            if section_type == "improvement" and re.match(r"^(2|3)\.\s+.+", stripped):
                formatted_lines.append("---")
                formatted_lines.append(f"**{stripped}**")
                continue

            formatted_lines.append(f"**{stripped}**")
            continue

        formatted_lines.append(line)

    return "\n\n".join(formatted_lines)


# =========================
# 숫자 / 데이터 유틸
# =========================
def safe_pct_change(current, previous):
    if previous is None or pd.isna(previous) or previous == 0:
        return None
    return ((current - previous) / previous) * 100


def format_pct(val):
    if val is None or pd.isna(val):
        return "계산 불가"
    return f"{val:.2f}%"


def format_num(val):
    if val is None or pd.isna(val):
        return "N/A"
    if isinstance(val, (int, float)):
        return f"{val:,.2f}" if not float(val).is_integer() else f"{int(val):,}"
    return str(val)


def normalize_pur(series: pd.Series):
    return pd.to_numeric(series.astype(str).str.replace(",", ""), errors="coerce")


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]

    if "일자" in df.columns:
        df["일자"] = pd.to_datetime(df["일자"], errors="coerce")
        df = df.sort_values("일자").reset_index(drop=True)

    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            if col == "PUR":
                df[col] = normalize_pur(df[col])
            else:
                df[col] = pd.to_numeric(
                    df[col].astype(str).str.replace(",", ""),
                    errors="coerce"
                )

    return df


def data_quality_check(df: pd.DataFrame):
    issues = []
    infos = []

    missing_cols = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing_cols:
        issues.append(f"누락 컬럼: {', '.join(missing_cols)}")

    if "일자" in df.columns:
        missing_dates = df["일자"].isna().sum()
        if missing_dates > 0:
            issues.append(f"해석 불가능한 일자 값 {missing_dates}건이 있습니다.")

        duplicated_dates = df["일자"].duplicated().sum()
        if duplicated_dates > 0:
            issues.append(f"중복된 일자 {duplicated_dates}건이 있습니다.")

    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            null_cnt = df[col].isna().sum()
            if null_cnt > 0:
                issues.append(f"{col}: 숫자 변환 실패 또는 결측 {null_cnt}건")
            if (df[col].dropna() < 0).any():
                issues.append(f"{col}: 음수 값이 포함되어 있습니다.")

    if "PUR" in df.columns:
        pur_max = df["PUR"].dropna().max() if len(df["PUR"].dropna()) > 0 else None
        if pur_max is not None:
            if pur_max <= 1:
                infos.append("PUR은 0~1 비율값으로 보입니다.")
            else:
                infos.append("PUR은 퍼센트(예: 35) 형태 값으로 보입니다.")

    return issues, infos


def make_metric_summary(df: pd.DataFrame, recent_days: int = 7) -> dict:
    summary = {}

    if "일자" not in df.columns or df["일자"].isna().all():
        recent_df = df.tail(recent_days).copy()
        prev_df = (
            df.iloc[:-recent_days].tail(recent_days).copy()
            if len(df) > recent_days
            else df.head(max(len(df) - 1, 1)).copy()
        )
    else:
        latest_date = df["일자"].max()
        recent_start = latest_date - pd.Timedelta(days=recent_days - 1)
        prev_end = recent_start - pd.Timedelta(days=1)
        prev_start = prev_end - pd.Timedelta(days=recent_days - 1)

        recent_df = df[(df["일자"] >= recent_start) & (df["일자"] <= latest_date)].copy()
        prev_df = df[(df["일자"] >= prev_start) & (df["일자"] <= prev_end)].copy()

    for col in NUMERIC_COLUMNS:
        if col not in df.columns:
            continue

        latest = recent_df[col].dropna().iloc[-1] if len(recent_df[col].dropna()) > 0 else None
        earliest_recent = recent_df[col].dropna().iloc[0] if len(recent_df[col].dropna()) > 0 else None
        current_avg = recent_df[col].mean() if len(recent_df) > 0 else None
        previous_avg = prev_df[col].mean() if len(prev_df) > 0 else None

        summary[col] = {
            "latest": latest,
            "recent_avg": current_avg,
            "previous_avg": previous_avg,
            "recent_avg_vs_prev_avg_pct": safe_pct_change(current_avg, previous_avg),
            "latest_vs_recent_start_pct": safe_pct_change(latest, earliest_recent),
            "recent_min": recent_df[col].min() if len(recent_df) > 0 else None,
            "recent_max": recent_df[col].max() if len(recent_df) > 0 else None,
        }

    return summary


def priority_from_change(metric: str, change):
    if change is None or metric not in THRESHOLDS:
        return None

    if change <= THRESHOLDS[metric]["critical"]:
        return "지금 당장 액션 필요"
    if change <= THRESHOLDS[metric]["warning"]:
        return "주의"
    return "정상"


def severity_icon(metric: str, change):
    if change is None or metric not in THRESHOLDS:
        return "🟢"
    if change <= THRESHOLDS[metric]["critical"]:
        return "🔴"
    if change <= THRESHOLDS[metric]["warning"]:
        return "🟡"
    return "🟢"


def display_pct(metric: str, val):
    if val is None or pd.isna(val):
        return "계산 불가"

    if metric == "PUR":
        return f"{val * 100:.2f}%"

    return f"{val:.2f}%"


def display_num(metric: str, val):
    if val is None or pd.isna(val):
        return "N/A"

    if metric == "PUR":
        return f"{val * 100:.2f}%"

    if isinstance(val, (int, float)):
        return f"{val:,.2f}" if not float(val).is_integer() else f"{int(val):,}"
    return str(val)


def trend_arrow(change):
    if change is None or pd.isna(change):
        return "-"
    if change > 0:
        return "▲"
    if change < 0:
        return "▼"
    return "-"


def make_summary_table(summary: dict):
    rows = []

    for metric, values in summary.items():
        change = values.get("recent_avg_vs_prev_avg_pct")
        priority = priority_from_change(metric, change)
        severity = severity_icon(metric, change)

        rows.append({
            "지표": metric,
            "최신값": display_num(metric, values.get("latest")),
            "최근 평균": display_num(metric, values.get("recent_avg")),
            "이전 평균": display_num(metric, values.get("previous_avg")),
            "변화 방향": trend_arrow(change),
            "최근 평균 vs 이전 평균": display_pct(metric, change),
            "최근 시작 대비 최신": display_pct(metric, values.get("latest_vs_recent_start_pct")),
            "심각도": severity,
            "우선순위": priority if priority else "-"
        })

    return pd.DataFrame(rows)


def make_highlight_summary(summary: dict):
    highlight_lines = []

    key_metrics = ["접속유저", "신규유저", "복귀유저", "총매출", "PU", "PUR", "ARPPU"]

    for metric in key_metrics:
        if metric not in summary:
            continue

        change = summary[metric].get("recent_avg_vs_prev_avg_pct")
        latest = summary[metric].get("latest")
        severity = severity_icon(metric, change)
        priority = priority_from_change(metric, change)

        line = (
            f"{severity} **{metric}** | "
            f"최신값: {display_num(metric, latest)} | "
            f"최근 평균 변화: {display_pct(metric, change)}"
        )

        if priority and priority != "정상":
            line += f" | {priority}"

        highlight_lines.append(line)

    return highlight_lines


def render_kpi_cards(summary: dict):
    card_metrics = ["접속유저", "신규유저", "복귀유저", "총매출", "PUR", "ARPPU"]

    cols = st.columns(3)
    idx = 0

    for metric in card_metrics:
        if metric not in summary:
            continue

        latest = summary[metric].get("latest")
        change = summary[metric].get("recent_avg_vs_prev_avg_pct")
        severity = severity_icon(metric, change)
        priority = priority_from_change(metric, change)

        with cols[idx % 3]:
            st.markdown(
                f"""
**{severity} {metric}**  
최신값: **{display_num(metric, latest)}**  
변화율: **{display_pct(metric, change)}**  
우선순위: **{priority if priority else "-"}**
"""
            )
        idx += 1


def detect_metric_risks(summary: dict):
    rows = []

    for metric, values in summary.items():
        change = values.get("recent_avg_vs_prev_avg_pct")
        if change is None:
            continue

        icon = severity_icon(metric, change)
        priority = priority_from_change(metric, change)

        if icon != "🟢":
            reason = f"최근 평균이 이전 평균 대비 {format_pct(change)} 변동"
            rows.append({
                "지표": metric,
                "변화율": format_pct(change),
                "심각도": icon,
                "우선순위": priority,
                "설명": reason
            })

    return rows


def detect_combined_risks(summary: dict):
    risks = []

    def ch(metric):
        return summary.get(metric, {}).get("recent_avg_vs_prev_avg_pct")

    dau = ch("접속유저")
    new = ch("신규유저")
    ret = ch("복귀유저")
    existing = ch("기존유저")
    revenue = ch("총매출")
    pu = ch("PU")
    pur = ch("PUR")
    arppu = ch("ARPPU")
    playtime = ch("플레이시간")
    avg_ccu = ch("평균동접")

    if dau is not None and revenue is not None and dau <= -8 and revenue <= -8:
        risks.append("🔴 접속유저와 총매출이 함께 하락해 서비스 전반의 활력 저하 가능성이 큽니다.")

    if new is not None and new <= -8 and existing is not None and existing <= -5:
        risks.append("🔴 신규유저 유입 감소와 기존유저 감소가 함께 발생해 유입/정착 모두 약화되고 있습니다.")

    if ret is not None and ret <= -8 and dau is not None and dau <= -5:
        risks.append("🟡 복귀유저 감소가 접속 감소와 함께 나타나 재방문 유도 장치가 약해졌을 수 있습니다.")

    if pu is not None and pu <= -8 and arppu is not None and arppu > 0:
        risks.append("🟡 과금 유저 수는 줄었지만 1인당 결제는 유지되어 소수 고과금 의존 가능성이 있습니다.")

    if pu is not None and pu <= -5 and pur is not None and pur <= -5:
        risks.append("🔴 과금 유저 수와 결제 전환률이 동시에 하락해 BM 구조 점검이 필요합니다.")

    if avg_ccu is not None and avg_ccu <= -8 and playtime is not None and playtime <= -8:
        risks.append("🔴 평균동접과 플레이시간이 함께 감소해 핵심 콘텐츠 몰입도 저하 신호가 있습니다.")

    if revenue is not None and revenue <= -5 and arppu is not None and arppu <= -5:
        risks.append("🟡 총매출과 ARPPU가 함께 하락해 결제 가치 제안이 약해졌을 수 있습니다.")

    return risks


def make_summary_text(summary: dict) -> str:
    lines = []
    for metric, values in summary.items():
        lines.append(
            f"- {metric}: "
            f"최신값={format_num(values['latest'])}, "
            f"최근평균={format_num(values['recent_avg'])}, "
            f"이전평균={format_num(values['previous_avg'])}, "
            f"최근평균 대비 증감률={format_pct(values['recent_avg_vs_prev_avg_pct'])}, "
            f"최근 시작 대비 최신 증감률={format_pct(values['latest_vs_recent_start_pct'])}"
        )
    return "\n".join(lines)


def extract_action_items(summary: dict):
    actions = []

    for metric, values in summary.items():
        change = values.get("recent_avg_vs_prev_avg_pct")
        priority = priority_from_change(metric, change)

        if priority == "지금 당장 액션 필요":
            actions.append({
                "지표": metric,
                "변화율": format_pct(change),
                "액션": f"{metric} 급변 대응 필요"
            })

    return actions


def build_status_message(action_items, metric_risks, combined_risks):
    urgent_count = len(action_items)
    risk_count = len(metric_risks)

    if urgent_count >= 3:
        return "🚨 현재 서비스 상태: 즉각 대응 필요", "error"
    if urgent_count >= 1 or len(combined_risks) >= 1:
        return "⚠️ 현재 서비스 상태: 주의 필요", "warning"
    if risk_count == 0:
        return "✅ 현재 서비스 상태: 전반적으로 안정적입니다.", "success"
    return "ℹ️ 현재 서비스 상태: 일부 변동이 있으나 모니터링이 필요합니다.", "info"


def plot_metric(df: pd.DataFrame, metric: str):
    if metric not in df.columns or "일자" not in df.columns:
        return None

    chart_df = df[["일자", metric]].dropna().copy()
    if len(chart_df) == 0:
        return None

    chart_df = chart_df.sort_values("일자").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(8, 3.5))

    ax.plot(
        chart_df["일자"],
        chart_df[metric],
        marker="o",
        linewidth=2
    )

    ax.set_title(metric, fontsize=14)
    ax.set_xlabel("일자")
    ax.set_ylabel(metric)

    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))

    if metric == "총매출":
        ax.set_ylabel("총매출 (억)")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{x/1e8:.1f}억"))
    elif metric == "PUR":
        ax.set_ylabel("PUR (%)")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{x*100:.1f}%"))
    else:
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{int(x):,}"))

    if len(chart_df) >= 14:
        recent_df = chart_df.tail(7)
        prev_df = chart_df.iloc[-14:-7]

        recent_avg = recent_df[metric].mean()
        prev_avg = prev_df[metric].mean()

        ax.axhline(
            y=recent_avg,
            linestyle="--",
            linewidth=1.5,
            alpha=0.8,
            label=f"최근 7일 평균: {recent_avg/1e8:.1f}억" if metric == "총매출"
                  else f"최근 7일 평균: {recent_avg*100:.1f}%" if metric == "PUR"
                  else f"최근 7일 평균: {recent_avg:,.0f}"
        )

        ax.axhline(
            y=prev_avg,
            linestyle=":",
            linewidth=1.5,
            alpha=0.8,
            label=f"이전 7일 평균: {prev_avg/1e8:.1f}억" if metric == "총매출"
                  else f"이전 7일 평균: {prev_avg*100:.1f}%" if metric == "PUR"
                  else f"이전 7일 평균: {prev_avg:,.0f}"
        )

    metric_mean = chart_df[metric].mean()
    if metric_mean and not pd.isna(metric_mean):
        upper_threshold = metric_mean * 1.15
        lower_threshold = metric_mean * 0.85

        outliers = chart_df[
            (chart_df[metric] >= upper_threshold) |
            (chart_df[metric] <= lower_threshold)
        ]

        if len(outliers) > 0:
            ax.scatter(
                outliers["일자"],
                outliers[metric],
                s=80,
                zorder=5,
                label="이상치"
            )

    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(fontsize=9)

    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig


def section_space():
    st.markdown("<div style='margin-top: 2.2rem;'></div>", unsafe_allow_html=True)


def sub_section_space():
    st.markdown("<div style='margin-top: 1.2rem;'></div>", unsafe_allow_html=True)


# =========================
# 세션 상태 초기화
# =========================
if "analyst_result" not in st.session_state:
    st.session_state.analyst_result = None
if "risk_result" not in st.session_state:
    st.session_state.risk_result = None
if "improvement_result" not in st.session_state:
    st.session_state.improvement_result = None
if "insight_result" not in st.session_state:
    st.session_state.insight_result = None

# =========================
# UI
# =========================
uploaded_file = st.file_uploader("게임 KPI CSV 업로드", type=["csv"])

if uploaded_file:
    try:
        raw_df = pd.read_csv(uploaded_file)
        df = clean_dataframe(raw_df)
    except Exception as e:
        st.error(f"CSV를 읽는 중 오류가 발생했습니다: {e}")
        st.stop()

    summary = make_metric_summary(df, recent_days=7)
    summary_text = make_summary_text(summary)
    metric_risks = detect_metric_risks(summary)
    combined_risks = detect_combined_risks(summary)
    action_items = extract_action_items(summary)

    status_message, status_level = build_status_message(
        action_items=action_items,
        metric_risks=metric_risks,
        combined_risks=combined_risks
    )

    sub_section_space()
    if status_level == "error":
        st.error(status_message)
    elif status_level == "warning":
        st.warning(status_message)
    elif status_level == "success":
        st.success(status_message)
    else:
        st.info(status_message)

    section_space()
    with st.container(border=True):
        st.subheader("🚨 지금 당장 액션 필요 (우선 대응 리스트)")
        if action_items:
            action_df = pd.DataFrame(action_items)
            st.dataframe(action_df, use_container_width=True, hide_index=True)
        else:
            st.success("긴급 대응이 필요한 항목이 없습니다.")

    section_space()
    with st.container(border=True):
        st.subheader("핵심 위험 신호 요약")
        if metric_risks:
            risk_df = pd.DataFrame(metric_risks)
            st.dataframe(risk_df, use_container_width=True, hide_index=True)
        else:
            st.success("임계치 기준 위험 신호가 감지되지 않았습니다.")

        if combined_risks:
            sub_section_space()
            st.markdown("**지표 조합 기반 위험 신호**")
            for risk in combined_risks:
                st.write(f"- {risk}")

    section_space()
    with st.container(border=True):
        st.subheader("핵심 KPI 카드")
        render_kpi_cards(summary)

    section_space()
    with st.container(border=True):
        st.subheader("핵심 지표 한눈에 보기")
        highlight_lines = make_highlight_summary(summary)
        for line in highlight_lines:
            st.markdown(f"- {line}")

    section_space()
    with st.container(border=True):
        st.subheader("지표 요약 표")
        summary_df = make_summary_table(summary)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

    section_space()
    with st.container(border=True):
        st.subheader("주요 지표 그래프")
        graph_cols = st.columns(2)

        chart_metrics = ["접속유저", "신규유저", "복귀유저", "총매출", "PU", "PUR", "ARPPU", "플레이시간"]
        shown = 0
        for metric in chart_metrics:
            fig = plot_metric(df, metric)
            if fig is not None:
                with graph_cols[shown % 2]:
                    st.pyplot(fig)
                shown += 1

    section_space()
    with st.container(border=True):
        st.subheader("AI 종합 분석")

        col_a, col_b = st.columns([1, 5])
        with col_a:
            run_analysis = st.button("AI 종합 분석 시작", use_container_width=True)
        with col_b:
            if st.session_state.insight_result:
                st.caption("분석 결과가 저장되어 있습니다. CSV를 다시 업로드하거나 버튼을 다시 누르면 갱신됩니다.")
            else:
                st.caption("버튼을 누르면 각 지표 분석, 위험 신호, 개선 방안, 종합 인사이트를 생성합니다.")

        if run_analysis:
    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        status_text.info("분석 준비 중... (10%)")
        progress_bar.progress(10)

        status_text.info("1/4 지표 분석 생성 중... (30%)")
        progress_bar.progress(30)
        analyst_result = ask_ai(
            "당신은 모바일 게임 KPI를 해석하는 시니어 데이터 분석가입니다.",
            f"""
아래는 게임 KPI의 사전 요약입니다.

[지표 요약]
{summary_text}

요청:
1. 각 지표에 대한 핵심 해석
2. 특히 의미 있는 상승/하락 지표 강조
3. 숫자 나열보다 PM 관점 해석 중심
4. 각 섹션은 간결하고 명확하게 작성

{ai_format_rules(["지표별 분석", "핵심 변화 요약"])}

추가 작성 규칙:
- '지표별 분석'에서는 각 지표명을 굵게 쓰지 말고 일반 텍스트로 시작하세요.
- 각 지표 해석은 한 줄 또는 두 줄 이내로 짧게 쓰세요.
- '핵심 변화 요약'에서는 가장 중요한 변화만 3~5개로 정리하세요.
"""
        )

        status_text.info("2/4 위험 신호 정리 중... (55%)")
        progress_bar.progress(55)
        risk_result = ask_ai(
            "당신은 라이브 게임 운영 리스크를 점검하는 시니어 운영 전략가입니다.",
            f"""
아래는 KPI 요약과 Python이 감지한 위험 신호입니다.

[지표 요약]
{summary_text}

[지표별 위험]
{pd.DataFrame(metric_risks).to_string(index=False) if metric_risks else "없음"}

[지표 조합 위험]
{chr(10).join(combined_risks) if combined_risks else "없음"}

요청:
1. 개선이 필요한 위험 신호를 중요도 순으로 정리
2. 왜 위험한지 설명
3. 우선순위 기준으로 분류
4. 실제 운영 의사결정에 도움이 되도록 작성

{ai_format_rules(["지금 당장 액션 필요", "주의", "정상"])}

추가 작성 규칙:
- 위험이 없는 섹션도 제목은 유지하고, 내용은 '- 해당 사항 없음'으로 작성하세요.
- '지금 당장 액션 필요'에는 진짜 중요한 항목만 1~3개 넣으세요.
- 각 불릿은 '무슨 문제인지 + 왜 위험한지'가 같이 드러나게 작성하세요.
"""
        )

        status_text.info("3/4 개선 방안 생성 중... (80%)")
        progress_bar.progress(80)
        improvement_result = ask_ai(
            "당신은 게임 사업 PM이자 라이브 운영 전략 전문가입니다.",
            f"""
아래는 KPI 분석 및 위험 신호입니다.

[지표 요약]
{summary_text}

[위험 신호]
{risk_result}

요청:
1. 위험 신호별 개선 방안 제안
2. 단기(이번 주), 중기(이번 달)로 구분
3. 이벤트, BM, UX, 리텐션, 복귀 전략 포함
4. 실행 가능한 액션 아이템으로 작성
5. 각 항목은 실제 팀이 바로 실행할 수 있는 문장으로 작성

{ai_format_rules(["단기 개선안", "중기 개선안", "바로 실행할 과제"])}

추가 작성 규칙:
- 절대로 가로줄(---)을 넣지 마세요.
- '1. 단기 개선안' 바로 아래에는 내용만 작성하세요.
- '2. 중기 개선안'은 단기 개선안 내용이 모두 끝난 뒤에 바로 이어서 작성하세요.
- '바로 실행할 과제'는 체크리스트처럼 아주 구체적으로 작성하세요.
- 각 섹션에는 3~5개 항목만 작성하세요.
- 추상적인 표현보다 실제 액션 중심으로 쓰세요.
- 예: '검토 필요'보다 '신규 가입 24시간 내 지급 보상 A/B 테스트 진행'처럼 작성하세요.
"""
        )

        status_text.info("4/4 종합 인사이트 정리 중... (95%)")
        progress_bar.progress(95)
        insight_result = ask_ai(
            "당신은 게임 사업 PM 총괄입니다. 팀 공유용으로 인사이트를 정리하세요.",
            f"""
아래는 게임 KPI 분석 결과입니다.

[지표 요약]
{summary_text}

[지표 분석]
{analyst_result}

[위험 신호]
{risk_result}

[개선 방안]
{improvement_result}

요청:
1. 서비스 상태 진단
2. 종합 인사이트 3개
3. 가장 먼저 봐야 할 KPI 3개
4. 한 줄 결론
5. 경영진과 실무자가 같이 봐도 이해되게 작성

{ai_format_rules(["서비스 상태 진단", "종합 인사이트", "주목 KPI", "한 줄 결론"])}

추가 작성 규칙:
- '서비스 상태 진단'은 현재 상황을 2~4줄로 요약하세요.
- '종합 인사이트'는 정확히 3개만 작성하세요.
- '주목 KPI'는 KPI명 + 왜 봐야 하는지까지 함께 쓰세요.
- '한 줄 결론'은 한 문장만 작성하세요.
"""
        )

        progress_bar.progress(100)
        status_text.success("분석 완료! 결과를 아래에서 확인하세요.")

        st.session_state.analyst_result = analyst_result
        st.session_state.risk_result = risk_result
        st.session_state.improvement_result = improvement_result
        st.session_state.insight_result = insight_result

    except Exception as e:
        status_text.error(f"분석 중 오류가 발생했습니다: {e}")

        if st.session_state.insight_result:
            section_space()
            st.markdown("---")
            sub_section_space()

            with st.expander("1. 각 지표 분석 결과", expanded=True):
                st.markdown(
                    format_ai_section_text(st.session_state.analyst_result),
                    unsafe_allow_html=False
                )

            sub_section_space()
            with st.expander("2. 위험 신호 정리", expanded=True):
                st.markdown(
                    format_ai_section_text(st.session_state.risk_result),
                    unsafe_allow_html=False
                )

            sub_section_space()
            with st.expander("3. 개선 방안", expanded=True):
                st.markdown(
                    format_ai_section_text(
                        st.session_state.improvement_result,
                        section_type="improvement"
                    ),
                    unsafe_allow_html=False
                )

            sub_section_space()
            with st.expander("4. 종합 인사이트", expanded=True):
                st.markdown(
                    format_ai_section_text(st.session_state.insight_result),
                    unsafe_allow_html=False
                )

            sub_section_space()
            result_text = f"""
[1. 각 지표 분석 결과]
{format_ai_section_text(st.session_state.analyst_result)}

[2. 위험 신호 정리]
{format_ai_section_text(st.session_state.risk_result)}

[3. 개선 방안]
{format_ai_section_text(st.session_state.improvement_result, section_type="improvement")}

[4. 종합 인사이트]
{format_ai_section_text(st.session_state.insight_result)}
"""
            st.download_button(
                label="분석 결과 다운로드 (.txt)",
                data=result_text,
                file_name="game_pm_ai_report.txt",
                mime="text/plain"
            )

    section_space()
    with st.expander("원본 데이터 및 품질 점검 보기", expanded=False):
        st.markdown("### 업로드 데이터 미리보기")
        st.dataframe(df.head(10), use_container_width=True)

        quality_issues, quality_infos = data_quality_check(df)

        sub_section_space()
        st.markdown("### 데이터 품질 점검")
        if quality_issues:
            st.error("점검 결과 이슈가 있습니다.")
            for issue in quality_issues:
                st.write(f"- {issue}")
        else:
            st.success("큰 데이터 품질 이슈는 발견되지 않았습니다.")

        for info in quality_infos:
            st.write(f"- {info}")

        if "일자" not in df.columns:
            st.warning("일자 컬럼이 없으면 정확한 기간 비교가 어려워집니다. 가능하면 일자 컬럼을 포함해주세요.")

else:
    st.info("먼저 CSV 파일을 업로드해주세요.")