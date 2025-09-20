# app.py
import streamlit as st
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

st.title("🎂Happy 18th Birthday Mr.Lee")

# --- helper to parse input ---
def parse_data(text):
    try:
        parts = text.replace(",", " ").split()
        return [float(p) for p in parts if p.strip() != ""]
    except:
        return None

# ---------------------------
# Minimal Feature 1 (kept simple here)
st.subheader("功能 1：预测值检验（独立输入）")
data1_text = st.text_area("功能1 样本数据（逗号或空格分隔）：", "82, 85, 90, 87, 88, 91, 84", key="data1")
data1 = parse_data(data1_text)
if data1 is None or len(data1) < 2:
    st.error("❌ 功能1 数据解析错误或样本太少（≥2）")
else:
    n1 = len(data1)
    mean1 = np.mean(data1)
    S1 = np.std(data1, ddof=1)
    df1 = n1 - 1
    alpha_map = {"90%": 0.10, "95%": 0.05, "99%": 0.01}
    conf1_choice = st.selectbox("功能1 选择 prediction-significance α：", list(alpha_map.keys()), index=1, key="a1")
    alpha1 = alpha_map[conf1_choice]
    pred_val = st.number_input("Your prediction (功能1):", value=95.0, key="pred1")

    if S1 == 0:
        st.error("❌ 功能1 样本标准差为0，无法计算")
    else:
        tcrit1 = stats.t.ppf(1 - alpha1/2, df1)
        pred_low1 = mean1 - tcrit1 * S1 * np.sqrt(1 + 1/n1)
        pred_high1 = mean1 + tcrit1 * S1 * np.sqrt(1 + 1/n1)
        t_pred1 = (pred_val - mean1) / (S1 * np.sqrt(1 + 1/n1))

        st.write(f"样本量 n = {n1}, 样本均值 = {mean1:.4f}, S = {S1:.4f}")
        st.markdown(f"Prediction interval ({conf1_choice}) = ({pred_low1:.4f}, {pred_high1:.4f})")
        st.markdown("公式：$$t=\\frac{X_{pred}-\\bar X}{S\\sqrt{1+1/n}}$$")
        st.markdown(f"计算：t = ({pred_val} - {mean1:.4f}) / ({S1:.4f}*sqrt(1+1/{n1})) = {t_pred1:.4f}")

        if pred_low1 <= pred_val <= pred_high1:
            st.success(f"✅ 预测值 {pred_val} 落在 prediction interval → 预测值合理")
        else:
            st.error(f"❌ 预测值 {pred_val} 落在 critical region → 预测值不合理")

        # plot
        x_min1 = mean1 - 4 * S1 * np.sqrt(1 + 1/n1)
        x_max1 = mean1 + 4 * S1 * np.sqrt(1 + 1/n1)
        x1 = np.linspace(x_min1, x_max1, 500)
        scale_pred = S1 * np.sqrt(1 + 1/n1)
        y1 = stats.t.pdf((x1 - mean1) / scale_pred, df1) / scale_pred

        fig1, ax1 = plt.subplots(figsize=(8,4))
        ax1.plot(x1, y1, label="PDF")
        ax1.fill_between(x1, 0, y1, where=(x1 >= pred_low1) & (x1 <= pred_high1), color="lightgreen", alpha=0.3, label="prediction interval")
        ax1.fill_between(x1, 0, y1, where=(x1 < pred_low1) | (x1 > pred_high1), color="lightcoral", alpha=0.15, label="critical region")
        ypredpt = stats.t.pdf((pred_val - mean1) / scale_pred, df1) / scale_pred
        ax1.plot(pred_val, ypredpt, 'ro', label=f"Your prediction = {pred_val:.2f}")
        ax1.set_xlabel("Value")
        ax1.set_ylabel("Probability Density")
        ax1.set_title("PDF")
        ax1.legend()
        ax1.grid(True)
        plt.tight_layout()
        st.pyplot(fig1)

# ---------------------------
# FUNCTION 2: Single-tail only, use p-value for conclusion (no table)
st.subheader("功能 2：样本均值假设检验（仅单尾）")

data2_text = st.text_area("功能2 样本数据（逗号或空格分隔）：", "80, 82, 85, 87, 88", key="data2")
data2 = parse_data(data2_text)
if data2 is None or len(data2) < 2:
    st.error("❌ 功能2 数据解析错误或样本太少（≥2）")
else:
    n2 = len(data2)
    mean2 = np.mean(data2)
    S2 = np.std(data2, ddof=1)
    df2 = n2 - 1

    alpha_map2 = {"90%": 0.10, "95%": 0.05, "99%": 0.01}
    alpha2_choice = st.selectbox("功能2 选择显著性水平 α（用于决策）:", list(alpha_map2.keys()), index=1, key="a2")
    alpha2 = alpha_map2[alpha2_choice]

    mu0 = st.number_input("请输入总体均值 μ₀:", value=0.0, key="mu0_2")

    st.write(f"样本量 n = {n2}, 样本均值 = {mean2:.4f}, S = {S2:.4f}, df = {df2}")

    # t statistic
    t_stat = (mean2 - mu0) / (S2 / np.sqrt(n2))
    st.write(f"t 统计量 = {t_stat:.4f}")

    # determine tail direction automatically and compute one-sided p-value
    if mean2 > mu0:
        tail_dir = "right"   # H1: mu > mu0
        Htext = "H₀: μ = μ₀   |   H₁: μ > μ₀ (right-tailed)"
        p_one = 1 - stats.t.cdf(t_stat, df2)
    else:
        tail_dir = "left"    # H1: mu < mu0
        Htext = "H₀: μ = μ₀   |   H₁: μ < μ₀ (left-tailed)"
        p_one = stats.t.cdf(t_stat, df2)

    st.markdown(Htext)
    st.write(f"one-sided p-value = {p_one:.6f}")

    # Decision based on p-value and chosen alpha2 (NOT by comparing t vs critical)
    if p_one < alpha2:
        # reject H0 at chosen alpha
        if tail_dir == "right":
            st.error(f"❌ p-value = {p_one:.6f} < α = {alpha2:.2f} → 有足够证据证明 μ > μ₀。  （样本均值 {mean2:.2f} > μ₀ = {mu0:.2f}）")
        else:
            st.error(f"❌ p-value = {p_one:.6f} < α = {alpha2:.2f} → 有足够证据证明 μ < μ₀。  （样本均值 {mean2:.2f} < μ₀ = {mu0:.2f}）")
    else:
        st.success(f"✅ p-value = {p_one:.6f} ≥ α = {alpha2:.2f} → 没有足够证据拒绝 H₀。  （样本均值 {mean2:.2f}）")

    # --- Plot single-tail PDF centered at mu0 scale = S / sqrt(n) ---
    scale_mean = S2 / np.sqrt(n2)
    x_min = mu0 - 4 * scale_mean
    x_max = mu0 + 4 * scale_mean
    x = np.linspace(x_min, x_max, 600)
    y = stats.t.pdf((x - mu0) / scale_mean, df2) / scale_mean

    # compute the mu critical boundary for chosen alpha2
    tcrit_chosen = stats.t.ppf(1 - alpha2, df2)
    if tail_dir == "right":
        mu_crit_chosen = mu0 + tcrit_chosen * S2 / np.sqrt(n2)
        accept_cond = (x >= mu0) & (x <= mu_crit_chosen)
        crit_cond = x > mu_crit_chosen
    else:
        mu_crit_chosen = mu0 - tcrit_chosen * S2 / np.sqrt(n2)
        accept_cond = (x >= mu_crit_chosen) & (x <= mu0)
        crit_cond = x < mu_crit_chosen

    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(x, y, label="PDF")
    # acceptance region (green)
    ax.fill_between(x, 0, y, where=accept_cond, color="lightgreen", alpha=0.3, label="acceptance region (based on chosen α)")
    # critical region (red)
    ax.fill_between(x, 0, y, where=crit_cond, color="lightcoral", alpha=0.25, label="critical region (based on chosen α)")
    # mark mu0, mu_critical and sample mean
    ax.axvline(mu0, color="black", linestyle="--", linewidth=1, label=f"μ₀ = {mu0:.2f}")
    ax.axvline(mu_crit_chosen, color="orange", linestyle="--", linewidth=1, label=f"μ_crit (α={alpha2:.2f}) = {mu_crit_chosen:.2f}")
    y_mean_on_scale = stats.t.pdf((mean2 - mu0) / scale_mean, df2) / scale_mean
    ax.plot(mean2, y_mean_on_scale, 'ro', label=f"Sample mean = {mean2:.2f}")
    ax.set_xlabel("Value")
    ax.set_ylabel("Probability Density")
    ax.set_title("PDF")
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)
