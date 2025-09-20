# app.py
import streamlit as st
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# ---- Page title (minimal) ----
st.title("🎂Happy 18th Birthday Mr.Lee")

# ---------------------------
# Helper: parse CSV/space input into float list, safe
def parse_data(text):
    try:
        parts = text.replace(",", " ").split()
        data = [float(p) for p in parts if p.strip() != ""]
        return data
    except Exception:
        return None

# ---------------------------
# FUNCTION 1: Prediction for a new observation
st.subheader("功能 1：预测值检验")

data1_text = st.text_area("请输入功能 1 的样本数据（逗号或空格分隔）：", "82, 85, 90, 87, 88, 91, 84", key="data1")
data1 = parse_data(data1_text)
if data1 is None:
    st.error("❌ 功能1 数据解析错误，请输入数字（逗号或空格分隔）")
    st.stop()
if len(data1) < 2:
    st.error("❌ 功能1 数据至少需要 2 个样本点")
    st.stop()

# separate alpha input for func1
alpha_map = {"90%": 0.10, "95%": 0.05, "99%": 0.01}
conf1_choice = st.selectbox("功能 1 显著性水平 (prediction interval):", list(alpha_map.keys()), index=1, key="alpha1")
alpha1 = alpha_map[conf1_choice]

# prediction value
pred_val = st.number_input("Your prediction (功能1):", value=95.0, key="pred_val")

# compute stats for func1
n1 = len(data1)
mean1 = np.mean(data1)
S1 = np.std(data1, ddof=1)  # sample standard deviation (unbiased denominator n-1)
df1 = n1 - 1

if S1 == 0:
    st.error("❌ 功能1 样本标准差为0，无法进行检验")
else:
    # prediction interval (for a single future observation)
    tcrit1 = stats.t.ppf(1 - alpha1/2, df1)
    pred_low1 = mean1 - tcrit1 * S1 * np.sqrt(1 + 1/n1)
    pred_high1 = mean1 + tcrit1 * S1 * np.sqrt(1 + 1/n1)
    # t-statistic for the prediction (comparison)
    t_pred1 = (pred_val - mean1) / (S1 * np.sqrt(1 + 1/n1))

    st.write(f"样本量 n = {n1}, 样本均值 = {mean1:.4f}, 样本标准差 (S) = {S1:.4f}")
    st.markdown(f"**Prediction interval ({conf1_choice}) = ({pred_low1:.4f}, {pred_high1:.4f})**")
    st.markdown("公式（prediction t）：$$t = \\frac{X_{pred} - \\bar X}{S \\sqrt{1 + 1/n}}$$")
    st.markdown(f"具体计算：$$t = ({pred_val} - {mean1:.4f}) / ({S1:.4f} \\times \\sqrt{{1 + 1/{n1}}}) = {t_pred1:.4f}$$")

    # decision using prediction interval --> critical region wording
    if pred_low1 <= pred_val <= pred_high1:
        st.success(f"✅ 预测值 {pred_val} 落在 prediction interval → 预测值合理")
    else:
        st.error(f"❌ 预测值 {pred_val} 落在 critical region → 预测值不合理")

    # Plot predictive PDF (centered at sample mean, scale = S * sqrt(1+1/n))
    x_min1 = mean1 - 4 * S1 * np.sqrt(1 + 1/n1)
    x_max1 = mean1 + 4 * S1 * np.sqrt(1 + 1/n1)
    x1 = np.linspace(x_min1, x_max1, 600)
    scale_pred = S1 * np.sqrt(1 + 1/n1)
    y1 = stats.t.pdf((x1 - mean1) / scale_pred, df1) / scale_pred

    fig1, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.plot(x1, y1, label="PDF")
    # prediction interval (acceptance region for prediction)
    ax1.fill_between(x1, 0, y1, where=(x1 >= pred_low1) & (x1 <= pred_high1), color="lightgreen", alpha=0.3, label="prediction interval (acceptance region)")
    # critical region as complement (shade)
    ax1.fill_between(x1, 0, y1, where=(x1 < pred_low1) | (x1 > pred_high1), color="lightcoral", alpha=0.2, label="critical region")
    # predicted point
    y_pred_pt = stats.t.pdf((pred_val - mean1) / scale_pred, df1) / scale_pred
    ax1.plot(pred_val, y_pred_pt, 'ro', label=f"Your prediction = {pred_val:.2f}")
    ax1.set_xlabel("Value")
    ax1.set_ylabel("Probability Density")
    ax1.set_title("PDF")
    ax1.grid(True)
    ax1.legend()
    plt.tight_layout()
    st.pyplot(fig1)

# ---------------------------
# FUNCTION 2: Hypothesis test for population mean (separate inputs)
st.subheader("功能 2：样本均值假设检验（双尾 & 单尾）")

data2_text = st.text_area("请输入功能 2 的样本数据（逗号或空格分隔）：", "80, 82, 85, 87, 88", key="data2")
data2 = parse_data(data2_text)
if data2 is None:
    st.error("❌ 功能2 数据解析错误，请输入数字（逗号或空格分隔）")
    st.stop()
if len(data2) < 2:
    st.error("❌ 功能2 数据至少需要 2 个样本点")
    st.stop()

# separate alpha for func2
conf2_choice = st.selectbox("功能 2 显著性水平 (for tests & CI):", list(alpha_map.keys()), index=1, key="alpha2")
alpha2 = alpha_map[conf2_choice]

mu0 = st.number_input("请输入总体均值 μ₀:", value=0.0, key="mu0")

n2 = len(data2)
mean2 = np.mean(data2)
S2 = np.std(data2, ddof=1)
df2 = n2 - 1
if S2 == 0:
    st.error("❌ 功能2 样本标准差为0，无法进行 t 检验")
    st.stop()

# show basic stats
st.write(f"样本量 n = {n2}, 样本均值 = {mean2:.4f}, 样本标准差 (S) = {S2:.4f}")
t_stat2 = (mean2 - mu0) / (S2 / np.sqrt(n2))
st.write(f"t 统计量 = {t_stat2:.4f} (df = {df2})")

# ---------------------------
# Double-tailed test (H0: mu = mu0, H1: mu ≠ mu0)
st.markdown("**双尾检验**  \nH₀: μ = μ₀   &nbsp;&nbsp; H₁: μ ≠ μ₀")

tcrit_two = stats.t.ppf(1 - alpha2/2, df2)
# critical mu boundaries on original axis (mu)
mu_left_two = mu0 - tcrit_two * S2 / np.sqrt(n2)
mu_right_two = mu0 + tcrit_two * S2 / np.sqrt(n2)
# confidence interval for μ (centered at sample mean)
ci_low_mean = mean2 - tcrit_two * S2 / np.sqrt(n2)
ci_high_mean = mean2 + tcrit_two * S2 / np.sqrt(n2)

st.markdown(f"置信区间 ({conf2_choice}) for μ (based on sample) = ({ci_low_mean:.4f}, {ci_high_mean:.4f})")
st.markdown(f"双尾临界 t 值 = ±{tcrit_two:.4f}; 对应 μ 临界边界（以 μ₀ 为中心） = ({mu_left_two:.4f}, {mu_right_two:.4f})")

# use t-statistic to decide (critical region)
if abs(t_stat2) > tcrit_two:
    # reject H0 in two-tailed
    direction_text = ">" if mean2 > mu0 else "<"
    st.error(f"样本均值 {mean2:.2f} 落在临界区 (critical region) → 有足够证据证明 μ {direction_text} μ₀. (|t|={abs(t_stat2):.3f} > t_crit={tcrit_two:.3f})")
else:
    st.info(f"样本均值 {mean2:.2f} 落在接受域 (acceptance region) → 没有足够证据拒绝 H₀. (|t|={abs(t_stat2):.3f} ≤ t_crit={tcrit_two:.3f})")

# Plot 1: Double-tailed PDF (centered at mu0 scale = S/sqrt(n))
x2_min = mu0 - 4 * S2 / np.sqrt(n2)
x2_max = mu0 + 4 * S2 / np.sqrt(n2)
x2 = np.linspace(x2_min, x2_max, 600)
scale_mean = S2 / np.sqrt(n2)
y2 = stats.t.pdf((x2 - mu0) / scale_mean, df2) / scale_mean

fig2, ax2 = plt.subplots(figsize=(8, 4.5))
ax2.plot(x2, y2, label="PDF")
# critical regions (two-tailed) relative to mu0
ax2.fill_between(x2, 0, y2, where=(x2 < mu_left_two) | (x2 > mu_right_two), color="lightcoral", alpha=0.3, label="critical region (two-tailed)")
ax2.fill_between(x2, 0, y2, where=(x2 >= mu_left_two) & (x2 <= mu_right_two), color="lightgreen", alpha=0.25, label="acceptance region (two-tailed)")
# sample mean marker
y_mean2_on_mu0 = stats.t.pdf((mean2 - mu0) / scale_mean, df2) / scale_mean
ax2.plot(mean2, y_mean2_on_mu0, 'ro', label=f"Sample mean = {mean2:.2f}")
ax2.set_xlabel("Value")
ax2.set_ylabel("Probability Density")
ax2.set_title("PDF")
ax2.grid(True)
ax2.legend()
plt.tight_layout()
st.pyplot(fig2)

# ---------------------------
# Single-tailed test (automatic direction based on mean vs mu0)
st.markdown("**单尾检验（自动选择方向）**")
if mean2 > mu0:
    # right-tailed
    tail_text = "右尾 (H₁: μ > μ₀)"
    tcrit_one = stats.t.ppf(1 - alpha2, df2)
    mu_crit_one = mu0 + tcrit_one * S2 / np.sqrt(n2)
    conf_low_one = mu0
    conf_high_one = mu_crit_one
    st.markdown("H₀: μ = μ₀   &nbsp;&nbsp; H₁: μ > μ₀")
else:
    # left-tailed
    tail_text = "左尾 (H₁: μ < μ₀)"
    tcrit_one = stats.t.ppf(1 - alpha2, df2)
    mu_crit_one = mu0 - tcrit_one * S2 / np.sqrt(n2)
    conf_low_one = mu_crit_one
    conf_high_one = mu0
    st.markdown("H₀: μ = μ₀   &nbsp;&nbsp; H₁: μ < μ₀")

st.markdown(f"{tail_text} 单尾临界 t 值 = {tcrit_one:.4f}; 对应 μ 临界边界 = {mu_crit_one:.4f}")
st.markdown(f"单尾{conf2_choice} 区间（基于 μ₀ 的方向） = ({conf_low_one:.4f}, {conf_high_one:.4f})")

# decide single-tail using t-statistic
if mean2 > mu0:
    reject_one = t_stat2 > tcrit_one
else:
    reject_one = t_stat2 < -tcrit_one

if reject_one:
    direction = "大于 μ₀" if mean2 > mu0 else "小于 μ₀"
    st.error(f"样本均值 {mean2:.2f} 落在临界区 (critical region) → 有足够证据证明 μ {direction}. (t={t_stat2:.3f}, t_crit={tcrit_one:.3f})")
else:
    st.info(f"样本均值 {mean2:.2f} 落在接受域 (acceptance region) → 没有足够证据拒绝 H₀. (t={t_stat2:.3f}, t_crit={tcrit_one:.3f})")

# Plot 2: Single-tailed PDF (showing one-side acceptance region)
x3_min = conf_low_one - 0.5 * S2 / np.sqrt(n2)
x3_max = conf_high_one + 0.5 * S2 / np.sqrt(n2)
x3 = np.linspace(x3_min, x3_max, 600)
y3 = stats.t.pdf((x3 - mu0) / scale_mean, df2) / scale_mean

fig3, ax3 = plt.subplots(figsize=(8, 4.5))
ax3.plot(x3, y3, label="PDF")
ax3.fill_between(x3, 0, y3, where=(x3 >= conf_low_one) & (x3 <= conf_high_one), color="lightgreen", alpha=0.3, label="acceptance region (single-tail)")
ax3.fill_between(x3, 0, y3, where=(x3 < conf_low_one) | (x3 > conf_high_one), color="lightcoral", alpha=0.2, label="critical region (single-tail)")
y_mean3_on_mu0 = stats.t.pdf((mean2 - mu0) / scale_mean, df2) / scale_mean
ax3.plot(mean2, y_mean3_on_mu0, 'ro', label=f"Sample mean = {mean2:.2f}")
ax3.set_xlabel("Value")
ax3.set_ylabel("Probability Density")
ax3.set_title("PDF")
ax3.grid(True)
ax3.legend()
plt.tight_layout()
st.pyplot(fig3)
