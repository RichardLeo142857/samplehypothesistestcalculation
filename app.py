import streamlit as st
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

st.title("🎂Happy 18th Birthday Mr.Lee")

# ---------------------------
# 功能 1：预测值检验
st.subheader("功能 1：预测值检验")

# 功能1数据输入
data1_input = st.text_area("请输入功能 1 的样本数据（逗号或空格分隔）:", "82, 85, 90, 87, 88, 91, 84")
try:
    data1 = [float(x) for x in data1_input.replace(",", " ").split()]
except:
    st.error("❌ 数据格式错误，请输入数字")
    st.stop()

if len(data1) < 2:
    st.error("❌ 数据量至少需要两个点")
    st.stop()

n1 = len(data1)
mean1 = np.mean(data1)
S1 = np.std(data1, ddof=1)
df1 = n1 - 1

# 功能1显著性水平
alpha_map = {"90%": 0.10, "95%": 0.05, "99%": 0.01}
conf1_choice = st.radio("功能 1 置信水平:", list(alpha_map.keys()), index=1, key="f1")
alpha1 = alpha_map[conf1_choice]

# 用户预测值
user_prediction = st.number_input("Your prediction:", value=95.0, key="pred1")

# 预测值置信区间
t_crit1 = stats.t.ppf(1 - alpha1/2, df1)
pred_low1 = mean1 - t_crit1 * S1 * np.sqrt(1 + 1/n1)
pred_high1 = mean1 + t_crit1 * S1 * np.sqrt(1 + 1/n1)
t_pred1 = (user_prediction - mean1) / (S1 * np.sqrt(1 + 1/n1))

st.write(f"样本均值 = {mean1:.4f}, 样本标准差 = {S1:.4f}, 样本量 n = {n1}")
st.markdown(f"预测值置信区间 ({conf1_choice}) = ({pred_low1:.4f}, {pred_high1:.4f})")
st.markdown(f"公式：$$t = \\frac{{X_{{pred}} - \\bar{{X}}}}{{S\\sqrt{{1 + 1/n}}}}$$")
st.markdown(f"计算：$$t = ({user_prediction} - {mean1:.4f}) / ({S1:.4f} * sqrt(1 + 1/{n1})) = {t_pred1:.4f}$$")

if pred_low1 <= user_prediction <= pred_high1:
    st.success(f"✅ 预测值落在置信区间 → 预测值合理")
else:
    st.error(f"❌ 预测值落在置信区间外 → 预测值不合理")

# 绘制 PDF 图
x1 = np.linspace(mean1 - 4*S1, mean1 + 4*S1, 500)
y1 = stats.t.pdf((x1 - mean1)/S1, df1)/S1

fig1, ax1 = plt.subplots(figsize=(8,5))
ax1.plot(x1, y1, label="PDF")
ax1.fill_between(x1, 0, y1, where=(x1 >= pred_low1) & (x1 <= pred_high1), color="lightgreen", alpha=0.3, label="confidence interval")
ax1.plot(user_prediction, stats.t.pdf((user_prediction-mean1)/S1, df1)/S1, 'ro', label="Your prediction")
ax1.set_xlabel("t")
ax1.set_ylabel("Probability Density")
ax1.set_title("PDF")
ax1.grid(True)
ax1.legend()
plt.tight_layout()
st.pyplot(fig1)

# ---------------------------
# 功能 2：样本均值假设检验
st.subheader("功能 2：样本均值假设检验")

# 功能2数据输入
data2_input = st.text_area("请输入功能 2 的样本数据（逗号或空格分隔）:", "80, 82, 85, 87, 88")
try:
    data2 = [float(x) for x in data2_input.replace(",", " ").split()]
except:
    st.error("❌ 数据格式错误，请输入数字")
    st.stop()

if len(data2) < 2:
    st.error("❌ 数据量至少需要两个点")
    st.stop()

n2 = len(data2)
mean2 = np.mean(data2)
S2 = np.std(data2, ddof=1)
df2 = n2 - 1

# 功能2显著性水平
conf2_choice = st.radio("功能 2 置信水平:", list(alpha_map.keys()), index=1, key="f2")
alpha2 = alpha_map[conf2_choice]

# 用户输入总体均值 μ₀
mu0 = st.number_input("请输入总体均值 μ₀:", value=0.0, key="mu0")

st.write(f"样本均值 = {mean2:.4f}, 样本标准差 = {S2:.4f}, 样本量 n = {n2}")
t_stat2 = (mean2 - mu0)/(S2/np.sqrt(n2))
st.write(f"t 统计量 = {t_stat2:.4f}")

# ---------------------------
# 双尾检验
st.markdown("双尾检验")
st.markdown("H₀: μ = μ₀  \nH₁: μ ≠ μ₀")

t_crit_two = stats.t.ppf(1 - alpha2/2, df2)
conf_low_two = mu0 - t_crit_two*S2/np.sqrt(n2)
conf_high_two = mu0 + t_crit_two*S2/np.sqrt(n2)

st.write(f"置信区间 ({conf2_choice}) = ({conf_low_two:.4f}, {conf_high_two:.4f})")

if conf_low_two <= mean2 <= conf_high_two:
    st.info(f"样本均值 {mean2:.2f} 落在置信区间 → 没有足够证据拒绝 H₀")
else:
    direction = "大于 μ₀" if mean2 > mu0 else "小于 μ₀"
    st.error(f"样本均值 {mean2:.2f} 落在临界区 (critical region) → 有足够证据证明 μ {direction}")

# 绘图 PDF 双尾
x2 = np.linspace(mu0 - 4*S2/np.sqrt(n2), mu0 + 4*S2/np.sqrt(n2), 500)
y2 = stats.t.pdf((x2 - mu0)/(S2/np.sqrt(n2)), df2)/(S2/np.sqrt(n2))
fig2, ax2 = plt.subplots(figsize=(8,5))
ax2.plot(x2, y2, label="PDF")
ax2.fill_between(x2, 0, y2, where=(x2 >= conf_low_two) & (x2 <= conf_high_two), color="lightgreen", alpha=0.3, label="confidence interval")
y_mean2 = stats.t.pdf((mean2 - mu0)/(S2/np.sqrt(n2)), df2)/(S2/np.sqrt(n2))
ax2.plot(mean2, y_mean2, 'ro', label=f"Sample mean = {mean2:.2f}")
ax2.set_xlabel("t")
ax2.set_ylabel("Probability Density")
ax2.set_title("PDF")
ax2.grid(True)
ax2.legend()
plt.tight_layout()
st.pyplot(fig2)

# ---------------------------
# 单尾检验
st.markdown("单尾检验")
if mean2 > mu0:
    tail_text = "右尾 (μ > μ₀)"
    t_crit_one = stats.t.ppf(1 - alpha2, df2)
    conf_low_one = mu0
    conf_high_one = mu0 + t_crit_one*S2/np.sqrt(n2)
else:
    tail_text = "左尾 (μ < μ₀)"
    t_crit_one = stats.t.ppf(1 - alpha2, df2)
    conf_low_one = mu0 - t_crit_one*S2/np.sqrt(n2)
    conf_high_one = mu0

st.markdown("H₀: μ = μ₀  \nH₁: μ > μ₀" if mean2 > mu0 else "H₀: μ = μ₀  \nH₁: μ < μ₀")
st.write(f"置信区间 ({conf2_choice}) ({tail_text}) = ({conf_low_one:.4f}, {conf_high_one:.4f})")

if conf_low_one <= mean2 <= conf_high_one:
    st.info(f"样本均值 {mean2:.2f} 落在置信区间 → 没有足够证据拒绝 H₀")
else:
    st.error(f"样本均值 {mean2:.2f} 落在临界区 (critical region) → 有足够证据证明 μ {tail_text}")

# 绘图 PDF 单尾
x3 = np.linspace(conf_low_one - 0.5*S2/np.sqrt(n2), conf_high_one + 0.5*S2/np.sqrt(n2), 500)
y3 = stats.t.pdf((x3 - mu0)/(S2/np.sqrt(n2)), df2)/(S2/np.sqrt(n2))
fig3, ax3 = plt.subplots(figsize=(8,5))
ax3.plot(x3, y3, label="PDF")
ax3.fill_between(x3, 0, y3, where=(x3 >= conf_low_one) & (x3 <= conf_high_one), color="lightgreen", alpha=0.3, label="confidence interval")
y_mean3 = stats.t.pdf((mean2 - mu0)/(S2/np.sqrt(n2)), df2)/(S2/np.sqrt(n2))
ax3.plot(mean2, y_mean3, 'ro', label=f"Sample mean = {mean2:.2f}")
ax3.set_xlabel("t")
ax3.set_ylabel("Probability Density")
ax3.set_title("PDF")
ax3.grid(True)
ax3.legend()
plt.tight_layout()
st.pyplot(fig3)
