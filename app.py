import streamlit as st
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

st.title("🎂Happy 18th Birthday Mr.Lee")

st.markdown("""
亲爱的 lzh叔叔【笑】，这个工具可以通过你给的样本预测一些数据【t分布，即便样本数小也可以用】。

你可以输入几次数学成绩，它会通过样本均值和样本方差呈现分布 PDF，你可以尝试预测能考多少分，并判断预测值合理性。

然后可以选择显著性水平 significance level。
""")

# ---------------------------
# 用户输入数据
st.subheader("数据输入")
data_input = st.text_area("请输入你的数据（用逗号或空格分隔）:", "82, 85, 90, 87, 88, 91, 84")
try:
    data = [float(x) for x in data_input.replace(",", " ").split()]
except:
    st.error("❌ 数据格式错误，请输入数字")
    st.stop()

if len(data) < 2:
    st.error("❌ 数据量至少需要两个点")
    st.stop()

n = len(data)
mean = np.mean(data)
S = np.std(data, ddof=1)
df = n - 1

# ---------------------------
# 用户选择显著性水平
alpha_map = {"90%": 0.10, "95%": 0.05, "99%": 0.01}
conf_choice = st.radio("选择置信水平:", list(alpha_map.keys()), index=1)
alpha = alpha_map[conf_choice]

# ---------------------------
# 样本统计量显示
st.subheader("样本统计量")
st.write(f"样本量 n = {n}, 样本均值 = {mean:.4f}, 样本标准差 = {S:.4f}")

mean_formula = f"\\bar{{X}} = ( {' + '.join([str(x) for x in data])} ) / {n} = {mean:.4f}"
st.markdown(f"样本均值公式：$$\\bar{{X}} = \\frac{{1}}{{n}} \\sum_{{i=1}}^{{n}} X_i$$  \n具体计算：$$ {mean_formula} $$")

deviations = [f"({x}-{mean:.2f})^2" for x in data]
S2 = S**2
S2_formula = f"S^2 = ( {' + '.join(deviations) } ) / ( {n}-1 ) = {S2:.4f}"
st.markdown(f"样本方差公式：$$S^2 = \\frac{{1}}{{n-1}} \sum_{{i=1}}^{{n}} (X_i - \\bar{{X}})^2$$  \n具体计算：$$ {S2_formula} $$")

st.markdown(f"样本标准差公式：$$S = \\sqrt{{S^2}}$$  \n具体计算：$$S = \\sqrt{{{S2:.4f}}} = {S:.4f}$$")

# ---------------------------
# 功能 1：预测值检验
st.subheader("📊 功能 1：预测值检验")
user_prediction = st.number_input("Your prediction:", value=95.0)

# t统计量和预测区间
t_crit = stats.t.ppf(1 - alpha/2, df)
pred_low = mean - t_crit * S * np.sqrt(1 + 1/n)
pred_high = mean + t_crit * S * np.sqrt(1 + 1/n)
t_pred = (user_prediction - mean) / (S * np.sqrt(1 + 1/n))

st.markdown(f"预测值置信区间 ({conf_choice}) = ({pred_low:.4f}, {pred_high:.4f})")
st.markdown(f"公式：$$t = \\frac{{X_{{pred}} - \\bar{{X}}}}{{S\\sqrt{{1 + 1/n}}}}$$")
st.markdown(f"具体计算：$$t = ({user_prediction} - {mean:.4f}) / ({S:.4f} * sqrt(1 + 1/{n})) = {t_pred:.4f}$$")

if pred_low <= user_prediction <= pred_high:
    st.success(f"✅ 预测值落在置信区间 → 预测值合理")
else:
    st.error(f"❌ 预测值落在置信区间外 → 预测值不合理")

# 绘制预测值 PDF
x = np.linspace(mean - 4*S, mean + 4*S, 500)
y = stats.t.pdf((x - mean)/S, df)/S

fig, ax = plt.subplots(figsize=(8,5))
ax.plot(x, y, label="PDF")
ax.fill_between(x, 0, y, where=(x >= pred_low) & (x <= pred_high), color="lightgreen", alpha=0.3, label="confidence interval")
ax.plot(user_prediction, stats.t.pdf((user_prediction-mean)/S, df)/S, 'ro', label="Your prediction")
ax.set_xlabel("t")
ax.set_ylabel("Probability Density")
ax.set_title("PDF")
ax.grid(True)
ax.legend()
plt.tight_layout()
st.pyplot(fig)

# ---------------------------
# 功能 2：样本均值假设检验
st.subheader("功能 2：样本均值假设检验")
mu0 = st.number_input("请输入总体均值 μ₀:", value=0.0)

# 样本均值 t 统计量
t_stat = (mean - mu0)/(S/np.sqrt(n))
st.markdown(f"样本均值 t 统计量 = {t_stat:.4f}")

# ---------------------------
# 双尾检验
st.markdown("### 双尾检验")
st.markdown("H₀: μ = μ₀  \nH₁: μ ≠ μ₀")

t_crit_two = stats.t.ppf(1 - alpha/2, df)
conf_low_two = mu0 - t_crit_two*S/np.sqrt(n)
conf_high_two = mu0 + t_crit_two*S/np.sqrt(n)

st.markdown(f"置信区间 ({conf_choice}) = ({conf_low_two:.4f}, {conf_high_two:.4f})")

if conf_low_two <= mean <= conf_high_two:
    st.info(f"样本均值 {mean:.2f} 落在置信区间 → 没有足够证据证明 μ ≠ μ₀")
else:
    direction = "大于 μ₀" if mean > mu0 else "小于 μ₀"
    st.error(f"样本均值 {mean:.2f} 落在置信区间外 → 有足够证据证明 μ {direction}")

# 绘图 PDF 双尾
x2 = np.linspace(mu0 - 4*S/np.sqrt(n), mu0 + 4*S/np.sqrt(n), 500)
y2 = stats.t.pdf((x2 - mu0)/(S/np.sqrt(n)), df)/(S/np.sqrt(n))
fig2, ax2 = plt.subplots(figsize=(8,5))
ax2.plot(x2, y2, label="PDF")
ax2.fill_between(x2, 0, y2, where=(x2 >= conf_low_two) & (x2 <= conf_high_two), color="lightgreen", alpha=0.3, label="confidence interval")
y_mean2 = stats.t.pdf((mean - mu0)/(S/np.sqrt(n)), df)/(S/np.sqrt(n))
ax2.plot(mean, y_mean2, 'ro', label=f"Sample mean = {mean:.2f}")
ax2.set_xlabel("t")
ax2.set_ylabel("Probability Density")
ax2.set_title("PDF")
ax2.grid(True)
ax2.legend()
plt.tight_layout()
st.pyplot(fig2)

# ---------------------------
# 单尾检验
st.markdown("### 单尾检验")
if mean > mu0:
    tail_text = "右尾 (μ > μ₀)"
    t_crit_one = stats.t.ppf(1 - alpha, df)
    conf_low_one = mu0
    conf_high_one = mu0 + t_crit_one*S/np.sqrt(n)
else:
    tail_text = "左尾 (μ < μ₀)"
    t_crit_one = stats.t.ppf(1 - alpha, df)
    conf_low_one = mu0 - t_crit_one*S/np.sqrt(n)
    conf_high_one = mu0

st.markdown("H₀: μ = μ₀  \nH₁: μ > μ₀" if mean > mu0 else "H₀: μ = μ₀  \nH₁: μ < μ₀")
st.markdown(f"置信区间 ({conf_choice}) ({tail_text}) = ({conf_low_one:.4f}, {conf_high_one:.4f})")

if conf_low_one <= mean <= conf_high_one:
    st.info(f"样本均值 {mean:.2f} 落在置信区间 → 没有足够证据证明 μ {tail_text}")
else:
    st.error(f"样本均值 {mean:.2f} 落在置信区间外 → 有足够证据证明 μ {tail_text}")

# 绘图 PDF 单尾
x3 = np.linspace(conf_low_one - 0.5*S/np.sqrt(n), conf_high_one + 0.5*S/np.sqrt(n), 500)
y3 = stats.t.pdf((x3 - mu0)/(S/np.sqrt(n)), df)/(S/np.sqrt(n))
fig3, ax3 = plt.subplots(figsize=(8,5))
ax3.plot(x3, y3, label="PDF")
ax3.fill_between(x3, 0, y3, where=(x3 >= conf_low_one) & (x3 <= conf_high_one), color="lightgreen", alpha=0.3, label="confidence interval")
y_mean3 = stats.t.pdf((mean - mu0)/(S/np.sqrt(n)), df)/(S/np.sqrt(n))
ax3.plot(mean, y_mean3, 'ro', label=f"Sample mean = {mean:.2f}")
ax3.set_xlabel("t")
ax3.set_ylabel("Probability Density")
ax3.set_title("PDF")
ax3.grid(True)
ax3.legend()
plt.tight_layout()
st.pyplot(fig3)
