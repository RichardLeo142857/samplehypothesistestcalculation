import streamlit as st
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

def main():
    st.title("🎂Happy 18th Birthday Mr.Lee")

    st.markdown("""
亲爱的 lzh叔叔【笑】，这个工具可以通过你给的样本预测一些数据【我这里用的是 t 分布因此即便你样本数比较小也可以用】。

比方说你可以试着输入你几次数学成绩，然后它会通过这个样本均值和样本方差来呈现你分数分布的 PDF，你可以尝试预估你能考多少分，然后它就会根据你已有的数据来进行假设检验判断你的预估合不合理（不过对你来说 150 可能也是合理的叭 LOL）。

然后你可以自己选择合适的显著性水平 significance level，假设你在卖一个产品你着急说明你的产品有效，你大可把你的 significance level 设得低点 bushi。
""")

    # ---------------------------
    # 用户输入数据
    st.subheader("数据输入")
    data_input = st.text_area("请输入你的数据（用逗号或空格分隔）:", "82, 85, 90, 87, 88, 91, 84")
    try:
        data = [float(x) for x in data_input.replace(",", " ").split()]
    except:
        st.error("❌ 数据格式错误，请输入数字，用逗号或空格分隔")
        return

    if len(data) < 2:
        st.error("❌ 数据量至少需要两个点")
        return

    n = len(data)
    mean = np.mean(data)
    S2 = np.var(data, ddof=1)
    S = np.sqrt(S2)
    df = n - 1

    # ---------------------------
    # 用户选择显著性水平
    alpha_map = {"90%": 0.10, "95%": 0.05, "99%": 0.01}
    conf_choice = st.radio("选择置信水平:", list(alpha_map.keys()), index=1)
    alpha = alpha_map[conf_choice]

    # ---------------------------
    # 样本统计量显示
    st.subheader("📌 样本统计量")
    st.write(f"样本量 n = {n}")

    mean_formula = f"\\bar{{X}} = ( {' + '.join([str(x) for x in data])} ) / {n} = {mean:.4f}"
    st.markdown(f"样本均值 = **{mean:.4f}**  \n公式：$$\\bar{{X}} = \\frac{{1}}{{n}} \\sum_{{i=1}}^{{n}} X_i$$  \n具体计算：$$ {mean_formula} $$")

    deviations = [f"({x}-{mean:.2f})^2" for x in data]
    S2_formula = f"S^2 = ( {' + '.join(deviations) } ) / ( {n}-1 ) = {S2:.4f}"
    st.markdown(f"样本方差 = **{S2:.4f}**  \n公式：$$S^2 = \\frac{{1}}{{n-1}} \\sum_{{i=1}}^{{n}} (X_i - \\bar{{X}})^2$$  \n具体计算：$$ {S2_formula} $$")

    st.markdown(f"样本标准差 = **{S:.4f}**  \n公式：$$S = \\sqrt{{S^2}}$$  \n具体计算：$$S = \\sqrt{{{S2:.4f}}} = {S:.4f}$$")

    # ---------------------------
    # 功能 1：预测值检验
    st.subheader("📊 功能 1：预测值检验")
    user_prediction = st.number_input("Your prediction:", value=95.0)
    t_crit = stats.t.ppf(1 - alpha/2, df)
    pred_low = mean - t_crit * S * np.sqrt(1 + 1/n)
    pred_high = mean + t_crit * S * np.sqrt(1 + 1/n)
    st.write(f"{conf_choice} 预测区间 = **({pred_low:.4f}, {pred_high:.4f})**")

    # 显示公式
    st.markdown(f"公式：$$t = \\frac{{X_{{pred}} - \\bar{{X}}}}{{S\\sqrt{{1 + 1/n}}}}$$")
    st.markdown(f"具体计算：$$t = ({user_prediction} - {mean:.4f}) / ({S:.4f} * \\sqrt{{1 + 1/{n}}})$$")

    if pred_low <= user_prediction <= pred_high:
        st.success(f"✅ 预测值落在 acceptance region → 预测值合理")
    else:
        st.error(f"❌ 预测值落在 rejection region → 预测值不合理")

    # 绘图 PDF 功能1
    st.subheader("📈 预测值 PDF")
    x_min = min(data) - 10
    x_max = max(data) + 10
    x = np.linspace(x_min, x_max, 500)
    y = stats.t.pdf((x - mean)/(S/np.sqrt(n)), df) / (S/np.sqrt(n))

    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(x, y, label=f"t-distribution PDF (df={df})")
    accept_low = mean - t_crit*S/np.sqrt(n)
    accept_high = mean + t_crit*S/np.sqrt(n)
    ax.fill_between(x, 0, y, where=(x >= accept_low) & (x <= accept_high), color="lightgreen", alpha=0.3, label="acceptance region")
    ax.fill_between(x, 0, y, where=(x < accept_low) | (x > accept_high), color="lightcoral", alpha=0.3, label="rejection region")
    y_pred = stats.t.pdf((user_prediction - mean)/(S/np.sqrt(n)), df) / (S/np.sqrt(n))
    ax.plot(user_prediction, y_pred, 'ro', label="Your prediction")
    ax.set_xlabel("t")
    ax.set_ylabel("Probability Density")
    ax.set_title("Prediction PDF")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

    # ---------------------------
    # 功能 2：样本均值假设检验
    st.subheader("💊 功能 2：样本均值假设检验")
    mu0 = st.number_input("请输入总体均值 μ₀:", value=0.0)

    t_stat = (mean - mu0) / (S / np.sqrt(n))
    t_crit_two = stats.t.ppf(1 - alpha/2, df)
    t_crit_one = stats.t.ppf(1 - alpha, df)

    # 双尾检验
    p_two = 2 * (1 - stats.t.cdf(abs(t_stat), df))
    # 单尾自动选择
    if mean > mu0:
        p_one = 1 - stats.t.cdf(t_stat, df)
        tail_text = "right-tailed (μ > μ₀)"
        t_crit_one_val = t_crit_one
    else:
        p_one = stats.t.cdf(t_stat, df)
        tail_text = "left-tailed (μ < μ₀)"
        t_crit_one_val = -t_crit_one

    # 显示公式和计算
    st.markdown(f"公式（t 统计量）：$$t = \\frac{{\\bar{{X}} - μ₀}}{{S / \\sqrt{{n}}}}$$")
    st.markdown(f"具体计算：$$t = ({mean:.4f} - {mu0}) / ({S:.4f} / \\sqrt{{{n}}}) = {t_stat:.4f}$$")
    st.markdown(f"临界值（双尾） ±t_crit = ±{t_crit_two:.4f}，单尾 t_crit = {t_crit_one_val:.4f}")

    # 双尾结果
    st.write(f"双尾检验 p 值 = {p_two:.4f}")
    if abs(t_stat) <= t_crit_two:
        st.info(f"✅ 双尾：样本均值落在 acceptance region → 没有足够证据证明 μ ≠ μ₀")
    else:
        st.error(f"❌ 双尾：样本均值落在 rejection region → 样本均值显著不同于 μ₀")

    # 单尾结果
    st.write(f"单尾 ({tail_text}) p 值 = {p_one:.4f}")
    if (mean > mu0 and t_stat > t_crit_one) or (mean < mu0 and t_stat < t_crit_one_val):
        st.error(f"❌ 单尾：样本均值落在 rejection region → 样本均值显著 {tail_text}")
    else:
        st.info(f"✅ 单尾：样本均值落在 acceptance region → 没有足够证据拒绝 H0")

    # 绘图 PDF 功能2（以 μ0 为中心）
    st.subheader("📈 样本均值假设检验 PDF")
    x_min2 = mu0 - 4*S/np.sqrt(n)
    x_max2 = mu0 + 4*S/np.sqrt(n)
    x2 = np.linspace(x_min2, x_max2, 500)
    y2 = stats.t.pdf((x2 - mu0)/(S/np.sqrt(n)), df) / (S/np.sqrt(n))

    fig2, ax2 = plt.subplots(figsize=(8,5))
    ax2.plot(x2, y2, label=f"t-distribution PDF (df={df})")

    # 拒绝域/接受域（双尾为参考）
    accept_low2 = mu0 - t_crit_two*S/np.sqrt(n)
    accept_high2 = mu0 + t_crit_two*S/np.sqrt(n)
    ax2.fill_between(x2, 0, y2, where=(x2 >= accept_low2) & (x2 <= accept_high2), color="lightgreen", alpha=0.3, label="acceptance region")
    ax2.fill_between(x2, 0, y2, where=(x2 < accept_low2) | (x2 > accept_high2), color="lightcoral", alpha=0.3, label="rejection region")

   # 样本均值红线
y_mean2 = stats.t.pdf((mean - mu0)/(S/np.sqrt(n)), df) / (S/np.sqrt(n))
ax2.plot([mean, mean], [0, y_mean2], color='purple', linestyle='--', label="Sample mean")
ax2.text(mean, y_mean2*1.05, f"{mean:.2f}", color='purple', ha='center')  # 添加标注

    ax2.set_xlabel("t")
    ax2.set_ylabel("Probability Density")
    ax2.set_title("Sample Mean PDF")
    ax2.grid(True)
    ax2.legend()
    st.pyplot(fig2)


if __name__ == "__main__":
    main()
