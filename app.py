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
    conf_choice = st.radio("选择显著性水平:", list(alpha_map.keys()), index=1)
    alpha = alpha_map[conf_choice]

    # ---------------------------
    # 显示样本统计量及公式
    st.subheader("样本统计量")
    st.write(f"样本量 n = {n}")

    mean_formula = f"\\bar{{X}} = ( {' + '.join([str(x) for x in data])} ) / {n} = {mean:.4f}"
    st.markdown(f"样本均值 = **{mean:.4f}**  \n公式：$$\\bar{{X}} = \\frac{{1}}{{n}} \\sum_{{i=1}}^{{n}} X_i$$  \n具体计算：$$ {mean_formula} $$")

    deviations = [f"({x}-{mean:.2f})^2" for x in data]
    S2_formula = f"S^2 = ( {' + '.join(deviations) } ) / ( {n}-1 ) = {S2:.4f}"
    st.markdown(f"样本方差 = **{S2:.4f}**  \n公式：$$S^2 = \\frac{{1}}{{n-1}} \\sum_{{i=1}}^{{n}} (X_i - \\bar{{X}})^2$$  \n具体计算：$$ {S2_formula} $$")

    st.markdown(f"样本标准差 = **{S:.4f}**  \n公式：$$S = \\sqrt{{S^2}}$$  \n具体计算：$$S = \\sqrt{{{S2:.4f}}} = {S:.4f}$$")

    # ---------------------------
    # 功能 1：预测值检验
    st.subheader("预测值检验")

    user_prediction = st.number_input("请输入你的预测值:", value=95.0)
    t_crit = stats.t.ppf(1 - alpha/2, df)
    pred_low = mean - t_crit * S * np.sqrt(1 + 1/n)
    pred_high = mean + t_crit * S * np.sqrt(1 + 1/n)
    st.write(f"{conf_choice} 新观测值预测区间 = **({pred_low:.4f}, {pred_high:.4f})**")

    if pred_low <= user_prediction <= pred_high:
        st.success(f"✅ 预测值 {user_prediction} 落在 acceptance region，接受 H0")
    else:
        st.error(f"❌ 预测值 {user_prediction} 落在 rejection region，拒绝 H0")

    # 绘图
    st.subheader("PDF")
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
    st.subheader("样本均值假设检验（单尾/双尾）")
    mu0 = st.number_input("第一步：你认为总体均值 μ₀ 是多少？", value=0.0)

    tail_choice = st.radio("第二步：选择检验类型", ["two-tailed", "left-tailed", "right-tailed"])

    t_stat = (mean - mu0) / (S / np.sqrt(n))

    # 计算 p 值
    if tail_choice == "two-tailed":
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))
    elif tail_choice == "left-tailed":
        p_value = stats.t.cdf(t_stat, df)
    else:  # right-tailed
        p_value = 1 - stats.t.cdf(t_stat, df)

    st.write(f"t 统计量 = {t_stat:.4f}")
    st.write(f"p 值 = {p_value:.4f}")

    if p_value < alpha:
        st.success(f"❌ p < {alpha}，拒绝 H0 → 样本均值显著不同于 μ₀")
    else:
        st.info(f"✅ p ≥ {alpha}，不拒绝 H0 → 样本均值与 μ₀ 无显著差异")

    # 绘图 t_stat
    st.subheader("PDF")
    fig2, ax2 = plt.subplots(figsize=(8,5))
    ax2.plot(x, y, label=f"t-distribution PDF (df={df})")

    # 拒绝域填充
    if tail_choice == "two-tailed":
        ax2.fill_between(x, 0, y, where=(x < accept_low) | (x > accept_high), color="lightcoral", alpha=0.3, label="rejection region")
        ax2.fill_between(x, 0, y, where=(x >= accept_low) & (x <= accept_high), color="lightgreen", alpha=0.3, label="acceptance region")
    elif tail_choice == "left-tailed":
        ax2.fill_between(x, 0, y, where=(x <= mean + t_crit*S/np.sqrt(n)), color="lightgreen", alpha=0.3, label="acceptance region")
        ax2.fill_between(x, 0, y, where=(x < x_min) | (x > mean + t_crit*S/np.sqrt(n*_
