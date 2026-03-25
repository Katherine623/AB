# Multi-Armed Bandits (MAB) Strategies Comparison

這份 repository 整理了我們在「多臂吃角子老虎機 (MAB)」隨堂作業 (In-Class Activity) 中的演算法實作、實驗模擬以及深入討論與分析報告。情境設定為 $10,000 的總資金預算分配在 3 種不同期望報償的賭博機台上，以此了解 6 種「探索優先還是開發優先 (Exploration-Exploitation)」策略背後的取捨與成效差異。

---

## 📝 檔案清單結構 (Repository Contents)

- **`simulate_bandits.py`**: 作業的重頭戲（Python 數學模擬程式碼）。包含了您所有 6 種指定演算法（A/B Testing, Optimistic Initial Values, ε-Greedy, Softmax, UCB, Thompson Sampling）的程式碼邏輯實作。執行後會自動產生這 3 張精美的圖表文件。
- **`MAB_Report.md`**: 依照作業要求格式生成的最終完整分析報告。包含了班級統籌比較表 (Class Comparison Table) 與討論題目問答 (Discussion Questions)。不僅說明了每個演算法在 Exploring 階段與 Exploitation 階段的預算法分配（2000步 / 8000步），還詳細寫出了計算預期回報與悔恨值 (Regret) 的原理跟優缺點。
- **產出的圖表 (.png)**:
  - `1_independent_running_avg.png`: 第一階段 3 機台不連動的流動平均圖。
  - `2_explore_exploit_diagram.png`: Explore-then-Exploit 分析圖（含標準差帶區），並附有探索完後的長條圖分佈（預估 vs 真實）。
  - `3_bandit_strategies_regret.png`: 上列全體這 6 種策略演算法歷經一萬步後的長期「累積悔恨值」成長與收斂曲線比較。
- **`對話紀錄.txt`**: 我們產生這份作業時，詳細一字一句的對話溝通史。

---

## 🚀 如何執行程式 (How to Run)

請先確保您的環境中已安裝 `numpy` 以及繪圖所需的 `matplotlib` 套件。
您可以透過以下環境指令確認並執行這次的模擬結果：

```bash
pip install numpy matplotlib
python simulate_bandits.py
```

執行這段程式碼後，您將會在當前目錄下看到最新的 `.png` 比較分析圖表被產出。
