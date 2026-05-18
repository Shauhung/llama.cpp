# Llama.cpp 核心架構與初始化流程筆記

## 🛠️ 核心角色與職責 (Role Definitions)
* `cli_context` (前端指揮官)：負責與使用者互動。管理終端機輸入輸出、Loading 動畫、DeepSeek 思考字體顏色切換等 UI 邏輯。
* `server_context` (後端萬能引擎)：核心控制中樞。負責管理任務隊列、模型推理計算。不管是 `llama-cli` 還是 `llama-server` (網頁版) 都共用此核心。
* `common_init_result` (底層大管家)：專職處理載入模型、分配硬體記憶體等沉重且可能失敗的 C 語言風格髒活。
* `pimpl->model` (安全保險箱)：獨佔型智慧指標 (`std::unique_ptr`)。利用 RAII 機制自動管理模型記憶體生命週期，防止記憶體洩漏 (Memory Leak)。

---

## 🗺️ 啟動與載入模型流程 (Initialization Phase)
*此階段在程式開機時「只執行一次」，採兩階段初始化（先建空殼，再載入模型），避免建構子失敗。*

1.  **解析參數 (`main`)**
    * 終端機輸入指令（如 `-m model.gguf -c 4096`）被 `common_params_parse` 讀取並打包成 `common_params params` 結構體。
2.  **建立前端空殼 (`cli_context cli(params)`)**
    * 建立 UI 殼層。其建構子僅做輕量級變數賦值（例如強制設定 `stream = true`），此時內部 `server_context` 仍為未載入模型的空引擎。
3.  **手動下令載入 (`cli.ctx_server.load_model(params)`)**
    * 主程式顯式呼叫此函式，正式啟動核心初始化鏈：
        $$\text{load\_model()} \longrightarrow \text{common\_init\_from\_params()} \longrightarrow \text{new common\_init\_result()}$$
4.  **硬體動態適應 (`common_params_fit_impl`)**
    * 在模型載入前，探測系統/顯示卡（如 Mac M5 Pro）的剩餘記憶體。
    * 若發現使用者設定的 Context 長度過大，會自動在幕後「等比調小參數」，做為防禦機制避免 OOM (Out of Memory) 崩潰。
5.  **模型載入與託管 (`llama_model_load_from_file`)**
    * 真正由硬碟讀取巨大的 GGUF 權重檔案進入記憶體。
    * 成功後立刻呼叫 `pimpl->model.reset(model)`，將原生指標的所有權移交給智慧指標，由其生命週期全權負責後續的自動釋放。
6.  **獲取模型字典 (`llama_model_get_vocab`)**
    * 從載入成功的模型中提取專屬的字彙表（Vocab），供後續將人類文字與 Token ID 互相轉換（Tokenization / Detokenization）。
7.  **綁定中斷回呼 (`llama_context::set_abort_callback`)**
    * 遍歷當前啟用的所有硬體後端（CPU / Metal / CUDA 等），動態尋找並設定 `ggml_backend_set_abort_callback`。
    * **作用**：當使用者在 UI 按下「停止生成」時，底層矩陣運算能即時收到訊號並立刻煞車中斷。

---

## 🔄 一問一答互動循環 (Interactive Loop Phase)
*模型載入完畢後，程式進入 `while(true)` 循環。模型常駐記憶體，不重複載入。*

1.  **等待輸入**：程式停在 `console::readline(">>> ")` 阻塞等待。
2.  **觸發推論**：使用者輸入新問題後，呼叫 `cli.generate_completion()`。
3.  **打包任務 (Task Post)**：
    * 將「歷史對話紀錄 (messages) + 最新問題」打包成 `SERVER_TASK_TYPE_COMPLETION` 類型的 `server_task`。
    * 針對 DeepSeek 等模型，在此處查 `vocab` 字典，寫入思考預算參數（`reasoning_budget_tokens`）。
    * 將任務丟入 `ctx_server` 的隊列中排隊運算。
4.  **串流輸出與狀態機切換 (UI Color State Machine)**：
    * 開啟轉圈動畫 (`spinner::start()`)，直到接收到第一個 Token 後關閉。
    * 進入 `while(result)` 迴圈，透過智慧指標 `result.get()` 異步或同步讀取回傳的增量數據：
        * **收到思考數據 (`reasoning_content_delta`)**：UI 切換至 `DISPLAY_TYPE_REASONING` 顯示模式（如字體變灰/斜體），印出思考過程。
        * **收到正式答案 (`content_delta`)**：UI 切換回 `DISPLAY_TYPE_RESET` 正常模式，接續印出答案文字。
5.  **還原狀態**：模型完全輸出完畢（收到 `cmpl_final`），跳出迴圈，釋放當前 task 資源，主導權還給 `main()` 繼續等待下一次輸入。

---

## 💡 C++ 原始碼閱讀關鍵線索 (Key C++ Idioms)
* **Pimpl (Pointer to Implementation) 模式**：原始碼中常見的 `pimpl(new impl{})`。將私有成員與底層實作細節隱藏在 `impl` 結構體中，確保標頭檔乾淨並加速編譯。
* **智慧指標成員函式**：
    * `.get()`：**唯讀查看**。僅借出內部的傳統原生指標（如傳給舊型 C 語言 API 參數），智慧指標並未放棄管理權。
    * `.reset(ptr)`：**所有權轉換/釋放**。當場 delete 釋放原本管理的舊資源，並轉為託管傳入的新原生指標 `ptr`。