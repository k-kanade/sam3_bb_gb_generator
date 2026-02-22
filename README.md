# はじめに

こちらは以下のプラグインの問題点を修復し、処理の最適化をしたものです  
元のプラグイン:https://github.com/clean262/sam3_bb_gb_generator  

## 主な変更点  
- プラグイン、フォルダ名を変更し、元のプラグインが導入されていても機能します  
- 動画の読み込み時のメモリ圧迫の改善  
- 「Propagate across segment」ですべてのフレームを処理するように変更  
- 「透過」がどうにもならなかったので「GB」として出力  
- 後色々と変更(下記見てね)  

こちらではPythonがインストールが必須になります  
ローカル環境にライブラリをインストールする前提で作成してます  
このkaizo版の問題点はこのリポジトリに連絡してください  

# 必須なもの  
- Python3.13.xx  
- CUDA  
- Hugging FaceアカウントとRead Token
- つよいPC　

# 導入方法  
1.releasesから`kanade-SAM.au2pkg.zip`をダウンロードし、AviUtl2にD&Dしてください  
2.ライブラリ  
・`pip install accelerate gradio av imageio-ffmpeg imageio[pyav] kernels opencv-python spaces transformers triton-windows huggingface-hub`  
3.CPUの場合 `pip install torch`  
  GPUの場合 https://pytorch.org/get-started/locally/ を見て自分のCUDAのバージョンに合わせてしてください  
4.Hugging Faceからfacebook/sam3のアクセス権を入手し、Read権限のあるTokenを取得してください  
5.`run hf auth login`を実行し、Tokenを設定してください  

# 修正と最適化一覧  

## 名前変更  
- ビルド出力名を `sam3` / `sam3mask` から `sam3-kaizo` / `sam3mask-kaizo` に変更  
- Lua 側のモジュール参照を `sam3mask-kaizo` に変更  
- プラグイン表示名、ウィンドウクラス名、登録名を `SAM3-kaizo` に統一  
- 旧版 `SAM3` と同時導入したときに衝突しないように名前を分離  
- `CMakeLists.txt` 側も `OUTPUT_NAME` を kaizo名へ統一し、配布物の識別を明確化  
- CMake に `Windowsのみ` `MSVCのみ` の明示チェックを追加し、ビルド時の誤環境実行を防止  

## C++(`cpp/sam3_aux2.cpp`)  
- 既存オブジェクトの扱いを「自動非表示」から「その場で置換」へ変更  
- タイムライン長の適用時に 1フレーム過長にならないように変更  
- Aliasパッチ処理を強化(動画パス置換失敗時の明示エラー化、日本語/英語キー対応強化)  
- `SAM3mask-kaizo` 効果へ `path/mask_src_start/mask_src_end` を冗長キーで確実に書き込み  
- `result.json` が遅延/欠落した場合の待機猶予とフォールバック生成を追加  
- Python 実行ファイル探索を強化(`SAM3_PYTHON_EXE`、同梱 Python、`PATH`)  
- `request.json` の `timeline` に `fps` を追加し、長さ同期を改善  
- 置換処理は「元オブジェクトを一時退避 -> 新規作成物を元位置へ移動 -> 元を削除 -> 名前復元」の順で実行  
- `focus_alias_utf8` からの動画差し替え時、パス置換に失敗したら即座に停止して原因を見える化  
- `json` 抽出処理をトップレベルキー探索 + 文字列リテラル解析に変更し、誤検出を減少  
- Python終了直後に `result.json` が未生成でも一定時間待機し、ログ末尾付きの失敗 `result.json` を自動生成  
- `result.json` が壊れている/不完全な場合も待機後に失敗JSONへ置換し、ランチャー側ハングを防止  
- `result success` なのに出力ファイル未準備のケースでタイムアウト監視を追加  

## Lua(`lua/SAM3mask.anm2`)  
- 効果名の揺れに対応(`動画ファイル` / `Video File` / `映像再生` / `Movie Playback`)  
- 時刻取得を `obj.getvalue("time")` 優先にして同期を安定化  
- `mask_src_start/end` 未取得時は再生位置情報へフォールバック  
- マスク適用時刻のクランプ処理を強化  
- `totaltime` が取れない環境でも例外にせず安全側の時刻計算へフォールバック  
- 変数読み取り失敗時に再生位置ベースで同期を取るため、プロジェクト差分でのズレが出にくい  
- 反転/オフセット/範囲制限の順序を固定し、透過適用結果の再現性を向上  
- `sam3mask-kaizo.mod2` と `SAM3mask-kaizo.anm2` の組み合わせを前提にモジュール名を統一  

## Python(`python/sam3_gradio_job.py`)  
- `torch.compile` 失敗時の自動無効化と再試行を追加  
- `status.json` / `result.json` 書き込みのリトライとフェイルセーフを追加  
- セグメント動画の ffmpeg 前処理を追加(フレーム不足対策つき)  
- フレーム保持を `OnDemandVideoFrames` に変更し、メモリ使用を抑制  
- 推論セッションをチャンク単位で遅延ロード(メモリ不足時に縮小リトライ)  
- マスクを packed 形式で保持してメモリ負荷を軽減  
- propagate を「フルセッション優先 + 失敗時チャンクフォールバック」に拡張  
- 出力ファイル名を `元動画名_GB/BB.mp4` へ整理し、中間生成物の掃除を追加  
- UI 待機タイムアウト、最終例外時の `result.json` 強制出力を追加  
- 安定化のため、Finish 時は透明指定でも GB/BB 出力へフォールバック  
- `request.json` のオプション読込を拡張し、`session_chunk_frames` やメモリ予算比率を反映可能化  
- CUDA実行時は `triton` 検出と TF32設定を行い、非対応環境では安全にCPU/非compileへ降格  
- PyAV(`av`)の事前チェックを追加し、SAM3動画セッション生成失敗を起動時に検出  
- `OnDemandVideoFrames` にフレームキャッシュ上限を持たせ、長尺でもメモリ常駐を抑制  
- セグメント前処理を `nvenc -> x264 -> accurate seek + pad` の順で再試行し、不足フレームに対応  
- 低メモリ時はチャンクサイズを段階的に縮小してセッション再初期化  
- propagate時に seed box/mask を注入して、チャンク境界またぎの追跡安定性を改善  
- マスク保存を `packbits` 化して `masks_by_frame` の保持サイズを削減  
- Finish時に中間生成物を掃除し、最終成果物だけ残す運用へ変更  
- UI放置時のタイムアウト失敗処理と、未処理例外時の最終 `result.json` 出力を実装  

## 依存関係・配布周り  
- `python/pyproject.toml` を更新  
  - `readme = "../README.md"` へ変更  
  - `av` `triton` を明示依存へ追加  
  - `torch` を固定版から下限指定へ変更  
  - `transformers` を Git rev 指定へ変更  
  - 旧 `uv` index/source 設定を削除  
- `python/requirements-cuda.txt` を追加し、pip導入で必要依存をまとめて入れられるように変更  
- `DL用/kanade-SAM.au2pkg` に kaizo版の配布用ファイル一式を追加  
- `DL用/kanade-SAM.au2pkg/Plugin/SAM3-kaizo/Python/sam3_gradio_job_kaizo.py` を同梱し、配布側と開発側のスクリプトを一致させた  

## 変更ファイル一覧  
### 追加  
- `DL用/kanade-SAM.au2pkg/Plugin/SAM3-kaizo/Python/sam3_gradio_job_kaizo.py`  
- `DL用/kanade-SAM.au2pkg/Plugin/SAM3-kaizo/sam3-kaizo.aux2`  
- `DL用/kanade-SAM.au2pkg/Script/SAM3-kaizo/SAM3color-kaizo.anm2`  
- `DL用/kanade-SAM.au2pkg/Script/SAM3-kaizo/SAM3mask-kaizo.anm2`  
- `DL用/kanade-SAM.au2pkg/Script/SAM3-kaizo/sam3mask-kaizo.mod2`  
- `python/requirements-cuda.txt`  

### 削除  
- `.github/workflows/release.yaml`  
- `.gitignore`  
- `installer/install.cmd`  
- `installer/install.ps1`  
- `python/uv.lock`  
- `uv_license/LICENSE-APACHE`  
- `uv_license/LICENSE-MIT`  

### 変更  
- `CMakeLists.txt`  
- `cpp/sam3_aux2.cpp`  
- `cpp/sam3mask.cpp`  
- `lua/SAM3mask.anm2`  
- `python/pyproject.toml`  
- `python/sam3_gradio_job.py`  
- `README.md`  


## ライセンス  
**MIT ライセンス**  
