# distributed training pvp-cartpole in KAGAYAKI  
## 誰向け  
kagayakiの開発環境を模索中の人  
kagayakiの使い方がわからない人  

kagayakiを使って分散強化学習をやってみたい方。    

## 概要  
スパコンKAGAYAKIを使って、cartpoleを元に作成した対戦ゲーム環境、pvp-cartpoleをselfplayで学習します。   
工夫:  
- [wandb](https://www.wandb.jp/)に学習経過を送信し、リアルタイムで経過をモニタリング
- 一つのsifからバッチ実行とjupyter notebook環境両方を使えるように。    
  

## 実行準備  
### 1. コードの修正  
(dummy)とある以下のファイルは修正が必要です。  
- run(dummpy).sh
- config(dummpy).ini  
コード内で指定された修正を行ってください。  
  
### 2. ビルド済みコンテナの取得  
repositoryのルートディレクトリから
```
./make_new_sif_env.sh
```
これで./singularity/python.sifが作成される。  
  
## 実行  
### 1. notebook実行  
以下を実行してGPUインスタンスに入る。  
```
qsub -q GPU-1 -l select=1:ngpus=1 -I
```
インスタンス内でrepositoryのルートディレクトリに移動し、
```
./start_notebook_server.sh
```  
しばらくすると、http://spc....みたいなリンクが出てくるので、コピペしてブラウザで開けばnotebook環境につなげる。  
  
### 2. batch jobの実行
repositoryのルートディレクトリから  
```
./start_batch_job.sh
```
  
### 結果確認    
確認したいメトリクスに関してこんな感じで確認できます。  
[wandbダッシュボード](https://api.wandb.ai/links/data_science_nichika/vdj1aubw)
  

## カスタム環境に関して  
cartpoleをもとにpvpの対戦環境を作りました。  
- stabilizerとdisturberの二つの役割で対戦を実施する。  
- stabilizerとdisturberが交互に行動する。  
- 勝敗判断の基準としてwin_rate_thresholdを変数として用意。  
    - win_rate_thresholdよりも多い回数でゲームが終了した場合、disturberの勝利。
        - disturberに報酬+1 / stabilizerに報酬-1  
    - win_rate_thresholdよりも少ない回数でゲームが終了した場合、stabilizerの勝利  
        - - disturberに報酬-1 / stabilizerに報酬+1
  
