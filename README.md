# distributed training pvp-cartpole in KAGAYAKI  
## 誰向け  
kagayakiでとにかく早くcafeobjを使いたい人      

## 概要  
kagayakiでcafeobjを実行    
  

## 実行準備  
### 1. ビルド済みコンテナの取得  
repositoryのルートディレクトリから
```
./make_new_sif_env.sh
```
これで./singularity/cafeobj.sifが作成される。  
  
(これは自分がローカルでビルドしたコンテナです。)  
(kagayaki上ではroot権限がなくビルドできないので...)
  
## 実行  
### 1. 作業ノードを起動し、singularityコンテナに入る    
以下を実行してインタラクティブインスタンスに入る。  
```
qsub -l select=1:ncpus=4 -I
```
インスタンス内でrepositoryのルートディレクトリに移動し、
```
./start_container.sh
```  

singularity内でcafeobjを起動  
```
cafeojb
```  

### .cafeファイルの実行  
例としてsrc/fibonacci.cafeを実行  
```
cafeobj -batch src/sample_fnc.cafe > cafe.out　
```  
コードは[公式チュートリアル](https://cafeobj.org/2015/02/tutorial-first-steps-in-cafeobj/)から拝借  
  
結果はcafe.outファイルに書き出されます。  

結果があっているのかは不明..

