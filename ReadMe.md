#ReadMe
## Code
The code here is the version that get the lastest result.

For the SRL task, you can run:

../code/MultiTask -T ../data/joint.train -d ../data/joint.dev -W -S -P -C -r -a -t

-T: train dataset
-d: dev dataset
-W: use Word
-S: use Char
-P: use PoS
-C: combine Word + Char + Pos with parameters (P1 x w + P2 x c + P3 x Pos)
-r: use dropout
-a: SRL Task
-t: train

For the Parser task, you can run:
../code/MultiTask -T ../data/joint.train -d ../data/joint.dev -W -S -P -C -r -b -t

-b: Parser Task

For the Joint task, you can run:

../code/MultiTask -T ../data/joint.train -d ../data/joint.dev -W -S -P -C -r -j -t

-j: joint task

## Model
Run 
```
bash getModel.sh
```
to get the model first.
There are the baseline model and joint model parameters.
You can use:

```
../code/MultiTask -T ../data/joint.train -d ../data/joint.dev -p ../data/joint.test -e -W -S -P -C -r -a -m ../model/base-srl.params
../code/MultiTask -T ../data/joint.train -d ../data/joint.dev -p ../data/joint.test -e -W -S -P -C -r -b -m ../model/base-parser.params
../code/MultiTask -T ../data/joint.train -d ../data/joint.dev -p ../data/joint.test -e -W -S -P -C -r -a -m ../model/joint-model.params
```
to test the result.

-p: test dataset
-e: evaluation
-m: model path

## Data
Run 
```
bash getData.sh
```
first to get the data.
The joint.dev joint.train joint.test is the data for the above tasks.
Chinese Semantic Treebank.txt is the original data.
chieseTranfer.py is the code to transfer the the original data to the specific version.

## Exec
There are some scripts for run the tasks.

For comparison, we set the random seed as 1.


