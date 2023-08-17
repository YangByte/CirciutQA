# CircuitQA 

## Environment
python=3.6、allennlp==0.9.0

### Preparing

git clone https://github.com/YangByte/CircuitQA.git

cd CircuitQA

pip install -r requirements.txt

Download the <a href="https://drive.google.com/file/d/1sHGwqoDlL0__993kp4YEkWdJVOVWrKhc/view?usp=sharing">data.zip</a>, move it to CircuitQA path, and unzip it.


### Training
    
    allennlp train config/Circuit_Aux.json --include-package Circuit_Aux -s save/test

### Evaluation
    
    allennlp evaluate save/test  data/CircuitQA/test.pk --include-package Circuit_Aux_test --cuda-device 0




