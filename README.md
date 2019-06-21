# GraphFlow modification for CHAMPS challenge

![](https://travis-ci.com/qbeer/GraphFlow.svg?branch=kaggle)

* Travis-CI
* Molecular structure

To do learning with a second order model do:

```bash
git clone https://github.com/qbeer/GraphFlow
cd GraphFlow
cd tests/ && g++ -std=c++11 -pthread test_SMP_theta_physics.cpp
./a.out
```

Only the first few parameters are shown during training.