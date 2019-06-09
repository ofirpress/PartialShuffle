# [Partially Shuffling the Training Data to Improve Language Models](https://arxiv.org/abs/1903.04167)

This repository contains the code for the [Partial Shuffle method](https://arxiv.org/abs/1903.04167), and a modified version of the [DOC language model](https://github.com/nttcslab-nlp/doc_lm) that utilizes this method.

If you'd like to run the DOC + Partial Shuffle models, use the same commands as in the original DOC model, presented [here](https://github.com/nttcslab-nlp/doc_lm).

The code for the Partial Shuffle method itself is in `partial_shuffle.py`. If you'd like to use this method in your own language model, simply import `partial_shuffle.py`, and call it before each epoch, as in line 196 in `main.py`. No other modifications are required.


## Reference
If you found this code useful, please cite the following paper:

```
@article{press2019partially,
  title={Partially Shuffling the Training Data to Improve Language Models},
  author={Press, Ofir},
  journal={arXiv preprint arXiv:1903.04167},
  year={2019}
}
```
