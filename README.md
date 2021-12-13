# My Kaggle Competitions

This repository contains the recording of my Kaggle competitions. The folder `goatwu` is my templates, and the folder `competitions` holds my notebooks.

## Classify Leaves

### Result

The web address: [Classify Leaves](https://www.kaggle.com/c/classify-leaves/)

My best result is $0.98795$ in private LB, and $0.98500$ in public LB. And it was a late submission. Here are all my results:

<table>
<thead>
  <tr>
    <th colspan="2"></th>
    <th>Public LB</th>
    <th>Private LB</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td rowspan="2">resnext-50</td>
    <td>one of the 5 folds</td>
    <td>0.98113</td>
    <td>0.98318</td>
  </tr>
  <tr>
    <td>ensemble of 5 folds</td>
    <td>0.98500</td>
    <td>0.98795</td>
  </tr>
  <tr>
    <td rowspan="2">densenet-161</td>
    <td>one of the 5 folds</td>
    <td>0.97500</td>
    <td>0.97931</td>
  </tr>
  <tr>
    <td>ensemble of 5 folds</td>
    <td>0.98181</td>
    <td>0.98636</td>
  </tr>
  <tr>
    <td rowspan="2">efficientnet-b2</td>
    <td>one of the 5 folds</td>
    <td>0.97204</td>
    <td>0.97727</td>
  </tr>
  <tr>
    <td>ensemble of 5 folds</td>
    <td>0.97795</td>
    <td>0.98340</td>
  </tr>
  <tr>
    <td colspan="2">ensemble of the 3 models</td>
    <td>0.98227</td>
    <td>0.98681</td>
  </tr>
</tbody>
</table>

### Tricks

In this competition, data augmentation is not really efficient. I think the reason is the training set is really similar to the test set, so overfitting the training set could get a good grade.

- Using `cutmix` can obviously increase the precision.

- Using `TTA` (Image Test Time Augmentation).

- Adjust the learning rate using cosine learning rate: 

  `scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)`

- Using `torch.optim.AdamW` instead of `Adam`

### Coding Details

- When using `kfold` algorithm, you should reset your model to the initial state. Here we should record not only the net’s `state_dict`, but also the optimizer’s and the scheduler’s.
- When using `Tensorboard` for visualization, you can merge the train accuracy and valid accuracy in a single graph.
