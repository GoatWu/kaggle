# My Kaggle Competitions

This repository contains the recording of my Kaggle competitions. The folder `goatwu` is my templates, and the folder `competitions` holds my notebooks.

## Classify Leaves

### Result

The web address: [Classify Leaves](https://www.kaggle.com/c/classify-leaves/)

My best result is $0.98795$ in private LB, and $0.98500$ in public LB. And it was a late submission. Here are all my results:

<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;}
.tg td{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  overflow:hidden;padding:10px 5px;word-break:normal;}
.tg th{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}
.tg .tg-9wq8{border-color:inherit;text-align:center;vertical-align:middle}
.tg .tg-c3ow{border-color:inherit;text-align:center;vertical-align:top}
.tg .tg-7btt{border-color:inherit;font-weight:bold;text-align:center;vertical-align:top}
</style>
<table class="tg">
<thead>
  <tr>
    <th class="tg-c3ow" colspan="2"></th>
    <th class="tg-c3ow">Public LB</th>
    <th class="tg-c3ow">Private LB</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-9wq8" rowspan="2">resnext-50</td>
    <td class="tg-c3ow">one of the 5 folds</td>
    <td class="tg-c3ow">0.98113</td>
    <td class="tg-c3ow">0.98318</td>
  </tr>
  <tr>
    <td class="tg-c3ow">ensemble of 5 folds</td>
    <td class="tg-7btt">0.98500</td>
    <td class="tg-7btt">0.98795</td>
  </tr>
  <tr>
    <td class="tg-9wq8" rowspan="2">densenet-161</td>
    <td class="tg-c3ow">one of the 5 folds</td>
    <td class="tg-c3ow">0.97500</td>
    <td class="tg-c3ow">0.97931</td>
  </tr>
  <tr>
    <td class="tg-c3ow">ensemble of 5 folds</td>
    <td class="tg-c3ow">0.98181</td>
    <td class="tg-c3ow">0.98636</td>
  </tr>
  <tr>
    <td class="tg-9wq8" rowspan="2">efficientnet-b2</td>
    <td class="tg-c3ow">one of the 5 folds</td>
    <td class="tg-c3ow">0.97204</td>
    <td class="tg-c3ow">0.97727</td>
  </tr>
  <tr>
    <td class="tg-c3ow">ensemble of 5 folds</td>
    <td class="tg-c3ow">0.97795</td>
    <td class="tg-c3ow">0.98340</td>
  </tr>
  <tr>
    <td class="tg-c3ow" colspan="2">ensemble of the 3 models</td>
    <td class="tg-c3ow">0.98227</td>
    <td class="tg-c3ow">0.98681</td>
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
