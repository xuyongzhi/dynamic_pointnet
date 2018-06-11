# Key items to be developed
- data augment
- set as channel first
- check optimizer
- check batch decay
- check weight decay
- check use bias or not
- add dropout
- enable use_xyz
- Use kernel>1 in shortcut may somewhat impede the identity forward, try optimize later
- enable normal
- log net configs


# Training notes
- batch_denom=256 better than 128
- batch_size=48 seems lead to Nan
