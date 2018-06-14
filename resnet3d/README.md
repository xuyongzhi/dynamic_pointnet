# Key items to be developed
- data augment
- set as channel first
- check weight decay
- check use bias or not
- add dropout
- Use kernel>1 in shortcut may somewhat impede the identity forward, try optimize later


# Training notes
- batch norm decay is small is good: ???
- batch_denom=256 better than 128
- batch_size=48 seems lead to Nan
