# Key items to be developed
- check batch size and batch norm decay
- data augment: random crops, more complex rotation
- set as channel first
- check weight decay
- check use bias or not
- add dropout
- Use kernel>1 in shortcut may somewhat impede the identity forward, try optimize later
- Try low resolution firstly. Andrew has achieved 0.95 with 32*32*32 resolution.


# Training notes
- batch norm decay is small is good: ???
- batch_denom=256 better than 128
- batch_size=48 seems lead to Nan
