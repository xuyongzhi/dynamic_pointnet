# May 2018 xyz



ETH_Meta = {}

ETH_Meta['label2class'] = {0: 'unlabeled points', 1: 'man-made terrain', 2: 'natural terrain',\
                    3: 'high vegetation', 4: 'low vegetation', 5: 'buildings', \
                    6: 'hard scape', 7: 'scanning artefacts', 8: 'cars'}
ETH_Meta['unlabelled_categories'] = [0]
ETH_Meta['easy_categories'] = []
ETH_Meta['label2color'] = \
                {0:	[0,0,0],1:	[0,0,255],2:	[0,255,255],3: [255,255,0],4: [255,0,255],
                6: [0,255,0],7: [170,120,200],8: [255,0,0],5:[10,200,100]}
ETH_Meta['label_names'] = [ETH_Meta['label2class'][l] for l in range(len(ETH_Meta['label2class']))]

