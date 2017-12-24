import os,csv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
mpcat40_fn = BASE_DIR+'/mpcat40.tsv'

label2class = {}
label2color = {}
label2WordNet_synset_key = {}
label2NYUv2_40label = {}
label25 = {}
label26 = {}


def hexcolor2int(hexcolor):
    hexcolor = hexcolor[1:]
    hex0 = '0x'+hexcolor[0:2]
    hex1 = '0x'+hexcolor[2:4]
    hex2 = '0x'+hexcolor[4:6]
    hex_int = [int(hex0,16),int(hex1,16),int(hex2,16)]
    return hex_int


with open(mpcat40_fn,'r') as f:
    reader = csv.reader(f,delimiter='\t')
    for i,line in enumerate(reader):
        if i==0: continue
        k = int(line[0])
        label2class[k] = line[1]
        label2color[k] = hexcolor2int( line[2] )
        label2WordNet_synset_key[k]=line[3]
        label2NYUv2_40label[k] = line[4]
        label25[k] = line[5]
        label26[k] = line[6]


print(label2class)
print(label2color)
print(label2WordNet_synset_key)
print(label2NYUv2_40label)
print(label25)
print(label26)
