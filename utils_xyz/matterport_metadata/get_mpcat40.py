import os,csv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
mpcat40_fn = BASE_DIR+'/mpcat40.tsv'



def hexcolor2int(hexcolor):
    hexcolor = hexcolor[1:]
    hex0 = '0x'+hexcolor[0:2]
    hex1 = '0x'+hexcolor[2:4]
    hex2 = '0x'+hexcolor[4:6]
    hex_int = [int(hex0,16),int(hex1,16),int(hex2,16)]
    return hex_int

MatterportMeta = {}
MatterportMeta['label2class'] = {}
MatterportMeta['label2color'] = {}
MatterportMeta['label2WordNet_synset_key'] = {}
MatterportMeta['label2NYUv2_40label'] = {}
MatterportMeta['label25'] = {}
MatterportMeta['label26'] = {}

with open(mpcat40_fn,'r') as f:
    reader = csv.reader(f,delimiter='\t')
    for i,line in enumerate(reader):
        if i==0: continue
        k = int(line[0])
        MatterportMeta['label2class'][k] = line[1]
        MatterportMeta['label2color'][k] = hexcolor2int( line[2] )
        MatterportMeta['label2WordNet_synset_key'][k]=line[3]
        MatterportMeta['label2NYUv2_40label'][k] = line[4]
        MatterportMeta['label25'][k] = line[5]
        MatterportMeta['label26'][k] = line[6]



#print(label2class)
#print(label2color)
#print(label2WordNet_synset_key)
#print(label2NYUv2_40label)
#print(label25)
#print(label26)
