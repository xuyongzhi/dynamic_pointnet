from __future__ import print_function
import glob,os,sys
import zipfile,gzip
from plyfile import PlyData, PlyElement

MATTERPORT3D_ROOTDIR = '/home/y/DS/Matterport3D/Matterport3D_Whole'
SCANS_DIR = MATTERPORT3D_ROOTDIR+'/v1/scans'


def read_house(house_name):
    readzip_house_segmentation(SCANS_DIR,house_name)

def readzip_house_segmentation(scans_dir,house_name):
    house_dir = scans_dir+'/%s'%(house_name)
    house_segmentation_zip_fn = house_dir+'/house_segmentations.zip'
    hs_zf = zipfile.ZipFile(house_segmentation_zip_fn,'r')
    print (hs_zf.namelist())

    with hs_zf.open('%s/house_segmentations/%s.house'%(house_name,house_name)) as house_fo:
        for i,line in enumerate( house_fo ):
            if i<1:
                print(line)

    ply_path = house_dir+'/%s/house_segmentations/%s.ply'%(house_name,house_name)
    if not os.path.exists(ply_path):
        print('extracting ply...')
        ply_path_extracted  = hs_zf.extract('%s/house_segmentations/%s.ply'%(house_name,house_name),house_dir)
        print('ply extracting finished: %s'%(ply_path_extracted) )
    else:
        print('ply file already extracted: %s'%(ply_path))
    ply_fn = '/home/y/DS/Matterport3D/Matterport3D_Whole/v1/scans/17DRP5sb8fy/17DRP5sb8fy/house_segmentations/17DRP5sb8fy.ply'
    with open(ply_fn,'r') as fo:
        plydata = PlyData.read(fo)
        print(plydata.elements)

    import pdb; pdb.set_trace()  # XXX BREAKPOINT
    #print(house)


#def read_house_file()

def view_house(house_name):
    house_dir = SCANS_DIR+'/%s/%s'%(house_name,house_name)
    house_segmentation_dir = house_dir+'/house_segmentations'
    region_segmentations_dir = house_dir+'/region_segmentations'

    view_house_segmentation(house_segmentation_dir)


def view_house_segmentation(house_segmentation_dir):
    house_segmentation_fls = glob.glob(house_segmentation_dir+'/*.house')[0]
    print(house_segmentation_dir)
    print(house_segmentation_fls)


if __name__ == '__main__':
    house_name = '17DRP5sb8fy'
    #view_house(house_name)
    read_house(house_name)
