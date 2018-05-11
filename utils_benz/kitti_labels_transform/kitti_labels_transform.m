% Transform the KITTI labels data into format of dynamic_pointnet
% the format
% "category", "length", "width", "height", "alpha(ration_y)", "x", "y" and "z"

% reading the label data and calibration data
% project ryz in camera coordinate into velodney coordinate
% save the data accroding to above sequence
function [] = main()

clear all; close all;
root_dir = 'H:\Dataset\datasets\kitti\object\';
data_set = 'training';
cam = 2; % 2 = left color camera
image_dir = fullfile(root_dir,[data_set '\image_' num2str(cam)]);
velo_dir = [root_dir,data_set,'\velodyne\'];
label_dir = fullfile(root_dir,[data_set '\label_' num2str(cam)]);
calib_dir = fullfile(root_dir,[data_set '\calib']);
% % % load data
calib = dir(fullfile(calib_dir,'*.txt'));   %7518 files
ima   = dir(fullfile(image_dir,'*.png'));    %7518 files

%% output dir
output_dir = 'C:\Users\z5108714\Desktop\output';

fst_frame =0; nt_frames = length(ima);
%% 
for frame = fst_frame: 1: nt_frames
    objects = readLabels(label_dir, frame);
    fid = fopen(sprintf('%s/%06d.txt',output_dir,frame),'w');
    
     for obj_idx=1:numel(objects)
        if  strcmp(objects(obj_idx).type , 'Car')
             T = Fun_open_calib(calib(frame+1).name, calib_dir);
             ctr_img = [objects(obj_idx).t(1)  objects(obj_idx).t(2)  objects(obj_idx).t(3) 1];
             %crt_pc   =  (inv(T.Tr_velo_to_cam)*inv(T.R0_rect)*inv(T.P2)*ctr_img')';
             crt_pc   =  (inv(T.Tr_velo_to_cam)*ctr_img')';
             fprintf(fid,'%s %.2f %.2f %.2f %.2f %.2f %.2f %.2f',...
                              objects(obj_idx).type,   objects(obj_idx).l,   objects(obj_idx).w,   objects(obj_idx).h,   abs(wrapToPi(objects(obj_idx).ry)),   crt_pc(1),   crt_pc(2),  crt_pc(3)) ;
             fprintf(fid,'\n');
        end
     end
     fclose(fid);
     
end
end

function alpha = wrapToPi(alpha)

% wrap to [0..2*pi]
alpha = mod(alpha,2*pi);

% wrap to [-pi..pi]
idx = alpha>pi;
alpha(idx) = alpha(idx)-2*pi;
end