""" Original Author: Haoqiang Fan """
import numpy as np
import ctypes as ct
import cv2
import sys
import os
from pprint import pprint
import argparse
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
showsz=800
mousex,mousey=0.5,0.5
zoom=1.0
changed=True
def onmouse(*args):
    global mousex,mousey,changed
    y=args[1]
    x=args[2]
    mousex=x/float(showsz)
    mousey=y/float(showsz)
    changed=True
cv2.namedWindow('show3d')
cv2.moveWindow('show3d',0,0)
cv2.setMouseCallback('show3d',onmouse)

dll=np.ctypeslib.load_library(os.path.join(BASE_DIR, 'render_balls_so'),'.')

# data_0_loop_1_idx_0_label_5_label_b_8_radius_0.149_mixed_lam_0.0263671875
# data_0_loop_1_idx_0_label_5_original
# data_0_loop_1_idx_0_label_8_original_2
# dataknn_0_loop_1_idx_0_label_5_label_b_8_radius_0.149_mixed_lam_0.150390625
# datamaska_0_loop_1_idx_0_lenidx_997_label_5_radius_0.149_mixed_lam_0.0263671875
# datamaskaknn_0_loop_1_idx_0_lenidx_870_label_5_radius_0.149_mixed_lam_0.150390625
# datamaskb_0_loop_1_idx_0_lenidx_989_label_b_8_radius_0.149_mixed_lam_0.0263671875
# datamaskbknn_0_loop_1_idx_0_lenidx_870_label_b_8_radius_0.149_mixed_lam_0.150390625

def showpoints(xyz,c_gt=None, c_pred = None ,waittime=0,showrot=False,magnifyBlue=0,
               freezerot=False,background=(0,0,0),normalizecolor=True, ballradius=10, mixed_color=False, 
               lam=0.0, save_name='_', lenidx=1024, mixed_gray=False, mask_a=False, mask_b=False, 
               part_a=False, part_b=False, save_path='./'):
    global showsz,mousex,mousey,zoom,changed
    xyz=xyz-xyz.mean(axis=0)
    radius=((xyz**2).sum(axis=-1)**0.5).max()
    xyz/=(radius*2.2)/showsz
    # if c_gt is None:
    #     c0=np.zeros((len(xyz),),dtype='float32')+255 # Green
    #     c1=np.zeros((len(xyz),),dtype='float32')+255 # Red
    #     c2=np.zeros((len(xyz),),dtype='float32')+255 # Blue
    # else:
    #     c0=c_gt[:,0]
    #     c1=c_gt[:,1]
    #     c2=c_gt[:,2]
    '''
        For mask, part, etc visualize from here
    '''
    print("lenidx : ",lenidx)
    print("len xyz : ",len(xyz))
    print("lam - show point : ",lam)
    print("xyz shape : ",xyz.shape)
    print("len(xyz)*lam) : ",len(xyz)*lam)
    print("np.around(len(xyz)*lam) : ",np.around(len(xyz)*lam))
    print("int(np.around(len(xyz)*lam)) : ",int(np.around(len(xyz)*lam)))
    if args.ball_mix: # mix, part_a, part_b from ball
        if args.mask_a or args.mask_b: # mask a mask b
            if args.mask_a:
                c0 = np.zeros((len(xyz),),dtype='float32')
                c1=np.zeros((len(xyz),),dtype='float32')
                c2=np.zeros((len(xyz),),dtype='float32')
                # green and red example
                mask_a = lenidx
                c0[:mask_a] += 200
                c0[mask_a:] += 200
                c1[:mask_a] += 200
                c1[mask_a:] += 200
                c2[:mask_a] += 0
                c2[mask_a:] += 200
            elif args.mask_b:
                c0 = np.zeros((len(xyz),),dtype='float32')
                c1=np.zeros((len(xyz),),dtype='float32')
                c2=np.zeros((len(xyz),),dtype='float32')
                # green and red example
                mask_b = lenidx
                # c0[:mask_b] += 200 # purple
                # c0[mask_b:] += 0
                # c1[:mask_b] += 200
                # c1[mask_b:] += 200
                # c2[:mask_b] += 200
                # c2[mask_b:] += 200
                c0[:mask_b] += 200 # yellow
                c0[mask_b:] += 200
                c1[:mask_b] += 200
                c1[mask_b:] += 200
                c2[:mask_b] += 200
                c2[mask_b:] += 0
        else: # mixed
            if args.part_a: # part a
                c0 = np.zeros((len(xyz),),dtype='float32')
                c1=np.zeros((len(xyz),),dtype='float32')
                c2=np.zeros((len(xyz),),dtype='float32')
                # green and red example
                c0[:] += 255
                # c0[original_point_num:] += 255
                c1[:] += 0
                # c1[original_point_num:] += 255
                c2[:] += 0
                # c2[original_point_num:] += 255
            elif args.part_b: # part 
                c0 = np.zeros((len(xyz),),dtype='float32')
                c1=np.zeros((len(xyz),),dtype='float32')
                c2=np.zeros((len(xyz),),dtype='float32')
                # green and red example
                # c0[:original_point_num] += 255
                c0[:] += 0
                # c1[:original_point_num] += 255
                c1[:] += 255
                # c2[:original_point_num] += 255
                c2[:] += 0
            else:
                if mixed_gray:
                    c0 = np.zeros((len(xyz),),dtype='float32')
                    c1=np.zeros((len(xyz),),dtype='float32')
                    c2=np.zeros((len(xyz),),dtype='float32')
                    # green and red example
                    original_point_num = len(xyz)-int(np.around(len(xyz)*lam))
                    c0[:original_point_num] += 200
                    c0[original_point_num:] += 200
                    c1[:original_point_num] += 200
                    c1[original_point_num:] += 200
                    c2[:original_point_num] += 200
                    c2[original_point_num:] += 200
                else:
                    c0 = np.zeros((len(xyz),),dtype='float32')
                    c1=np.zeros((len(xyz),),dtype='float32')
                    c2=np.zeros((len(xyz),),dtype='float32')
                    # green and red example
                    original_point_num = len(xyz)-int(np.around(len(xyz)*lam))
                    c0[:original_point_num] += 255
                    c0[original_point_num:] += 0
                    c1[:original_point_num] += 0
                    c1[original_point_num:] += 255
                    c2[:original_point_num] += 0
                    c2[original_point_num:] += 0
            
    elif args.ori: # original a
        if c_gt is None:
            c0=np.zeros((len(xyz),),dtype='float32')+255 # Green
            c1=np.zeros((len(xyz),),dtype='float32')+255 # Red
            c2=np.zeros((len(xyz),),dtype='float32')+255 # Blue      ==> White with 3
        else:
            c0=c_gt[:,0]
            c1=c_gt[:,1]
            c2=c_gt[:,2]
        
    elif args.ori2: # original b
        if c_gt is None:
            c0=np.zeros((len(xyz),),dtype='float32')+255 # Green
            c1=np.zeros((len(xyz),),dtype='float32')+255 # Red
            c2=np.zeros((len(xyz),),dtype='float32')+255 # Blue      ==> White with 3
        else:
            c0=c_gt[:,0]
            c1=c_gt[:,1]
            c2=c_gt[:,2]
    
    elif args.knn_mix: # mask_a, mask_b from knn, part_a, part_b from knn        
        if args.mask_a:
            c0 = np.zeros((len(xyz),),dtype='float32')
            c1=np.zeros((len(xyz),),dtype='float32')
            c2=np.zeros((len(xyz),),dtype='float32')
            # green and red example
            mask_a = lenidx
            c0[:mask_a] += 200
            c0[mask_a:] += 255
            c1[:mask_a] += 100
            c1[mask_a:] += 255
            c2[:mask_a] += 200
            c2[mask_a:] += 255           
        elif args.mask_b:
            c0 = np.zeros((len(xyz),),dtype='float32')
            c1=np.zeros((len(xyz),),dtype='float32')
            c2=np.zeros((len(xyz),),dtype='float32')
            # green and red example
            mask_b = lenidx
            c0[:mask_b] += 255
            c0[mask_b:] += 150
            c1[:mask_b] += 255
            c1[mask_b:] += 250
            c2[:mask_b] += 255
            c2[mask_b:] += 150
        if args.part_a:
            c0 = np.zeros((len(xyz),),dtype='float32')
            c1=np.zeros((len(xyz),),dtype='float32')
            c2=np.zeros((len(xyz),),dtype='float32')
            # green and red example
            # mask_a = lenidx
            c0[:] += 190
            # c0[mask_a:] += 255
            c1[:] += 0
            # c1[mask_a:] += 255
            c2[:] += 130
            # c2[mask_a:] += 255
        elif args.part_b:
            c0 = np.zeros((len(xyz),),dtype='float32')
            c1=np.zeros((len(xyz),),dtype='float32')
            c2=np.zeros((len(xyz),),dtype='float32')
            # green and red example
            # mask_b = lenidx
            # c0[:mask_b] += 255
            c0[:] += 130
            # c1[:mask_b] += 255
            c1[:] += 190
            # c2[:mask_b] += 255
            c2[:] += 0
        else:
            if mixed_gray:
                    c0 = np.zeros((len(xyz),),dtype='float32')
                    c1=np.zeros((len(xyz),),dtype='float32')
                    c2=np.zeros((len(xyz),),dtype='float32')
                    # green and red example
                    original_point_num = len(xyz)-int(np.around(len(xyz)*lam))
                    c0[:original_point_num] += 200
                    c0[original_point_num:] += 200
                    c1[:original_point_num] += 200
                    c1[original_point_num:] += 200
                    c2[:original_point_num] += 200
                    c2[original_point_num:] += 200
            else:
                c0 = np.zeros((len(xyz),),dtype='float32')
                c1=np.zeros((len(xyz),),dtype='float32')
                c2=np.zeros((len(xyz),),dtype='float32')
                # green and red example
                mask_a = lenidx
                c0[:mask_a] += 190
                c0[mask_a:] += 130
                c1[:mask_a] += 0
                c1[mask_a:] += 190
                c2[:mask_a] += 130
                c2[mask_a:] += 0  
    else:
        raise ValueError('Invalid arguments. Please input args to notice what kind of view.')
    # '''
    #     To here
    # '''
    # if mixed_color:
    #     c0 = np.zeros((len(xyz),),dtype='float32')
    #     c1=np.zeros((len(xyz),),dtype='float32')
    #     c2=np.zeros((len(xyz),),dtype='float32')
    #     # green and red example
    #     original_point_num = len(xyz)-int(np.around(len(xyz)*lam))
    #     c0[:original_point_num] += 255
    #     c0[original_point_num:] += 0
    #     c1[:original_point_num] += 0
    #     c1[original_point_num:] += 255
    #     c2[:original_point_num] += 0
        # c2[original_point_num:] += 0
    ###------------------------------------------------------------
    if normalizecolor:
        c0/=(c0.max()+1e-14)/255.0
        c1/=(c1.max()+1e-14)/255.0
        c2/=(c2.max()+1e-14)/255.0

    c0=np.require(c0,'float32','C')
    c1=np.require(c1,'float32','C')
    c2=np.require(c2,'float32','C')

    show=np.zeros((showsz,showsz,3),dtype='uint8')
    def render():
        rotmat=np.eye(3)
        if not freezerot:
            xangle=(mousey-0.5)*np.pi*1.2
        else:
            xangle=0
        rotmat=rotmat.dot(np.array([
            [1.0,0.0,0.0],
            [0.0,np.cos(xangle),-np.sin(xangle)],
            [0.0,np.sin(xangle),np.cos(xangle)],
            ]))
        if not freezerot:
            yangle=(mousex-0.5)*np.pi*1.2
        else:
            yangle=0
        rotmat=rotmat.dot(np.array([
            [np.cos(yangle),0.0,-np.sin(yangle)],
            [0.0,1.0,0.0],
            [np.sin(yangle),0.0,np.cos(yangle)],
            ]))
        rotmat*=zoom
        nxyz=xyz.dot(rotmat)+[showsz/2,showsz/2,0]

        ixyz=nxyz.astype('int32')
        show[:]=background
        dll.render_ball(
            ct.c_int(show.shape[0]),
            ct.c_int(show.shape[1]),
            show.ctypes.data_as(ct.c_void_p),
            ct.c_int(ixyz.shape[0]),
            ixyz.ctypes.data_as(ct.c_void_p),
            c0.ctypes.data_as(ct.c_void_p),
            c1.ctypes.data_as(ct.c_void_p),
            c2.ctypes.data_as(ct.c_void_p),
            ct.c_int(ballradius)
        )

        if magnifyBlue>0:
            show[:,:,0]=np.maximum(show[:,:,0],np.roll(show[:,:,0],1,axis=0))
            if magnifyBlue>=2:
                show[:,:,0]=np.maximum(show[:,:,0],np.roll(show[:,:,0],-1,axis=0))
            show[:,:,0]=np.maximum(show[:,:,0],np.roll(show[:,:,0],1,axis=1))
            if magnifyBlue>=2:
                show[:,:,0]=np.maximum(show[:,:,0],np.roll(show[:,:,0],-1,axis=1))
        if showrot:
            cv2.putText(show,'xangle %d'%(int(xangle/np.pi*180)),(30,showsz-30),0,0.5,cv2.cv.CV_RGB(255,0,0))
            cv2.putText(show,'yangle %d'%(int(yangle/np.pi*180)),(30,showsz-50),0,0.5,cv2.cv.CV_RGB(255,0,0))
            cv2.putText(show,'zoom %d%%'%(int(zoom*100)),(30,showsz-70),0,0.5,cv2.cv.CV_RGB(255,0,0))
    changed=True
    while True:
        if changed:
            render()
            changed=False
        cv2.imshow('show3d',show)
        if waittime==0:
            cmd=cv2.waitKey(10) % 256
        else:
            cmd=cv2.waitKey(waittime) % 256
        if cmd==ord('q'):
            break
        elif cmd==ord('Q'):
            sys.exit(0)

        if cmd==ord('t') or cmd == ord('p'):
            if cmd == ord('t'):
                if c_gt is None:
                    c0=np.zeros((len(xyz),),dtype='float32')+255
                    c1=np.zeros((len(xyz),),dtype='float32')+255
                    c2=np.zeros((len(xyz),),dtype='float32')+255
                else:
                    c0=c_gt[:,0]
                    c1=c_gt[:,1]
                    c2=c_gt[:,2]
            else:
                if c_pred is None:
                    c0=np.zeros((len(xyz),),dtype='float32')+255
                    c1=np.zeros((len(xyz),),dtype='float32')+255
                    c2=np.zeros((len(xyz),),dtype='float32')+255
                else:
                    c0=c_pred[:,0]
                    c1=c_pred[:,1]
                    c2=c_pred[:,2]
            if normalizecolor:
                c0/=(c0.max()+1e-14)/255.0
                c1/=(c1.max()+1e-14)/255.0
                c2/=(c2.max()+1e-14)/255.0
            c0=np.require(c0,'float32','C')
            c1=np.require(c1,'float32','C')
            c2=np.require(c2,'float32','C')
            changed = True



        if cmd==ord('n'):
            zoom*=1.1
            changed=True
        elif cmd==ord('m'):
            zoom/=1.1
            changed=True
        elif cmd==ord('r'):
            zoom=1.0
            changed=True
        elif cmd==ord('s'):
            # img_name = str(datetime.now())+'_'+save_name+'.png'
            img_name = os.path.join(save_path,str(datetime.now())+'_'+save_name+'.png')
            cv2.imwrite(img_name,show)
            # cv2.imwrite('show3d.png',show)
        if waittime!=0:
            break
    return cmd
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ball_radius', type=int, default=10, help='Initial learning rate [default: 0.001]')
    parser.add_argument('--path', default='./data_mixed', help='mixed data dir [default: ./data_mixed]')
    parser.add_argument('--show_rot', action='store_true', help='mix_data_save')
    parser.add_argument('--freeze_rot', action='store_true', help='mix_data_save')
    parser.add_argument('--not_normalize_color', action='store_false', help='mix_data_save')
    parser.add_argument('--background_white', action='store_true', help='mix_data_save')
    parser.add_argument('--mixed_color', action='store_true', help='mix_data_save')
    parser.add_argument('--ball_mix', action='store_true', help='mix_data_save')
    parser.add_argument('--knn_mix', action='store_true', help='mix_data_save')
    parser.add_argument('--mask_a', action='store_true', help='mask_view')
    parser.add_argument('--mask_b', action='store_true', help='mask_view')
    parser.add_argument('--part_a', action='store_true', help='part_view')
    parser.add_argument('--part_b', action='store_true', help='part_view')
    parser.add_argument('--ori', action='store_true', help='origin')
    parser.add_argument('--ori2', action='store_true', help='origin')
    parser.add_argument('--gray', action='store_true', help='origin')
    parser.add_argument('--save_path', default='./', help='mixed data dir [default: ./data_mixed]')
    
    
# data_0_loop_1_idx_0_label_5_label_b_8_radius_0.149_mixed_lam_0.0263671875
# data_0_loop_1_idx_0_label_5_original
# data_0_loop_1_idx_0_label_8_original_2
# dataknn_0_loop_1_idx_0_label_5_label_b_8_radius_0.149_mixed_lam_0.150390625
# datamaska_0_loop_1_idx_0_lenidx_997_label_5_radius_0.149_mixed_lam_0.0263671875
# datamaskaknn_0_loop_1_idx_0_lenidx_870_label_5_radius_0.149_mixed_lam_0.150390625
# datamaskb_0_loop_1_idx_0_lenidx_989_label_b_8_radius_0.149_mixed_lam_0.0263671875
# datamaskbknn_0_loop_1_idx_0_lenidx_870_label_b_8_radius_0.149_mixed_lam_0.150390625
    
    
    args = parser.parse_args()

    np.random.seed(100)
    # if len(sys.argv) < 2:
        # exit('Enter pointcloud path')
    # path = sys.argv[1]
    print("path : ",args.path)
    if args.ball_mix: # mix, part_a, part_b from ball
        if args.mask_a or args.mask_b:
            lam = float(os.path.basename(os.path.splitext(args.path.split('/')[-1])[0]).split('_')[-1])
            print("lam : ",lam)
            lenidx = os.path.basename(os.path.splitext(args.path.split('/')[-1])[0]).split('_')[7]
            filename_label = os.path.basename(os.path.splitext(args.path.split('/')[-1])[0]).split('_')[9]
            if args.mask_a:
                filename = 'lenidx_'+lenidx+'_label_'+filename_label+'_lam_'+str(lam)+'_mask_a_ball'
            elif args.mask_b:
                filename = 'lenidx_'+lenidx+'_label_'+filename_label+'_lam_'+str(lam)+'_mask_b_ball'
        else:
            lenidx = 1024
            lam = float(os.path.basename(os.path.splitext(args.path.split('/')[-1])[0]).split('_')[-1])
            print("lam : ",lam)
            filename_label_a = os.path.basename(os.path.splitext(args.path.split('/')[-1])[0]).split('_')[7]
            filename_label_b = os.path.basename(os.path.splitext(args.path.split('/')[-1])[0]).split('_')[10]
            filename = 'label_a_'+filename_label_a+'_label_b_'+filename_label_b+'_lam_'+str(lam)+'_mix_ball'
            if args.part_a:
                filename = 'label_a_'+filename_label_a+'_label_b_'+filename_label_b+'_lam_'+str(lam)+'_part_a_ball'
            elif args.part_b:
                filename = 'label_a_'+filename_label_a+'_label_b_'+filename_label_b+'_lam_'+str(lam)+'_part_b_ball'
            
    elif args.ori: # original a
        lam = 0.0
        lenidx = 1024
        filename_label_a = os.path.basename(os.path.splitext(args.path.split('/')[-1])[0]).split('_')[7]
        filename = 'label_a_'+filename_label_a+'_original'
        
    elif args.ori2: # original b
        lam = 0.0
        lenidx = 1024
        filename_label_b = os.path.basename(os.path.splitext(args.path.split('/')[-1])[0]).split('_')[7]
        filename = 'label_b_'+filename_label_b+'_original_2'
    
    elif args.knn_mix: # mask_a, mask_b from knn, part_a, part_b from knn        
        lam = float(os.path.basename(os.path.splitext(args.path.split('/')[-1])[0]).split('_')[-1])
        print("lam : ",lam)
        lenidx = os.path.basename(os.path.splitext(args.path.split('/')[-1])[0]).split('_')[7]
        filename_label = os.path.basename(os.path.splitext(args.path.split('/')[-1])[0]).split('_')[9]
        if args.mask_a:
            filename = 'lenidx_'+lenidx+'_label_'+filename_label+'_lam_'+str(lam)+'_mask_a_knn'
        elif args.mask_b:
            filename = 'lenidx_'+lenidx+'_label_'+filename_label+'_lam_'+str(lam)+'_mask_b_knn'
        if args.part_a:
            filename = 'lenidx_'+lenidx+'_label_b_'+filename_label+'_lam_'+str(lam)+'_part_a_knn'
        elif args.part_b:
            filename = 'lenidx_'+lenidx+'_label_b_'+filename_label+'_lam_'+str(lam)+'_part_b_knn'
        else:
            filename_label_a = os.path.basename(os.path.splitext(args.path.split('/')[-1])[0]).split('_')[7]
            filename_label_b = os.path.basename(os.path.splitext(args.path.split('/')[-1])[0]).split('_')[10]
            lenidx = int(np.trunc(1024*(1-lam)))
            filename = 'label_a'+filename_label_a+'_label_b_'+filename_label_b+'_lam_'+str(lam)+'_lendix_'+str(lenidx)+'_mix_knn'
            
    else:
        raise ValueError('Invalid arguments. Please input args to notice what kind of view.')
    
    lenidx = int(lenidx)
    point_set = np.loadtxt(args.path ,delimiter=',').astype(np.float32)
    # random_idx = np.random.randint(point_set.shape[0], size=1024)
    print("point set shape : ",point_set.shape)
    if args.ball_mix:
        part_len_b = int(np.around(len(point_set)*lam))
        part_len_a = len(point_set)-part_len_b
        if args.part_a:
            point_set = point_set[0:part_len_a,0:3]
        elif args.part_b:
            point_set = point_set[part_len_a:,0:3]
        else:
            point_set = point_set[0:1024,0:3]
    elif args.knn_mix:
        if args.part_a:
            point_set = point_set[0:lenidx,0:3]
        elif args.part_b:
            point_set = point_set[lenidx:,0:3]
        else:
            point_set = point_set[0:1024,0:3]
    else:
        point_set = point_set[0:1024,0:3]

    #point_set = point_set[random_idx,0:3]
    #pprint(point_set)
    #pprint(np.random.randn(2500,3))
    if args.background_white:
        c_background = (255, 255, 255)
    else:
        c_background = (0, 0, 0)
    #showpoints(np.random.randn(2500,3))
    showpoints(point_set, showrot=args.show_rot, freezerot=args.freeze_rot,background=c_background,
               normalizecolor=args.not_normalize_color,ballradius=args.ball_radius, mixed_color=args.mixed_color, lam=lam, save_name=filename, lenidx=lenidx, mixed_gray=args.gray, 
               mask_a=args.mask_a, mask_b=args.mask_b, part_a=args.part_a, part_b=args.part_b, save_path=args.save_path)


