import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import glob

import skimage.data
import matplotlib.pyplot as plt
# import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
# import selectivesearch
# import cv2
import numpy as np
import os.path
from PIL import Image
import subprocess
import timeit
import sys
from hgroup import hierarchical

def area(x):
    l=x[2]-x[0]
    b=x[3]-x[1]
    return l*b

def cal_iou(k,m,boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = (xB - xA + 1) * (yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    # iou = interArea / float(boxAArea + boxBArea - interArea)
    iou = interArea / min(float(boxAArea),float(boxBArea))

    # return the intersection over union value
    if iou>=.4:
        return k
    else:
        return m

def both(aff_mat,filt):
    flag=[]
    # large=[]
    new=[]
    for each in aff_mat:
        large=aff_mat.index(each)
        aff_mat[large]=100
        mx=area(filt[large])
        
        if each not in flag:
            for ea in aff_mat:
                if each==ea and mx<area(filt[aff_mat.index(ea)]):
                    mx=area(filt[aff_mat.index(ea)])
                    large=aff_mat.index(ea)
                if each==ea:
                    aff_mat[ea]=100
            new.append(large)
            flag.append(each)
    return new

def main(image_path):
    list=[]
    # loading astronaut image
    img = skimage.io.imread(image_path)
    #img = skimage.io.imread('/home/user/000654.jpg')
    #img = skimage.data.lenna()
    #img = '/home/a.jpg'
    ct=0
    # perform selective search
    # img_lbl, regions = selectivesearch.selective_search(img, scale=100, sigma=.9, min_size=200)
    img_lbl, regions = hierarchical(img, scale=600, sigma=.9, min_size=100)
    candidates = set()
    candidates1 = set()
    candidates3 = set()
    #print regions

    l=0
    # l1=0



    l1=0
    for r in regions:
        l1+=1
        # excluding same rectangle (with different segments)
        # if r['rect'] in candidates:
        #     continue
        # excluding regions smaller than 2000 pixels
        if r['size'] < 2000 :
            continue
        # distorted rects
        x, y, w, h = r['rect']
        if x==0 and y==0:
            continue
        if h/w>1:
            candidates.add(r['rect'])
        else:
            if x!=0 and y!=0:
                candidates1.add(r['rect'])
        # l+=1




    # draw rectangles on the original image
    t=0
    k=0
    # fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 10))
    # ax.imshow(img)

    filt=[[0,0,0,0] for i in range(len(candidates))]
    filt1=[[0,0,0,0] for i in range(len(candidates1))]

    for x, y, w, h in candidates:
        # print x, y, w, h
        w1=x+w
        h1=y+h
        x1=['red','green','blue','yellow','violet']
        p=t%5
        b=x1[p]
        t+=1
        #print b
        filt[k][0]=x
        filt[k][1]=y
        filt[k][2]=w1
        filt[k][3]=h1

        k+=1
        # rect = mpatches.Rectangle((x, y), w, h, fill=False, edgecolor=b, linewidth=3)
        # ax.add_patch(rect)
    k=0
    for x, y, w, h in candidates1:
        # print x, y, w, h
        w1=x+w
        h1=y+h
        x1=['red','green','blue','yellow','violet']
        p=t%5
        b=x1[p]
        t+=1
        #print b
        filt1[k][0]=x
        filt1[k][1]=y
        filt1[k][2]=w1
        filt1[k][3]=h1

        k+=1
        # rect = mpatches.Rectangle((x, y), w, h, fill=False, edgecolor=b, linewidth=3)
        # ax.add_patch(rect)

    # plt.show()



    # print "filt1 : ",filt
    # print "filt1 : ",filt1

    aff_mat=[i for i in range(len(filt))]
    aff_mat1=[i for i in range(len(filt1))]

    for i in range(len(filt)):
        if aff_mat[i]!=i:
            continue
        for j in range(len(filt)):
            if i==j or aff_mat[j]!=j:
                continue
            aff_mat[j]=cal_iou(i,j,filt[i],filt[j])

    #print aff_mat

    for i in range(len(filt1)):
        if aff_mat1[i]!=i:
            continue
        for j in range(len(filt1)):
            if i==j or aff_mat1[j]!=j:
                continue
            aff_mat1[j]=cal_iou(i,j,filt1[i],filt1[j])

    #print aff_mat1
    



    # aff_mat1=aff_mat.copy()
    # uniqe=[]
    # clusters=list(set(aff_mat))

    ########################################
    # fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 10))
    # ax.imshow(img)

    new=both(aff_mat,filt)
    new1=both(aff_mat1,filt1)


    #print new,new1
    z=[]
    z1=[]

    for each in new:
        w=filt[each][2]-filt[each][0]
        h=filt[each][3]-filt[each][1]
        x1=['blue','yellow','violet']
        p=t%3
        b=x1[p]
        t+=1
        x=filt[each][0]
        y=filt[each][1]
        z.append(filt[each][0])
        z.append(filt[each][1])
        z.append(filt[each][2])
        z.append(filt[each][3])

        # rect = mpatches.Rectangle((x, y), w, h, fill=False, edgecolor=b, linewidth=3)
        # ax.add_patch(rect)


    for each in new1:
        w=filt1[each][2]-filt1[each][0]
        h=filt1[each][3]-filt1[each][1]
        x1=['red','green']
        p=t%2
        b=x1[p]
        t+=1
        x=filt1[each][0]
        y=filt1[each][1]
        z1.append(filt1[each][0])
        z1.append(filt1[each][1])
        z1.append(filt1[each][2])
        z1.append(filt1[each][3])

        # rect = mpatches.Rectangle((x, y), w, h, fill=False, edgecolor=b, linewidth=3)
        # ax.add_patch(rect)


    # plt.show()
    return z,z1
def compute_single_labeled(files):
    for image_path in files:
        start_time = timeit.default_timer()
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(12, 12))
        head,tail = os.path.split(image_path)
        im=Image.open(image_path)
        width,height=im.size # (width,height) tuple


        # p3='deploy.prototxt'
        # p4='nin_imagenet_conv.caffemodel'
        # p5='imagenet_mean.binaryproto'

        # p6='labels_map.txt'
        # p7=image_path


        # # os.system(p1)
        # # ct+=1
        # print "**********************************************************************************************************************************************\n"
        # print "computing...."
        # start_time = timeit.default_timer()
        # pd = subprocess.Popen([p2,p3,p4,p5,p6,p7], stdout=subprocess.PIPE)
        # out,err=pd.communicate()
        # # print out
        # import re
        # # x=out.split('""')
        # x=re.findall(r"['\"](.*?)['\"]", out)
        p1='python'
        p2='d.py'

        p3='--image_file'
        p4=image_path
        p5='2>&1|tail'
        # p6='|'
        # p7='tail'
        p8='-1'

        # p6='labels_map.txt'
        # p7=img_path


        # os.system(p1)
        # ct+=1
        print "**********************************************************************************************************************************************\n"
        # print p1
        
        pd = subprocess.Popen([p1,p2,p3,p4], stdout=subprocess.PIPE)
        out,err=pd.communicate()
        # print out
        import re
        x=out.split('\n')
        # x=re.findall(r"['\"](.*?)['\"]", out)
        # code you want to evaluate
        
        print x

        class_name=x[0]
        # print out

        head1,tail1 = os.path.split(image_path)
        
        split_pt=tail1.split('.')
        # code you want to evaluate
        
        print x[0]

        class_name=x[0]
        x=10
        y=10
        w=width-20
        h=height-20
        input_file_name = os.path.basename(image_path)
        rect = mpatches.Rectangle(
            (x, y), w, h, fill=False, edgecolor='green', linewidth=3)
        ax.add_patch(rect)
        # score=0.1
        ax.text(x,y- 2,
        '{:s}'.format(class_name),
        bbox=dict(facecolor='blue', alpha=0.5),
        fontsize=16, color='white')


        elapsed = timeit.default_timer() - start_time
        thresh=.5


        print "Time taken for execution : ",elapsed

        name='Region detections'
        ax.set_title(('{} detections with '
            'p({} | box) >= {:.1f}').format(name, name,
            thresh),fontsize=20)
        plt.show()
        # out_file = "/home/user/Dropbox/project/{0}.png".format(input_file_name)

        # delete_similr = "__delete_similr" if DELETE_SIMILR_INCLUDE else ''
        ax.imshow(im)
        out_file = "/home/user/project/demo/output/{0}".format(
            # SELECTIVESEARCH_SCALE,
            # SELECTIVESEARCH_SIGMA,
            # SELECTIVESEARCH_MIN_SIZE,
            # delete_similr,
            input_file_name)
        dirname = os.path.dirname(out_file)
        if not os.path.exists(dirname):
            os.mkdir(dirname)
        plt.savefig(out_file)



if __name__ == "__main__":
    #files = glob.glob('/home/user/pictures/pic/*.jpg')
    #for image_path in files:
    # image_path='/home/user/Dropbox/project/000035.jpg'
    j=1
    arg=0
    if len(sys.argv)==1:
        files = glob.glob('/home/user/project/demo/pic/*.jpg')
    else:
        files = glob.glob('/home/user/project/demo/pic1/*.jpg')
        arg=1
        compute_single_labeled(files)
    if arg==0:
        for image_path in files:
            head,tail = os.path.split(image_path)
            # main(image_path)
            # image_path='/home/user/Desktop/pic/000740.jpg'


            print "Getting bboxes ......... ."


            im = Image.open(image_path)
            ct=0

            # setting timer
            start_time = timeit.default_timer()

            cd,cd1=main(image_path)
            path1="/home/user/project/demo/result/{0}/".format(tail)

            dirname = os.path.dirname(path1)
            if not os.path.exists(dirname):
                os.mkdir(dirname)
            # plt.savefig(out_file)

            for i in range(0,len(cd),4):
                # print "hai"
                im.crop((cd[i],cd[i+1],cd[i+2],cd[i+3])).save(path1+str(ct)+".jpg")
                ct+=1
                # print cd[i],cd[i+1],cd[i+2],cd[i+3]
            ct2=0
            for i in range(0,len(cd1),4):
                # print "hai"
                im.crop((cd1[i],cd1[i+1],cd1[i+2],cd1[i+3])).save(path1+str(ct) +".jpg")
                ct+=1
                ct2+=1
            # plt.savefig(path1+str(ct)+".jpg")
            print"Done---------------------------------------------------"

            ###########################################################################################
            path2=path1+"*.jpg"
            print path1
            # print path2
            files1 = glob.glob(path2)



            input_file_name = os.path.basename(image_path)
            # image_array = pre_process(image_path)
            fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(12, 12))
            # ax.imshow(image_array)
            for img_path in files1:
                
                p1='python'
                p2='d.py'

                p3='--image_file'
                p4=img_path
                p5='2>&1|tail'
                # p6='|'
                # p7='tail'
                p8='-1'

                # p6='labels_map.txt'
                # p7=img_path


                # os.system(p1)
                # ct+=1
                print "**********************************************************************************************************************************************\n"
                # print p1
                
                pd = subprocess.Popen([p1,p2,p3,p4], stdout=subprocess.PIPE)
                out,err=pd.communicate()
                # print out
                import re
                x=out.split('\n')
                # x=re.findall(r"['\"](.*?)['\"]", out)
                # code you want to evaluate
                
                print x

                class_name=x[0]
                # print out

                head1,tail1 = os.path.split(img_path)
                
                split_pt=tail1.split('.')


                # plt.show()

                # print split_pt
                if int(split_pt[0])>=(ct-ct2):
                    fn=int(split_pt[0])-(ct-ct2)
                    print "in second",fn

                    # visualize(cd[fn],cd[fn+1],cd[fn+2],cd[fn+3],image_path)
                    x=cd1[fn*4]
                    y=cd1[(fn*4)+1]
                    w=cd1[(fn*4)+2]-x
                    h=cd1[(fn*4)+3]-y
                    # for (x, y, w, h) in candidates:
                    rect = mpatches.Rectangle(
                        (x, y), w, h, fill=False, edgecolor='red', linewidth=2)
                    ax.add_patch(rect)
                    # score=0.3
                    ax.text(x,y- 2,
                    '{:s}'.format(class_name),
                    bbox=dict(facecolor='blue', alpha=0.5),
                    fontsize=14, color='white')


                else:
                    fn=int(split_pt[0])
                    print"in first",fn

                    # visualize(cd1[fn],cd1[fn+1],cd1[fn+2],cd1[fn+3],image_path)
                    x=cd[fn*4]
                    y=cd[(fn*4)+1]
                    w=cd[(fn*4)+2]-x
                    h=cd[(fn*4)+3]-y
                    # for (x, y, w, h) in candidates:
                    rect = mpatches.Rectangle(
                        (x, y), w, h, fill=False, edgecolor='green', linewidth=2)
                    ax.add_patch(rect)
                    score=0.1
                    ax.text(x,y- 2,
                    '{:s}'.format(class_name),
                    bbox=dict(facecolor='blue', alpha=0.5),
                    fontsize=14, color='white')


            elapsed = timeit.default_timer() - start_time
            thresh=.5


            print "Time taken for execution : ",elapsed

            name='Region detections'
            ax.set_title(('{} detections with '
                'p({} | box) >= {:.1f}').format(name, name,
                thresh),fontsize=20)
            plt.show()
            # out_file = "/home/user/Dropbox/project/{0}.png".format(input_file_name)

            # delete_similr = "__delete_similr" if DELETE_SIMILR_INCLUDE else ''
            ax.imshow(im)
            out_file = "/home/user/project/demo/output/{0}".format(
                # SELECTIVESEARCH_SCALE,
                # SELECTIVESEARCH_SIGMA,
                # SELECTIVESEARCH_MIN_SIZE,
                # delete_similr,
                input_file_name
                
            )
            
            dirname = os.path.dirname(out_file)
            if not os.path.exists(dirname):
                os.mkdir(dirname)
            plt.savefig(out_file)


