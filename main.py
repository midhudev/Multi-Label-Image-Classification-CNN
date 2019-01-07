#!/usr/bin/env python

import sys
import time
from random import randrange
import Image
import ImageFilter
from filterfft import *
from disjointset import DisjointSet
from graph import Graph

K = 500
MinCC = 10
Sigma = 0.9


def tau(C):
    return K/float(C)

def segment(w, h, ds, g):
    dist = [tau(1.0)]*(w*h)
    me = 0
    for e in g.E:
        #print('Edge %d and %d = %f' % (e.v1, e.v2, e.dist))
        p1 = ds.find(e.v1)
        p2 = ds.find(e.v2)
        if p1 != p2:
            if e.dist <= min(dist[p1.key], dist[p2.key]):
                pn = ds.unionNode(p1, p2)
                dist[pn.key] = e.dist + tau(pn.size)
                #print('Merging %d and %d, sz = %d, tau = %lf, dist = %lf, tot = %lf' % (p1.key, p2.key, pn.size, tau(pn.size), e.dist, dist[pn.key]))
                me = me + 1
    #for i in range(w*h):
    #    print dist[i],
    #print('Total merges: ',me)

def postprocess(ds, g):
    for e in g.E:
        p1 = ds.find(e.v1)
        p2 = ds.find(e.v2)
        if p1 != p2:
            if p1.size < MinCC or p2.size < MinCC:
                ds.unionNode(p1, p2)

def randomColour(w, pix, ds,h):
    mat=[[0 for i in range(w)] for j in range(h) ]
    col = {}
    for (pp, node) in ds.data.items():
        # print ds.data.items()

        rep = ds.findNode(node)
        # print pp,rep
        if col.get(rep) == None:
            col[rep] = tuple([randrange(0, 255) for _ in node.data])
        (j,i) = (pp/w, pp%w)
        pix[i,j] = col[rep]
        mat[j][i]=col.keys().index(rep)
    # print col
    return len(col),mat

def main(im,K1,MinCC1,Sigma1):
    # if len(sys.argv) > 2:
    K = int(K1)
    # if len(sys.argv) > 3:
    MinCC = int(MinCC1)
    # if len(sys.argv) > 4:
    Sigma = float(Sigma1)

    # print('Processing image %s, K = %d' % (sys.argv[1], K))
    start = time.time()

    # Apply gaussian filter to all color channels separately
    im = Image.open('b.jpg')
    (width, height) = im.size
    l1=0
    # print im
    print('Image width = %d, height = %d' % (width, height))

    print('Blurring with Sigma = %f' % Sigma)
    source = im.split()
    blurred = []
    # print source
    print len(source)
    for c in range(len(source)):
        I = numpy.asarray(source[c])
        I = filter(I, gaussian(Sigma))
        blurred.append(Image.fromarray(numpy.uint8(I)))
    im = Image.merge(im.mode, tuple(blurred))
    # print blurred
    im.show()

    pix = im.load()

    # for j in range(height):
    #     l1=0
    #     for i in range(width):
    #         print pix[i,j]
    #         l1+=1
    #     print l1
    # imshow(pix)
    ds = DisjointSet(width*height)
    # print ds
    for j in range(height):
        for i in range(width):
            ds.makeSet(j*width + i, pix[i,j])
    # print ds

    print('Number of pixels: %d' % len(ds.data))
    g = Graph(width, height, pix)
    print('Number of edges in the graph: %d' % len(g.E))
    print('Time: %lf' % (time.time() - start))

    segstart = time.time()
    segment(width, height, ds, g)
    print('Segmentation done in %lf, found %d segments' % (time.time() - segstart, ds.num))

    print('Postprocessing small components, min = %d' % MinCC)
    postproc = time.time()
    postprocess(ds, g)
    print('Postprocessing done in %lf' % (time.time() - postproc))

    l,mat= randomColour(width, pix, ds,height)
    print mat
    print('Regions produced: %d' % l)

    print
    print('Time total: %lf' % (time.time() - start))

    im.show()
    # mat1=[[0 for i in range(height)] for j in range(width) ]
    mat1=[list(x) for x in zip(*mat)]    
    return mat
if __name__ == '__main__':
    img=1
    k=200
    mi=20
    sig=.9
    main(img,k,mi,sig)

