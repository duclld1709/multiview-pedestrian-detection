"""Aspect-preserving display and mouse mapping for the legacy OpenCV canvas."""
import cv2
import numpy as np


class Viewport:
    def __init__(self):
        self.size = (1, 1)
        self.source_size = (1, 1)
        self.zoom = 1.0
        self.center = None
        self.scale = 1.0
        self.offset = np.zeros(2)
        self.pan_start = None

    def reset(self):
        self.zoom = 1.0
        self.center = None

    def transform(self):
        w,h = self.size
        sw,sh = self.source_size
        if self.center is None:
            self.center = np.array([sw/2,sh/2], dtype=float)
        self.scale = min(w/sw,h/sh)*self.zoom
        self.offset = np.array([w/2,h/2])-self.center*self.scale

    def source_point(self,x,y):
        return (np.array([x,y])-self.offset)/self.scale

    def zoom_at(self,x,y,factor):
        point = self.source_point(x,y)
        self.zoom = float(np.clip(self.zoom*factor,1,16))
        if self.zoom == 1:
            self.center = None
            self.transform()
        else:
            self.transform()
            self.center += point-self.source_point(x,y)
            self.transform()

    def draw(self,source,size):
        self.size = tuple(max(1,int(v)) for v in size)
        source_size = source.shape[1],source.shape[0]
        if source_size != self.source_size:
            self.source_size = source_size
            self.reset()
        self.transform()
        # Explicit area downsampling avoids the backend's low-quality resize.
        if self.scale < 1:
            sw,sh = self.source_size
            rw,rh = max(1,round(sw*self.scale)),max(1,round(sh*self.scale))
            reduced = cv2.resize(source,(rw,rh),interpolation=cv2.INTER_AREA)
            matrix = np.array([[self.scale*sw/rw,0,self.offset[0]],
                               [0,self.scale*sh/rh,self.offset[1]]],dtype=float)
            result = cv2.warpAffine(reduced,matrix,self.size,flags=cv2.INTER_LINEAR,
                                    borderValue=(18,18,18))
        else:
            matrix = np.array([[self.scale,0,self.offset[0]],
                               [0,self.scale,self.offset[1]]],dtype=float)
            result = cv2.warpAffine(source,matrix,self.size,flags=cv2.INTER_LANCZOS4,
                                    borderValue=(18,18,18))
        text = f'{self.scale*100:.0f}% source | Wheel: zoom | Middle drag: pan | 1: native | Home/0: fit'
        cv2.rectangle(result,(0,0),(min(self.size[0],760),27),(18,18,18),-1)
        cv2.putText(result,text,(8,19),cv2.FONT_HERSHEY_SIMPLEX,.47,(220,220,220),1,cv2.LINE_AA)
        return result
