"""Engineering visual check of exact normal teacher frames and transformed boxes."""
import cv2,numpy as np
from common import ROOT,PROJECT,read
from extract import letterbox

def main():
    tiles=[];meta=[]
    teachers=read(ROOT/'supervision.json')
    for task in ['quesadilla','pinwheels']:
        archive=np.load(PROJECT/f'code/AEM/data/egoper/effect_frames/effect_frames_{task}.npz')
        chosen=[t for t in teachers if t['task']==task and t['split']=='train'][:2]
        for t in chosen:
            image,geom=letterbox(archive[t['teacher_image_path']])
            im=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
            for slot,b in enumerate(t['boxes_cxcywh_normalized']):
                if b is None:continue
                cx,cy,w,h=b
                x0,y0,x1,y1=[(cx-w/2)*geom['ww']+geom['dx'],(cy-h/2)*geom['hh']+geom['dy'],
                    (cx+w/2)*geom['ww']+geom['dx'],(cy+h/2)*geom['hh']+geom['dy']]
                color=(0,255,0) if slot==0 else (0,140,255)
                cv2.rectangle(im,(round(x0),round(y0)),(round(x1),round(y1)),color,2)
                cv2.putText(im,t['names'][slot],(max(1,round(x0)),max(14,round(y0)-4)),cv2.FONT_HERSHEY_SIMPLEX,.42,color,1)
            cv2.putText(im,f'{task} step {t["step"]}',(5,16),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1)
            tiles.append(im);meta.append(t['id'])
    out=np.concatenate([np.concatenate(tiles[:2],1),np.concatenate(tiles[2:],1)],0)
    cv2.imwrite(str(ROOT/'build/normal-teacher-qa.jpg'),out)
    print(meta)

if __name__=='__main__':main()
