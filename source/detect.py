"""Matched fixed-ROI baseline using frozen Grounding DINO on the same tail frames."""
import json, re, sys, time
import numpy as np
import torch
from PIL import Image
from common import ROOT,OEV,read,dump,sha
sys.path.insert(0,str(OEV/'cache/deps'))
from transformers import AutoProcessor,AutoModelForZeroShotObjectDetection
from extract import mask_box

def words(s):return set(re.findall('[a-z]+',s.lower()))-{'a','the','of'}
def main():
    torch.set_num_threads(4);start=time.monotonic()
    modeldir=OEV/'cache/grounding-dino-tiny'
    processor=AutoProcessor.from_pretrained(str(modeldir),local_files_only=True)
    processor.image_processor.size={'shortest_edge':480,'longest_edge':854}
    model=AutoModelForZeroShotObjectDetection.from_pretrained(str(modeldir),local_files_only=True,disable_custom_kernels=True).cuda().eval()
    model.requires_grad_(False)
    out=ROOT/'roi';out.mkdir(exist_ok=True)
    videos=read(ROOT/'feature-index.json')
    for iv,v in enumerate(videos):
        path=out/(v['video']+'.json')
        if path.exists():continue
        geom={e['uid']:e['geometry'] for e in v['entries'] if e['type']=='tail'}
        requests=[];seen=set()
        for r in v['rows']:
            for uid in r['frame_uids']:
                key=(uid,tuple(r['objects']))
                if key not in seen:requests.append((uid,r['objects']));seen.add(key)
        records=[]
        for i in range(0,len(requests),4):
            batch=requests[i:i+4]
            ims=[Image.open(ROOT/'cache/tail-images'/(uid+'.jpg')).convert('RGB') for uid,_ in batch]
            prompts=[' . '.join(n.lower() for n in names)+' .' for _,names in batch]
            inp=processor(images=ims,text=prompts,padding=True,return_tensors='pt').to('cuda')
            with torch.no_grad():pred=model(**inp)
            dets=processor.post_process_grounded_object_detection(pred,inp.input_ids,box_threshold=.25,text_threshold=.2,
                target_sizes=[(im.height,im.width) for im in ims])
            for (uid,names),im,d in zip(batch,ims,dets):
                boxes=[{'label':l,'score':float(s),'box':b.cpu().tolist()} for l,s,b in zip(d['labels'],d['scores'],d['boxes'])]
                masks=[];chosen=[]
                for name in names:
                    nw=words(name);candidates=[]
                    for b in boxes:
                        bw=words(b['label']);overlap=len(nw&bw)/max(1,len(nw|bw))
                        if overlap>=.5:candidates.append((overlap,b['score'],b))
                    if candidates:
                        b=max(candidates,key=lambda x:(x[0],x[1]))[2]
                        xy=np.clip(np.array(b['box'])/np.array([im.width,im.height,im.width,im.height]),0,1)
                        m=mask_box(xy,geom[uid]);chosen.append(b)
                    else:m=np.zeros(576,np.float32);chosen.append(None)
                    masks.append(m)
                records.append({'uid':uid,'names':names,'masks':[m.tolist() for m in masks],'selected':chosen,'detections':boxes})
            del inp,pred
        dump(path,records)
        status={'phase':'fixed_roi','completed_videos':iv+1,'total_videos':len(videos),'seconds':time.monotonic()-start}
        dump(ROOT/'status.json',status);print(json.dumps(status),flush=True)
    dump(ROOT/'roi-signature.json',{'source_sha256':sha(__file__),'weights_sha256':sha(modeldir/'model.safetensors'),
        'box_threshold':.25,'text_threshold':.2,'shortest_edge':480,'longest_edge':854,
        'queries':'training-only majority object pair, one prompt containing both; name-token Jaccard >= 0.5',
        'fallback':'uniform over image-content patches if a slot has no qualifying box'})
    print('DETECTION_COMPLETE',flush=True)

if __name__=='__main__':main()
