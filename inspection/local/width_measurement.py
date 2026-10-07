"""Reference PCA centreline and subpixel perpendicular widths."""
import cv2
import numpy as np

def moving_average(values,window):
    window=min(max(1,int(window)//2*2+1),max(1,len(values)//2*2-1))
    if window==1:return values.copy()
    return np.column_stack([np.convolve(np.pad(values[:,i],window//2,mode='edge'),np.ones(window)/window,mode='valid') for i in range(values.shape[1])])

def resample(points,spacing):
    arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(points,axis=0),axis=1))]
    valid=np.r_[True,np.diff(arc)>1e-8]
    if arc[-1]<1:raise ValueError('Strip centerline is too short.')
    distances=np.linspace(0,arc[-1],max(3,int(np.ceil(arc[-1]/spacing))+1))
    return np.column_stack([np.interp(distances,arc[valid],points[valid,i]) for i in (0,1)]),distances

def normals(points):
    tangent=np.gradient(points,axis=0)
    length=np.linalg.norm(tangent,axis=1)
    tangent/=np.maximum(length,1e-9)[:,None]
    return np.column_stack((-tangent[:,1],tangent[:,0]))

def boundary_rays(mask,points,directions,step=.25):
    """First bilinear 0.5 crossing, with subpixel interpolation on each ray."""
    field=np.float32(mask>0);h,w=field.shape
    def sample(p):return cv2.remap(field,p[:,0].astype(np.float32)[:,None],p[:,1].astype(np.float32)[:,None],cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)[:,0]
    start=sample(points);active=start>.5
    distance=np.full(len(points),np.nan);previous=start.copy()
    for t in np.arange(step,np.hypot(h,w)+step,step):
        indices=np.flatnonzero(active)
        if not len(indices):break
        positions=points[indices]+directions[indices]*t
        inside=(positions[:,0]>=0)&(positions[:,0]<=w-1)&(positions[:,1]>=0)&(positions[:,1]<=h-1)
        values=sample(positions)
        hit=inside&(values<=.5)
        good=indices[hit]
        denominator=previous[good]-values[hit]
        distance[good]=t-step+step*(previous[good]-.5)/np.maximum(denominator,1e-9)
        # Exiting the photo without crossing a real mask edge is not a measurement.
        active[indices[hit|~inside]]=False
        previous[indices]=values
    return distance

def measure_component(mask,spacing=5.,segments=10,tip_trim=0.,center_window=11,ray_step=.25):
    yy,xx=np.where(mask>0);pixels=np.column_stack((xx,yy)).astype(float)
    origin=pixels.mean(axis=0)
    _,_,axes=np.linalg.svd(pixels-origin,full_matrices=False)
    along=axes[0]
    if (along[1]<0 if abs(along[1])>=abs(along[0]) else along[0]<0):along=-along
    across=np.array([-along[1],along[0]])
    u=(pixels-origin)@along;v=(pixels-origin)@across
    bins=np.rint(u).astype(int);points=[]
    for key in np.unique(bins):
        cross=v[bins==key]
        points.append(origin+key*along+(cross.min()+cross.max())*.5*across)
    points=moving_average(np.asarray(points),center_window)
    points,arc=resample(points,spacing)
    # Recenter on the midpoint of actual perpendicular intersections twice.
    for _ in range(2):
        normal=normals(points)
        left=boundary_rays(mask,points,-normal,ray_step);right=boundary_rays(mask,points,normal,ray_step)
        valid=np.isfinite(left)&np.isfinite(right)
        points[valid]+=normal[valid]*((right[valid]-left[valid])/2)[:,None]
        points=moving_average(points,max(1,round(center_window/spacing)))
        points,arc=resample(points,spacing)
    normal=normals(points)
    left=boundary_rays(mask,points,-normal,ray_step);right=boundary_rays(mask,points,normal,ray_step)
    valid=np.isfinite(left)&np.isfinite(right)&(arc>=tip_trim)&(arc<=arc[-1]-tip_trim)
    low=float(tip_trim);high=float(arc[-1]-tip_trim)
    if high<=low:raise ValueError('Tip trim removes the entire strip centerline.')
    labels=np.clip(np.floor((arc-low)/(high-low)*segments).astype(int),0,segments-1)+1
    samples=[];summary=[]
    for i in range(len(points)):
        p=points[i];l=p-normal[i]*left[i];r=p+normal[i]*right[i]
        samples.append(dict(segment=int(labels[i]),arc_px=float(arc[i]),center_x=float(p[0]),center_y=float(p[1]),
                            valid=bool(valid[i]),left_px=float(left[i]) if np.isfinite(left[i]) else None,
                            right_px=float(right[i]) if np.isfinite(right[i]) else None,
                            width_px=float(left[i]+right[i]) if valid[i] else None,
                            left_x=float(l[0]) if np.isfinite(left[i]) else None,left_y=float(l[1]) if np.isfinite(left[i]) else None,
                            right_x=float(r[0]) if np.isfinite(right[i]) else None,right_y=float(r[1]) if np.isfinite(right[i]) else None))
    for segment in range(1,segments+1):
        widths=np.array([s['width_px'] for s in samples if s['segment']==segment and s['valid']])
        summary.append(dict(segment=segment,start_arc_px=low+(segment-1)*(high-low)/segments,end_arc_px=low+segment*(high-low)/segments,
                            sample_count=len(widths),average_width_px=float(widths.mean()) if len(widths) else None,
                            minimum_width_px=float(widths.min()) if len(widths) else None,maximum_width_px=float(widths.max()) if len(widths) else None))
    return dict(length_px=float(arc[-1]),samples=samples,segments=summary)
